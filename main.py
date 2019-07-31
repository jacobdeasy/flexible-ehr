import argparse, logging, numpy as np, os, sys, torch

from flexehr import Trainer
from flexehr.utils.modelIO import load_metadata, load_model, save_model
from flexehr.models.losses import BCE
from flexehr.models.models import MODELS, init_model
from utils.datasets import get_dataloaders
from utils.helpers import get_n_param, new_model_dir, set_seed, FormatterNoDuplicate


def parse_arguments(args_to_parse):
    """Parse command line arguments."""
    description = 'Pytorch implementation and evaluation of flexible EHR embedding.'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help='Name of the model for storing and loading purposes.')
    general.add_argument('-r', '--results', type=str,
                         default='results',
                         help='Directory to store results.')
    general.add_argument('--p-bar', action='store_true',
                         default=True,
                         help='Show progress bar.')
    general.add_argument('--cuda', action='store_true',
                         default=True,
                         help='Whether to use CUDA training.')
    general.add_argument('-s', '--seed', type=int,
                         default=0,
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training options')
    training.add_argument('data', type=str,
                          help='Path to data directory')
    training.add_argument('-e', '--epochs', type=int,
                          default=20,
                          help='Maximum number of epochs.')
    training.add_argument('-bs', type=int,
                          default=128,
                          help='Batch size for training.')
    training.add_argument('--lr', type=float,
                          default=1e-3,
                          help='Learning rate.')
    training.add_argument('--early-stopping', type=int,
                          default=5,
                          help='Epochs to train without validation auroc increase.')

    # Model options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default='Mortality',
                       choices=MODELS,
                       help='Type of decoder to use.')
    model.add_argument('-t', '--t_hours', type=int,
                       default=48,
                       help='ICU data time length.')
    model.add_argument('-n', '--n_bins', type=int,
                       default=20,
                       help='Number of bins per continuous variable.')
    # TO-DO: Need to assert somewhere that dividing by dt produces int
    model.add_argument('--dt', type=float,
                       default=1.0,
                       help='Time increment between sequence steps.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=32,
                       help='Dimension of the token embedding.')
    model.add_argument('-h', '--hidden-dim', type=int,
                       default=256,
                       help='Dimension of the LSTM hidden state.')
    model.add_argument('-p', '--p-dropout', type=float,
    				   default=0.0,
    				   help='Dropout rate.')
    model.add_argument('-w', '--weighted', type=bool,
                       default=True,
                       help='Whether to weight embeddings.')
    model.add_argument('-D', '--dynamic', type=bool,
                       default=True,
                       help='Whether to perform dynamic prediction.')

    # Evaluation options
    evaluation = parser.add_argument_group('Evaluation options')
    evaluation.add_argument('--eval', action='store_true',
                            default=False,
                            help='Whether to evaluate using pretrained model `name`.')
    evaluation.add_argument('--test', action='store_true',
                            default=True,
                            help='Whether to compute test losses.')
    evaluation.add_argument('--eval-bs', type=int,
                            default=128,
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)

    return args


def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """

    # Logging info
    formatter = logging.Formatter('%(asctime)s %(levelname)s - '
                                  '%(funcName)s: %(message)s',
                                  '%H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    stream = logging.StreamHandler()
    stream.setLevel('INFO')
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    model_name = f'{args.name}_lr{args.lr}_z{args.latent_dim}_h{args.hidden_dim}' \
    			 + f'_p{args.p_dropout}'
    model_dir = os.path.join(args.results, model_name)
    logger.info(f'Directory for saving and loading models: {model_dir}')

    if not args.eval:
        # Model directory
        new_model_dir(model_dir, logger=logger)

        # Dataloaders
        train_loader, valid_loader = get_dataloaders(
            args.data, args.t_hours, args.n_bins,
            validation=True, dynamic=args.dynamic, batch_size=args.bs, logger=logger)
        logger.info(f'Train {args.model_type}-{args.t_hours} with {len(train_loader.dataset)} samples')

        # Load model
        n_tokens = len(np.load(os.path.join(args.data, f'token2index_{args.t_hours}_{args.n_bins}.npy')).item())
        model = init_model(
            args.model_type, n_tokens, args.latent_dim, args.hidden_dim,
            p_dropout=args.p_dropout, dt=args.dt, weighted=args.weighted, dynamic=args.dynamic)
        logger.info(f'#params in model: {get_n_param(model)}')

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_f = BCE()
        model = model.to(device)

        # Training
        trainer = Trainer(
            model, loss_f, optimizer,
            device=device, logger=logger, save_dir=model_dir, p_bar=args.p_bar)
        trainer.train(
        	train_loader, valid_loader,
            epochs=args.epochs, early_stopping=args.early_stopping)

        # Save model
        metadata = vars(args)
        metadata['n_tokens'] = n_tokens
        save_model(trainer.model, model_dir, metadata=metadata)

    if args.test:
        #Load model
        model = load_model(model_dir, is_gpu=args.cuda)
        metadata = load_metadata(model_dir)

        # Dataloader
        test_loader, _ = get_dataloaders(
            metadata['data'], metadata['t_hours'], metadata['n_bins'],
            validation=False, dynamic=metadata['dynamic'], batch_size=args.eval_bs,
            shuffle=False, logger=logger)

        # Evaluate
        loss_f = BCE()
        evaluator = Trainer(
        	model, loss_f,
            device=device, logger=logger, save_dir=model_dir, p_bar=args.p_bar)
        evaluator._valid_epoch(test_loader)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
