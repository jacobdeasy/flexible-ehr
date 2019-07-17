import argparse
import logging
import sys
import os
from configparser import ConfigParser

from torch import optim

from flexehr import init_specific_model, Trainer, Evaluator
from flexehr.utils.modelIO import save_model, load_model, load_metadata
from flexehr.models.losses import LOSSES, get_loss_f
from flexehr.models.models import MODELS
from utils.datasets import get_dataloader
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           FormatterNoDuplicate)


def parse_arguments(args_to_parse):
    """Parse command line arguments."""
    description = 'Pytorch implementation and evaluation of flexible EHR embedding.'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help='Name of the model for storing and loading purposes.')
    general.add_argument('-R', '--results', type=str,
                         default='results',
                         help='Directory to store results.')
    general.add_argument('--progress-bar', action='store_true',
                         default=True,
                         help='Show progress bar.')
    general.add_argument('--cuda', action='store_true',
                         default=True,
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int,
                         default=0,
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('data', type=str,
                          help='Path to data directory')
    training.add_argument('--checkpoint-every', type=int,
                          default=30,
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-e', '--epochs', type=int,
                          default=10,
                          help='Maximum number of epochs.')
    training.add_argument('-bs', type=int,
                          default=128,
                          help='Batch size for training.')
    training.add_argument('--lr', type=float,
                          default=1e-3,
                          help='Learning rate.')

    # Model options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default='Mortality',
                       choices=MODELS,
                       help='Type of decoder to use.')
    model.add_argument('-T', '--time', type=int,
                       default=48,
                       help='ICU data time length.')
    model.add_argument('-b', '--n_bins', type=int,
                       default=20,
                       help='Number of bins per continuous variable.')
    model.add_argument('-n', '--n_tokens', type=int,
                       default=36686,
                       help='Number of unique tokens in dataset.')
    # TO-DO: Need to assert somewhere that dividing by dt produces int
    model.add_argument('--dt', type=float,
                       default=1.0,
                       help='Time increment between sequence steps.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=32,
                       help='Dimension of the token embedding.')
    model.add_argument('-H', '--hidden-dim', type=int,
                       default=256,
                       help='Dimension of the LSTM hidden state.')
    model.add_argument('-l', '--loss',
                       default='BCE',
                       choices=LOSSES,
                       help='Type of loss function to use.')
    model.add_argument('-w', '--weighted', type=bool,
                       default=True,
                       help='Whether to weight embeddings.')
    model.add_argument('-d', '--dynamic', type=bool,
                       default=True,
                       help='Whether to perform dynamic prediction.')

    # Evaluation options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--eval', action='store_true',
                            default=False,
                            help='Whether to evaluate using pretrained model `name`.')
    evaluation.add_argument('--metrics', action='store_true',
                            default=False,
                            help='Whether to compute metrics.')
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
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  '%H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    stream = logging.StreamHandler()
    stream.setLevel('INFO')
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=args.cuda)
    model_dir = os.path.join(RES_DIR, args.name)
    logger.info(f'Directory for saving and loading experiments: {model_dir}')

    if not args.eval:

        create_safe_directory(model_dir, logger=logger)

        # Dataloader
        train_loader = get_dataloader(args.data, args.time, args.n_bins,
                                      dynamic=args.dynamic,
                                      batch_size=args.bs,
                                      logger=logger)
        logger.info(f'Train {args.model_type} {args.time} with {len(train_loader.dataset)} samples')

        # Load model
        model = init_specific_model(args.model_type, args.n_tokens,
                                    args.latent_dim, args.hidden_dim,
                                    dt=args.dt,
                                    weighted=args.weighted,
                                    dynamic=args.dynamic)
        logger.info(f'Num parameters in model: {get_n_param(model)}')

        # Train
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model = model.to(device)
        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                          device=device,
                          logger=logger,
                          save_dir=model_dir,
                          is_progress_bar=args.progress_bar)
        trainer(train_loader,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every)

        # Save model
        save_model(trainer.model, model_dir, metadata=vars(args))

    if args.metrics or args.test:

        model = load_model(model_dir, is_gpu=args.cuda)
        metadata = load_metadata(model_dir)
        # TO-DO: currently uses train dataset
        test_loader = get_dataloader(metadata['data'], metadata['time'], metadata['n_bins'],
                                     dynamic=metadata['dynamic'],
                                     batch_size=args.eval_bs,
                                     shuffle=False,
                                     logger=logger)
        loss_f = get_loss_f(args.loss,
                            device=device,
                            **vars(args))
        evaluator = Evaluator(model, loss_f,
                              device=device,
                              logger=logger,
                              save_dir=model_dir,
                              is_progress_bar=args.progress_bar)

        evaluator(test_loader, is_metrics=args.metrics, is_losses=args.test)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
