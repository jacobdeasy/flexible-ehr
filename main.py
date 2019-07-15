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
                           get_config_section, update_namespace_, FormatterNoDuplicate)


CONFIG_FILE = 'hyperparam.ini'
RES_DIR = 'results'
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', 'debug']
EXPERIMENTS = ADDITIONAL_EXP + [f'{loss}' for loss in LOSSES]


def parse_arguments(args_to_parse):
    """Parse the command line arguments."""
    default_config = get_config_section([CONFIG_FILE], 'Custom')

    description = "Pytorch implementation and evaluation of flexible EHR embedding."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help='Name of the model for storing and loading purposes.')
    general.add_argument('-L', '--log-level',
                         default=default_config['log_level'],
                         help="Logging levels.",
                         choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int,
                         default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('root', type=str,
                          help='Path to root data directory')
    training.add_argument('--checkpoint-every', type=int,
                          default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of decoder to use.')
    model.add_argument('-T', '--time', type=int,
                       default=default_config['time_len'],
                       help='ICU data time length.')
    model.add_argument('-n', '--n_tokens', type=int,
                       default=default_config['n_tokens'],
                       help='Number of unique tokens in dataset.')
    # TO-DO: Need to assert somewhere that dividing by dt produces int
    model.add_argument('-t', '--dt', type=float,
                       default=default_config['time_step'],
                       help='Time increment between sequence steps.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the token embedding.')
    model.add_argument('-H', '--hidden-dim', type=int,
                       default=default_config['hidden_dim'],
                       help='Dimension of the LSTM hidden state.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of loss function to use.")
    model.add_argument('-w', '--weighted', type=bool,
                       default=default_config['weighted'],
                       help='Whether to use weighted embeddings.')
    model.add_argument('-d', '--dynamic', type=bool,
                       default=default_config['dynamic'],
                       help='Whether to perform dynamic prediction.')

    # Evaluation options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the metrcics.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)

    # Experiment stuff?

    return args


def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info(f'Root directory for saving and loading experiments: {exp_dir}')

    if not args.is_eval_only:

        create_safe_directory(exp_dir, logger=logger)

        # PREPARES DATA
        train_loader = get_dataloader(args.root, args.time,
                                      dynamic=args.dynamic,
                                      batch_size=args.batch_size,
                                      logger=logger)
        logger.info(f'Train {args.model_type} {args.time} with {len(train_loader.dataset)} samples')

        # PREPARES MODEL
        model = init_specific_model(args.model_type, args.n_tokens,
                                    args.latent_dim, args.hidden_dim,
                                    dt=args.dt, weighted=args.weighted,
                                    dynamic=args.dynamic)
        logger.info(f'Num parameters in model: {get_n_param(model)}')

        # TRAINS
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model = model.to(device)
        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar)
        trainer(train_loader,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))

    if args.is_metrics or not args.no_test:
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        # TO-DO: currently uses train dataset
        test_loader = get_dataloader(metadata['root'], metadata['time'],
                                     dynamic=metadata['dynamic'],
                                     batch_size=args.eval_batchsize,
                                     shuffle=False,
                                     logger=logger)
        loss_f = get_loss_f(args.loss,
                            device=device,
                            **vars(args))
        evaluator = Evaluator(model, loss_f,
                              device=device,
                              logger=logger,
                              save_dir=exp_dir,
                              is_progress_bar=not args.no_progress_bar)

        evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
