"""Helpers module."""

import argparse
import numpy as np
import os
import random
import shutil
import torch


def array(x):
    """Place tensor on cpu and convert to numpy.ndarray."""
    if x.device != torch.device('cpu'):
        x = x.cpu()

    return x.detach().numpy()


def get_device(is_gpu=True):
    """Return the correct device"""
    return torch.device(
        'cuda' if torch.cuda.is_available() and is_gpu else 'cpu')


def get_model_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device


def get_n_param(model):
    """Return the number of parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])

    return nParams


def new_model_dir(directory, logger=None):
    """Create a directory and archive the previous one if already existed."""
    if os.path.exists(directory):
        if logger is not None:
            logger.warning(
                f'{directory} already exists. Archiving to {directory}.zip')
        shutil.make_archive(directory, 'zip', directory)
        shutil.rmtree(directory)
    os.makedirs(directory)


def set_seed(seed):
    """Set all random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True


class FormatterNoDuplicate(argparse.ArgumentDefaultsHelpFormatter):
    """Formatter overriding `argparse.ArgumentDefaultsHelpFormatter` to show
    `-e, --epoch EPOCH` instead of `-e EPOCH, --epoch EPOCH`

    Note
    ----
    - code modified from cPython:
    https://github.com/python/cpython/blob/master/Lib/argparse.py
    """

    def _format_action_invocation(self, action):
        # no args given
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)

            return metavar

        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)
            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    # don't store the DEFAULT
                    parts.append('%s' % (option_string))
                # store DEFAULT for the last one
                parts[-1] += ' %s' % args_string

            return ', '.join(parts)
