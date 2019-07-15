import os
import logging
from collections import defaultdict
import json
from timeit import default_timer

from tqdm import tqdm, trange
import numpy as np
import torch

from flexehr.models.losses import get_loss_f
from flexehr.utils.modelIO import save_metadata


TEST_LOSSES_FILE = 'test_losses.log'
METRICS_FILENAME = 'metrics.log'
METRIC_HELPERS_FILE = 'metric_helpers.pth'


class Evaluator:
    """
    Class to handle training of model

    Parameters
    ----------
    model: flexehr.models.model
    """

    def __init__(self, model, loss_f,
                 device=torch.device('cpu'),
                 logger=logging.getLogger(__name__),
                 save_dir='results',
                 is_progress_bar=True):

        self.model = model.to(device)
        self.loss_f = loss_f
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info(f'Testing Device: {self.device}')

    def __call__(self, data_loader, is_metrics=False, is_losses=True):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_metrics: bool, optional
            Whether to compute and store the disentangling metrics.

        is_losses: bool, optional
            Whether to compute and store the test losses.
        """
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        metric, losses = None, None
        if is_metrics:
            self.logger.info('Computing metrics...')
            metrics = self.compute_metrics(data_loader)
            self.logger.info(f'Losses: {metrics}')
            save_metadata(metrics, self.save_dir, filename=METRICS_FILENAME)

        if is_losses:
            self.logger.info('Computing losses...')
            losses = self.compute_losses(data_loader)
            self.logger.info(f'Losses: {losses}')
            save_metadata(losses, self.save_dir, filename=TEST_LOSSES_FILE)

        if is_still_training:
            self.model.train()

        time_elapsed = (default_timer() - start) / 60
        self.logger.info(f'Finished evaluating after {time_elapsed:.1f} min.')

        return metric, losses

    def compute_losses(self, data_loader):
        """Compute all test losses.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        storer = defaultdict(list)
        for data, batch_true in tqdm(data_loader, leave=False, disable=not self.is_progress_bar):
            data = data.to(self.device)
            batch_true = batch_true.to(self.device)

            batch_pred = self.model(data)
            _ = self.loss_f(batch_pred, batch_true, self.model.training, storer)

        losses = {k: sum(v) / len(data_loader) for k, v in storer.items()}

        return losses

    # def compute_metrics(self, data_loader):
    #     """Compute all the metrics.

    #     Parameters
    #     ----------
    #     data_loader : torch.utils.data.DataLoader
    #     """
    #     self.logger.info('Computing the ')

    #     with torch.no_grad():
    #         for x, label in data_loader:
