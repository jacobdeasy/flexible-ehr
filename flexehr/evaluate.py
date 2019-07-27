import logging
import json
import numpy as np
import os
import torch
from collections import defaultdict
from timeit import default_timer
from tqdm import tqdm, trange

from flexehr.models.losses import get_loss_f
from flexehr.utils.modelIO import save_metadata


class Evaluator():
    """
    Class to handle model evaluation

    Parameters
    ----------
    model: flexehr.models.model
        Model to be evaluated.

    loss_f: flexehr.models.losses


    device: torch.device, optional
        Device used for running the model.

    logger: logger.Logger, optional


    save_dir: str, optional
        Name of save directory.

    progress_bar: bool, optional
        Whether to have a progress bar.    
    """

    def __init__(self, model, loss_f,
                 device=torch.device('cpu'),
                 logger=logging.getLogger(__name__),
                 save_dir='results',
                 progress_bar=True):

        self.model = model.to(device)
        self.loss_f = loss_f
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.progress_bar = progress_bar
        self.logger.info(f'Testing Device: {self.device}')

    def __call__(self, data_loader, is_metrics=False, is_losses=True):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_metrics: bool, optional
            Whether to compute metrics.

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
            save_metadata(metrics, self.save_dir, filename='metrics.log')

        if is_losses:
            self.logger.info('Computing losses...')
            losses = self.compute_losses(data_loader)
            self.logger.info(f'Losses: {losses}')
            save_metadata(losses, self.save_dir, filename='test_losses.log')

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
        for data, y_true in tqdm(data_loader, leave=False, disable=not self.progress_bar):
            data = data.to(self.device)
            y_true = y_true.to(self.device)

            y_pred = self.model(data)
            _ = self.loss_f(y_pred, y_true, self.model.training, storer)

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
