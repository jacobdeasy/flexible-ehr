"""Training module."""

import logging
import numpy as np
import os
import torch

from collections import defaultdict
from sklearn.metrics import roc_auc_score
from timeit import default_timer
from tqdm import trange

from flexehr.utils.modelIO import save_model
from utils.helpers import array


class Trainer():
    """
    Class to handle model training and evaluation

    Parameters
    ----------
    model: flexehr.models.model
        Model to be evaluated.

    loss_f: flexehr.models.losses
        Loss function.

    optimizer: torch.optim.optimizer
        PyTorch optimizer used to minimize `loss_f`.

    device: torch.device, optional
        Device used for running the model.

    early_stopping: bool, optional
        Whether to make use of early stopping.

    save_dir: str, optional
        Name of save directory.

    p_bar: bool, optional
        Whether to have a progress bar.

    logger: logger.Logger, optional
        Logger to record info.
    """

    def __init__(self, model, loss_f,
                 optimizer=None,
                 device=torch.device('cpu'),
                 early_stopping=True,
                 save_dir='results',
                 p_bar=True,
                 logger=logging.getLogger(__name__)):

        self.model = model
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = 0 if early_stopping else None
        self.save_dir = save_dir
        self.p_bar = p_bar
        self.logger = logger

        self.max_v_auroc = -np.inf
        if self.optimizer is not None:
            self.losses_logger = LossesLogger(
                os.path.join(self.save_dir, 'train_losses.log'))
        self.logger.info(f'Device: {self.device}')

    def train(self, train_loader, valid_loader,
              epochs=10,
              early_stopping=5):
        """Trains the model.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader

        valid_loader : torch.utils.data.DataLoader

        epochs : int, optional
            Number of training epochs.

        early_stopping : int, optional
            Number of epochs to allow before early stopping.
        """
        start = default_timer()

        for epoch in range(epochs):
            storer = defaultdict(list)

            self.model.train()
            t_loss = self._train_epoch(train_loader, storer)

            self.model.eval()
            v_loss = self._valid_epoch(valid_loader, storer)

            self.logger.info(f'Train loss {t_loss:.4f}')
            self.logger.info(f'Valid loss {v_loss:.4f}')
            self.logger.info(f'Valid auroc {storer["auroc"][0]:.4f}')
            self.losses_logger.log(epoch, storer)

            if storer['auroc'][0] > self.max_v_auroc:
                self.max_v_auroc = storer['auroc'][0]
                save_model(self.model, self.save_dir, filename='model.pt')
                self.early_stopping = 0

            if self.early_stopping == early_stopping:
                break
            self.early_stopping += 1

        delta_time = (default_timer() - start) / 60
        self.logger.info(f'Finished training after {delta_time:.1f} minutes.')

    def _train_epoch(self, data_loader, storer):
        """Trains the model on the validation set for one epoch."""
        epoch_loss = 0.

        with trange(len(data_loader)) as t:
            for data, y_true in data_loader:
                data = data.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(data)
                iter_loss = self.loss_f(
                    y_pred, y_true, self.model.training, storer)
                epoch_loss += iter_loss.item()

                self.optimizer.zero_grad()
                iter_loss.backward()
                self.optimizer.step()

                if self.p_bar:
                    t.set_postfix(loss=iter_loss.item())
                    t.update()

        return epoch_loss / len(data_loader)

    def _valid_epoch(self, data_loader, storer=defaultdict(list)):
        """Trains the model on the validation set for one epoch."""
        epoch_loss = 0.
        y_preds = []

        with trange(len(data_loader)) as t:
            for data, y_true in data_loader:
                data = data.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(data)
                y_preds += [array(y_pred)]
                iter_loss = self.loss_f(
                    y_pred, y_true, self.model.training, storer)
                epoch_loss += iter_loss.item()

                if self.p_bar:
                    t.set_postfix(loss=iter_loss.item())
                    t.update()

        y_preds = np.concatenate(y_preds)
        y_trues = data_loader.dataset.Y

        metrics = self.compute_metrics(y_preds, y_trues)
        storer.update(metrics)

        return epoch_loss / len(data_loader)

    def compute_metrics(self, y_pred, y_true):
        """Compute metrics for predicted vs true labels."""
        if not isinstance(y_pred, np.ndarray):
            y_pred = array(y_pred)
        if not isinstance(y_true, np.ndarray):
            y_true = array(y_true)

        if y_pred.ndim == 2:
            y_pred = y_pred[:, -1]
            y_true = y_true[:, -1]

        metrics = {}
        metrics['auroc'] = [roc_auc_score(y_true, y_pred)]

        return metrics


class LossesLogger(object):
    """
    Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """Create a logger to store information for plotting."""
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger('losses_logger')
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ','.join(['Epoch', 'Train Loss', 'Valid Loss', 'AUROC'])
        self.logger.debug(header)

    def log(self, epoch, storer):
        """Write to the log file."""
        log_string = [epoch+1]
        for k in storer.keys():
            log_string += [sum(storer[k]) / len(storer[k])]
        log_string = ','.join(str(item) for item in log_string)
        self.logger.debug(log_string)
