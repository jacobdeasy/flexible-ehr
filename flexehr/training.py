import logging
import os
import torch
from timeit import default_timer
from collections import defaultdict

from tqdm import trange
from torch.nn import functional as F

from flexehr.utils.modelIO import save_model


TRAIN_LOSSES_LOGFILE = 'train_losses.log'


class Trainer():
	"""
	Class to handle training of model.
	"""

	def __init__(self, model, optimizer, loss_f,
				 device=torch.device('cpu'),
				 logger=logging.getLogger(__name__),
				 save_dir='results',
				 is_progress_bar=True):

		self.model = model
		self.optimizer = optimizer
		self.loss_f = loss_f
		self.device = device
		self.logger = logger
		self.save_dir = save_dir
		self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
		self.logger.info(f'Training Device: {self.device}')

	def __call__(self, data_loader,
				 epochs=10,
				 checkpoint_every=10):
		"""
		Trains the model
		"""
		start = default_timer()
		self.model.train()
		for epoch in range(epochs):
			storer = defaultdict(list)
			mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
			self.logger.info(f'Epoch: {epoch+1} Average loss {mean_epoch_loss:.4f}')
			self.losses_logger.log(epoch, storer)

			if epoch % checkpoint_every == 0:
				save_model(self.model, self.save_dir,
						   filename=f'model-{epoch}.pt')

		self.model.eval()

		delta_time = (default_timer() - start) / 60
		self.logger.info(f'Finished training after {delta_time:.1f} min.')

	def _train_epoch(self, data_loader, storer, epoch):
		"""
		Trains the model for one epoch
		"""
		epoch_loss = 0.
		#kwarg stuff
		with trange(len(data_loader)) as t:
			for _, (data, y_true) in enumerate(data_loader):
				iter_loss = self._train_iteration(data, y_true, storer)
				epoch_loss += iter_loss

				t.set_postfix(loss=iter_loss)
				t.update()

		mean_epoch_loss = epoch_loss / len(data_loader)

		return mean_epoch_loss

	def _train_iteration(self, data, batch_true, storer):
		"""
		Trains the model for one iteration on a batch of data.
		"""
		data = data.to(self.device)
		batch_true = batch_true.to(self.device)

		batch_pred = self.model(data)
		loss = self.loss_f(batch_pred, batch_true, self.model.training, storer)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()


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

		header = ','.join(['Epoch', 'Loss', 'Value'])
		self.logger.debug(header)

	def log(self, epoch, losses_storer):
		"""Write to the log file."""
		for k, v in losses_storer.items():
			log_string = ','.join(str(item) for item in [epoch, k, mean(v)])
			self.logger.debug(log_string)


# HELPERS
def mean(l):
	"""Compute the mean of a list."""
	return sum(l) / len(l)
