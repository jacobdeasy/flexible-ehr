"""
Module containing all binary classification losses.
"""
import abc
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim


LOSSES = ['BCE', 'wBCE']


def get_loss_f(loss_name, **kwargs_parse):
	"""Return the correct loss function given the argparse arguments."""
	if loss_name == 'BCE':
		return BCE()
	elif loss_name == 'wBCE':
		pass
	else:
		assert loss_name not in LOSSES
		raise ValueError(f'Unknown loss: {loss_name}')


class BaseLoss(abc.ABC):
	"""
	Base class for losses.

	Parameters
	----------
	record_loss_every: int, optional
		How many steps between each loss record.
	"""

	def __init__(self, record_loss_every=1):
		self.n_train_steps = 0
		self.record_loss_every = record_loss_every

	@abc.abstractmethod
	def __call__(self, y_pred, y_true, is_train, storer, **kwargs):
		"""
		Calculates loss for a batch of data.

		Parameters
		----------
		y_true : torch.Tensor
		"""

	def _pre_call(self, is_train, storer):
		if is_train:
			self.n_train_steps += 1

		if not is_train or self.n_train_steps % self.record_loss_every == 0:
			storer = storer
		else:
			storer = None

		return storer


class BCE(BaseLoss):
	"""
	Compute the binary cross entropy loss.

	Parameters
	----------
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def __call__(self, y_pred, y_true, is_train, storer, **kwargs):
		storer = self._pre_call(is_train, storer)

		loss = F.binary_cross_entropy(y_pred, y_true)

		if storer is not None:
			storer['loss'].append(loss.item())

		return loss
