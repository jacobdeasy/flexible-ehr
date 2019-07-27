"""
Module containing all binary classification losses.
"""
import abc
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim


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
	def __call__(self, y_pred, y_true, is_train, storer):
		"""Calculates loss for a batch of data."""

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
	"""

	def __init__(self):
		super().__init__()

	def __call__(self, y_pred, y_true, is_train, storer):
		"""Calculates binary cross entropy loss.

		Parameters
		----------
		y_pred: torch.Tensor


		y_true: torch.Tensor


		is_trin: bool
			Whether model is training.

		storer: collections.defaultdict
			
		"""
		storer = self._pre_call(is_train, storer)

		loss = F.binary_cross_entropy(y_pred, y_true)

		if storer is not None:
			if is_train:
				storer['train_loss'].append(loss.item())
			else:
				storer['valid_loss'].append(loss.item())

		return loss
