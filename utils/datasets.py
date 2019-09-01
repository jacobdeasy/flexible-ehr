import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def get_dataloaders(data, t_hours, n_bins, validation, dt=1.0, dynamic=True,
				   shuffle=True, pin_memory=True, batch_size=128,
				   logger=logging.getLogger(__name__)):
	"""A generic data loader.

	Parameters
	----------
	data: str
		Data directory.

	t_hours: int


	n_bins: int


	validation: bool
		Whether or not to return a validation DataLoader also (when training).

	dt: float, optional
		Time step between intervals.

	dynamic: bool, optional
		Whether the model should predict in a dynamic fashion.

	shuffle: bool, optional
		Whether to shuffle data during training.

	pin_memory: bool, optional
		Whether to pin memory in the GPU when using CUDA.

	batch_size: int, optional

	logger: logging.Logger, optional
	"""
	pin_memory = pin_memory and torch.cuda.is_available

	arrs = np.load(os.path.join(data, f'arrs_{t_hours}_{n_bins}.npy')).item()

	if validation:
		X_train, X_valid, y_train, y_valid = train_test_split(
			arrs['X_train'], arrs['Y_train'], test_size=1000, stratify=arrs['Y_train'])
		train_dataset = EHR(X_train, y_train, t_hours, dt, dynamic)
		valid_dataset = EHR(X_valid, y_valid, t_hours, dt, dynamic)

		train_dataloader = DataLoader(train_dataset,
									  batch_size=batch_size,
									  shuffle=shuffle,
									  pin_memory=pin_memory)
		valid_dataloader = DataLoader(valid_dataset,
									  batch_size=batch_size,
									  shuffle=False,
									  pin_memory=pin_memory)

		return train_dataloader, valid_dataloader

	else:
		test_dataset = EHR(arrs['X_test'], arrs['Y_test'], t_hours, dt, dynamic)
		test_dataloader = DataLoader(test_dataset,
									 batch_size=batch_size,
									 shuffle=False,
									 pin_memory=pin_memory)

		return test_dataloader, None


class EHR(Dataset):
	"""
	EHR Dataset.

	Parameters
	----------
	X: numpy.ndarray
		Array containing patient sequences, shape (n_patients, 10000, 2)

	Y: numpy.ndarray
		Array containing patient outcomes, shape (n_patients,)

	t_hours: int, optional


	dt: float, optional
		Time step between intervals.

	dynamic: bool, optional
		Whether the model should predict in a dynamic fashion.

	logger: logging.Logger, optional
	"""

	def __init__(self, X, Y, t_hours=48, dt=1.0, dynamic=True,
				 logger=logging.getLogger(__name__)):

		self.logger = logger

		self.X = torch.tensor(X)
		if dynamic:  # shape (n_patients,) -> (n_patients, n_intervals)
			Y = np.tile(Y[:, None], (1, int(t_hours / dt)))
		self.Y = torch.tensor(Y).float()

		self.len = len(self.X)

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx]
