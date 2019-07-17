import abc
import logging
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset


def get_dataloader(root, time, bins, dt=1.0, dynamic=True,
				   shuffle=True, pin_memory=True, batch_size=128,
				   logger=logging.getLogger(__name__), **kwargs):
	"""A generic data loader

	Parameters
	----------
	dataset : str
		Path to dataset.

	root : 
	"""
	pin_memory = pin_memory and torch.cuda.is_available
	dataset = EHR(root, time, bins, dt=dt, dynamic=dynamic, logger=logger)

	return DataLoader(dataset,
					  batch_size=batch_size,
					  shuffle=shuffle,
					  pin_memory=pin_memory,
					  **kwargs)


class AbstractDataset(Dataset, abc.ABC):
	"""Base Class for the EHR dataset.

	Parameters
	----------
	root : str
		Root directory of dataset.
	"""

	def __init__(self, root, time, bins, logger=logging.getLogger(__name__)):
		self.root = root
		self.train_data = os.path.join(root, f'train_{time}_{bins}.npy')
		self.logger = logger

	def __len__(self):
		return len(self.seqs)

	def __getitem__(self, idx):
		"""Get the sequence of `idx`.

		Return
		------
		sample : torch.Tensor
			Tensor of shape (10000, 2)."""
		pass


class EHR(Dataset):
	"""
	EHR Dataset.

	Parameters
	----------
	data : string
		Data directory.
	"""

	def __init__(self, data, time, bins, dt=1.0, dynamic=True,
				 logger=logging.getLogger(__name__)):

		self.data = data
		self.train_data = os.path.join(data, f'train_{time}_{bins}.npy')
		self.logger = logger

		dataset_dict = np.load(self.train_data).item()
		labels = dataset_dict['labels'].astype(np.float32)
		if dynamic:
			labels = np.tile(labels[:, None], (1, int(time / dt)))

		self.labels = torch.tensor(labels)
		self.seqs = torch.tensor(dataset_dict['seqs'])
		self.len = len(self.seqs)

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		seq = self.seqs[idx]
		label = self.labels[idx]

		return seq, label


# class EHR(AbstractDataset):
# 	"""EHR Dataset.

# 	Parameters
# 	----------
# 	root : string
# 		Root directory of dataset.
# 	"""
# 	def __init__(self, root, time, bins, dt=1.0, dynamic=True, **kwargs):
# 		super().__init__(root, time, bins, **kwargs)

# 		dataset_dict = np.load(self.train_data).item()
# 		self.seqs = torch.tensor(dataset_dict['seqs'])
# 		labels = dataset_dict['labels'].astype(np.float32)
# 		if dynamic:
# 			labels = np.tile(labels[:, None], (1, int(time / dt)))
# 		self.labels = torch.tensor(labels)

# 	def __getitem__(self, idx):
# 		"""Get the sequence of `idx`.

# 		Return
# 		------
# 		sample : torch.Tensor
# 			Tensor of shape (10000, 2).
# 		"""
# 		seq = self.seqs[idx]
# 		label = self.labels[idx]

# 		return seq, label
