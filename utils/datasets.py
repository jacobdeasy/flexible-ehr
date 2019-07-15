import abc
import os
import logging
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DIR = os.path.abspath(os.path.dirname(__file__))
# DATASETS_DICT = {'ehr': 'EHR'}
# DATASETS = list(DATASETS_DICT.keys())


# def get_dataset(dataset):
#     """Return the correct dataset."""
#     dataset = dataset.lower()
#     try:
#         # eval because stores name as string in order to put it at top of file
#         return eval(DATASETS_DICT[dataset])
#     except KeyError:
#         raise ValueError(f'Unknown dataset: {dataset}')


def get_dataset():
    """Return the correct dataset."""
    # eval because stores name as string in order to put it at top of file
    return eval('EHR')


def get_dataloader(root, time=48, dt=1.0, dynamic=True, shuffle=True, pin_memory=True,
				   batch_size=128, logger=logging.getLogger(__name__), **kwargs):
	"""A generic data loader

	Parameters
	----------
	dataset : str
		Path to dataset.

	root : 
	"""
	pin_memory = pin_memory and torch.cuda.is_available
	Dataset = get_dataset()
	dataset = EHR(root=root, time=time, dt=dt, dynamic=dynamic, logger=logger)

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

	def __init__(self, root, time, transforms_list=[], logger=logging.getLogger(__name__)):
		self.root = root
		self.train_data = os.path.join(root, f'train_{time}.npy')
		self.transforms = transforms.Compose(transforms_list)
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


class EHR(AbstractDataset):
	"""EHR Dataset.

	Parameters
	----------
	root : string
		Root directory of dataset.
	"""
	def __init__(self, root=os.path.join(DIR, '../data/'), time=48, dt=1.0,
				 dynamic=True, **kwargs):
		super().__init__(root, time, **kwargs)

		dataset_zip = np.load(self.train_data).item()
		self.seqs = torch.tensor(dataset_zip['seqs'])
		labels = dataset_zip['labels'].astype(np.float32)
		if dynamic:
			labels = np.tile(labels[:, None], (1, int(time / dt)))
		self.labels = torch.tensor(labels)

	def __getitem__(self, idx):
		"""Get the sequence of `idx`.

		Return
		------
		sample : torch.Tensor
			Tensor of shape (10000, 2).
		"""
		seq = self.seqs[idx]
		label = self.labels[idx]

		seq = self.transforms(seq)

		return seq, label
