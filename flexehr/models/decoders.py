"""
Module containing the decoders.
"""
import torch
from torch import nn


DECODER_DICT = {'Mortality': 'Binary', 'Readmission': 'Binary',
				'LOS': 'Regression', 'LOS7': 'Binary'}


# All decoders should be called Decoder<Model>
def get_decoder(model_type):
	model_type = model_type.lower().capitalize()
	model_type = DECODER_DICT[model_type]
	return eval(f'Decoder{model_type}')


class DecoderBinary(nn.Module):
	def __init__(self, hidden_dim):
		"""Hidden state decoder for binary classification tasks.

		Parameters
		----------
		hidden_dim : int
			Dimensionality of LSTM hidden state
		"""
		super(DecoderBinary, self).__init__()

		self.hidden_dim = hidden_dim

		# Fully connected layer
		self.lin = nn.Linear(hidden_dim, 1)

	def forward(self, h):
		y = self.lin(h).squeeze().sigmoid()

		return y


class DecoderRegression(nn.Module):
	def __init__(self, hidden_dim):
		"""Hidden state decoder for regression tasks.

		Parameters
		----------
		hidden_dim : int
			Dimensionality of LSTM hidden state
		"""
		super(DecoderRegression, self).__init__()

		self.hidden_dim = hidden_dim

		# Fully connected layer
		self.lin = nn.Linear(hidden_dim, 1)

	def forward(self, h):
		y = self.lin(h).squeeze()

		return y
