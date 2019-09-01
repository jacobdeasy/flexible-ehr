"""
Module containing the main model class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from flexehr.utils.initialization import weights_init
from .embedders import Embedder
from .decoders import get_decoder
from .lstms import LSTM


MODELS = ['Mortality', 'LOS', 'LOS7']


def init_model(model_type, n_tokens, latent_dim, hidden_dim,
			   p_dropout=0.0, dt=1.0, weighted=True, dynamic=True):
	"""Return an instance of a LSTM with an embedding."""
	model_type = model_type.lower().capitalize()
	if model_type not in MODELS:
		raise ValueError(f'Unknown model_type={model_type}. Possible values: {MODELS}')

	decoder = get_decoder(model_type)
	model = Model(decoder, n_tokens, latent_dim, hidden_dim,
				  p_dropout=p_dropout, dt=dt, weighted=weighted, dynamic=dynamic)
	model.model_type = model_type  # store to help reloading

	return model


class Model(nn.Module):
	def __init__(self, decoder, n_tokens, latent_dim, hidden_dim,
				 p_dropout=0.0, dt=1.0, weighted=True, dynamic=True):
		"""
		Class which defines model and forward pass.

		Parameters
		----------
		n_tokens : int
			Number of tokens in vocabulary
		"""
		super(Model, self).__init__()

		self.n_tokens = n_tokens
		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim
		self.p_dropout = p_dropout
		self.dt = dt
		self.weighted = weighted
		self.dynamic = dynamic

		self.embedder = Embedder(self.n_tokens, self.latent_dim,
								 dt=self.dt, weighted=self.weighted)
		self.dropout = nn.Dropout(self.p_dropout)
		self.lstm = LSTM(self.latent_dim, self.hidden_dim)
		self.decoder = decoder(self.hidden_dim)

		self.reset_parameters()

	def forward(self, input):
		"""
		Forward pass of model.

		Parameters
		----------
		input: torch.Tensor
			Batch of data. Shape (batch_size, 10000, 2)
		"""
		emb = self.dropout(F.relu(self.embedder(input)))
		all_hidden, (final_hidden, _) = self.lstm(emb)
		if self.dynamic:
			output = self.decoder(self.dropout(all_hidden))
		else:
			output = self.decoder(self.dropout(final_hidden))

		return output

	def reset_parameters(self):
		self.apply(weights_init)
