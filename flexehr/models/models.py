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


def init_specific_model(model_type, n_tokens, latent_dim, hidden_dim,
						dt=1.0, weighted=True, dynamic=True):
	"""Return an instance of a LSTM with an embedding."""
	model_type = model_type.lower().capitalize()
	if model_type not in MODELS:
		raise ValueError(f'Unknown model_type={model_type}. Possible values: {MODELS}')

	decoder = get_decoder(model_type)
	model = Model(decoder, n_tokens, latent_dim, hidden_dim,
				  dt=dt, weighted=weighted, dynamic=dynamic)
	model.model_type = model_type  # store to help reloading

	return model


class Model(nn.Module):
	def __init__(self, decoder, n_tokens, latent_dim, hidden_dim,
				 dt=1.0, weighted=True, dynamic=True):
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
		self.dt = dt
		self.weighted = weighted
		self.dynamic = dynamic

		self.embedder = Embedder(self.n_tokens, self.latent_dim,
								 dt=dt, weighted=weighted)
		self.lstm = LSTM(self.latent_dim, self.hidden_dim)
		self.decoder = decoder(self.hidden_dim)

		self.reset_parameters()

	def forward(self, x):
		"""
		Forward pass of model.

		Parameters
		----------
		x : torch.Tensor
			Batch of data. Shape (batch_size, 10000, 2)
		"""
		latent_emb = self.embedder(x)
		all_hidden, (final_hidden, _) = self.lstm(latent_emb)
		if self.dynamic:
			output = self.decoder(all_hidden)
		else:
			output = self.decoder(final_hidden)

		return output

	def reset_parameters(self):
		self.apply(weights_init)


class EmbedEHRLSTM(nn.Module):
	def __init__(self, n_tokens, x_n_emb, n_hidden, p_dropout=0.0, dt=1.0):
		super(EmbedEHRLSTM, self).__init__()
		self.n_tokens = n_tokens
		self.n_hidden = n_hidden
		self.x_n_emb = x_n_emb
		self.p_dropout = p_dropout
		self.dt = dt

		# self.embedding = EmbedTimeseries(n_tokens, x_n_emb, dt)
		self.embedding = EmbedTimeseriesW(n_tokens, x_n_emb, dt)
		self.dropout = nn.Dropout(p_dropout)
		self.lstm = LSTM(x_n_emb, n_hidden)
		self.out = nn.Linear(n_hidden, 1)

		self.dropout2 = nn.Dropout(0.4)

	def forward(self, x):
		x = self.embedding(x)
		x = F.relu(x)
		x = self.dropout(x)
		x, (h, c) = self.lstm(x)

		x = self.dropout2(x)

		x = self.out(x).squeeze().sigmoid()

		return x


class EmbedEHRLSTMLOS(nn.Module):

	def __init__(self, n_tokens, x_n_emb, n_hidden, p_dropout=0.0, dt=1.0):
		super(EmbedEHRLSTMLOS, self).__init__()
		self.n_tokens = n_tokens
		self.x_n_emb = x_n_emb
		self.n_hidden = n_hidden
		self.p_dropout = p_dropout
		self.dt = dt

		# self.embedding = EmbedTimeseries(n_tokens, x_n_emb, dt)
		self.embedding = EmbedTimeseriesW(n_tokens, x_n_emb, dt)
		self.dropout = nn.Dropout(p_dropout)
		self.lstm = LSTM(x_n_emb, n_hidden)
		self.out = nn.Linear(n_hidden, 1)

	def forward(self, x):
		x = self.embedding(x)
		x = F.relu(x)
		x = self.dropout(x)
		x, (h, c) = self.lstm(x)
		x = self.out(x).squeeze()

		return x


class EmbedEHRTranformer(nn.Module):

	def __init__(self, n_tokens, d_model, N, heads,	dt=1.0, p_dropout=0.0):
		super(EmbedEHRTranformer, self).__init__()
		self.n_tokens = n_tokens
		self.d_model = d_model
		self.N = N
		self.heads = heads
		self.dt = dt
		self.p_dropout = p_dropout

		self.embedding = EmbedTimeseries(n_tokens, d_model, dt)
		self.dropout = nn.Dropout(p_dropout)
		self.transformer = EHRTransformer(d_model, N, heads, 48, p_dropout=p_dropout)
		self.out = nn.Linear(d_model, 1)

	def forward(self, x):
		x = self.embedding(x)
		x = F.relu(x)
		x = self.dropout(x)

		x, W = self.transformer(x)

		x = self.out(x).squeeze().sigmoid()

		return x, W
