"""
Module containing the embedder
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from scipy.stats import truncnorm


class Embedder(nn.Module):
	def __init__(self, n_tokens, latent_dim,
				 dt=1.0, weighted=True):
		"""Token embedder.

		Parameters
		----------
		n_tokens : int
			Number of tokens in vocabulary

		latent_dim : int
			Dimensionality of latent embedding.

		weighted : bool
			Whether or not to add embedding weights.
		"""
		super(Embedder, self).__init__()

		self.n_tokens = n_tokens
		self.latent_dim = latent_dim
		self.dt = dt
		self.weighted = weighted

		# Embedding layers
		self.embeddingX = nn.Embedding(n_tokens, latent_dim, padding_idx=0)
		if self.weighted:
			self.embeddingW = Embedding(n_tokens+1, 1)

	def forward(self, X):
		T = X[:, :, 0]
		X = X[:, :, 1].long()

		# Remove excess in time dimension
		t_max = T.max()
		n = (T < t_max).sum(dim=1).max()
		T = T[:, :n]
		X = X[:, :n]

		# Embed token
		embedded = self.embeddingX(X)

		# Extract relevant weights (and keep them positive)
		if self.weighted:
			w = torch.exp(self.embeddingW(X))

		# Step through sequence
		output = []
		for t in torch.arange(0, t_max, self.dt, dtype=torch.float32).cuda():
			t_idx = ((t <= T) & (T < t+1)).unsqueeze(2).float()
			counts = t_idx.sum(dim=1, keepdim=True)

			if self.weighted:
				X_t = t_idx * w * embedded
			else:
				X_t = t_idx * embedded

			X_t_avg = X_t.sum(dim=1, keepdim=True) / (counts + 1e-6)
			output += [X_t_avg]
		output = torch.cat(output, dim=1)

		return output


# HELPERS
class Embedding(nn.Module):
	def __init__(self, num_embeddings, embedding_dim,
			padding_idx=None, p_keep=None, init='truncnorm'):
		super(Embedding, self).__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		if padding_idx is not None:
			if padding_idx > 0:
				assert padding_idx < self.num_embeddings, \
					'padding_idx must be within num_embeddings'
			elif padding_idx < 0:
				assert padding_idx >= -self.num_embeddings, \
					'padding_idx must be within num_embeddings'
		self.padding_idx = padding_idx
		if p_keep is not None:
			self.keep_mask = torch.Tensor(num_embeddings, 1) 
		self.p_keep = p_keep
		self.init = init

		self.reset_parameters()

	def reset_parameters(self):
		with torch.no_grad():
			if self.init == 'truncnorm':
				t = 1. / (self.num_embeddings ** (1 / 2))
				weight = truncnorm.rvs(-t, t, size=[self.num_embeddings, self.embedding_dim])
				self.weight = Parameter(torch.tensor(weight).float())
			elif self.init == 'ones':
				self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
				self.weight.fill_(1.0)
		if self.padding_idx is not None:
			with torch.no_grad():
				self.weight[self.padding_idx].fill_(0)

	def forward(self, x):
		if self.training and self.p_keep is not None:
			batch_weights = self.weight * self.keep_mask.bernoulli_(self.p_keep)
			batch_weights = batch_weights / self.p_keep  # Scale embedding like normal dropout???
			x = F.embedding(x, batch_weights, padding_idx=0)
		else:
			x = F.embedding(x, self.weight, padding_idx=0)

		return x