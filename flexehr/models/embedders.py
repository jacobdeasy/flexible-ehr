"""Module containing the embedders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import truncnorm
from torch.nn.parameter import Parameter

from utils.helpers import get_device


class Embedder(nn.Module):
    def __init__(self, n_tokens, latent_dim, dt=1.0, weighted=True):
        """Token embedder.

        Parameters
        ----------
        n_tokens: int
            Number of tokens in vocabulary

        latent_dim: int
            Dimensionality of latent embedding.

        dt: float
            Time increment between sequence steps.

        weighted: bool
            Whether or not to add embedding weights.
        """
        super(Embedder, self).__init__()

        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.dt = dt
        self.weighted = weighted

        self.device = get_device()

        self.embedX = Embedding(n_tokens+1, latent_dim, padding_idx=0)
        if self.weighted:
            self.embedW = Embedding(n_tokens+1, 1)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        T = X[:, :, 0]
        X = X[:, :, 1].long()

        # Remove excess in time dimension
        t_max = T.max()
        n = (T < t_max).sum(dim=1).max()
        T = T[:, :n]
        X = X[:, :n]

        # Embed tokens
        embedded = self.embedX(X)

        # Extract token weights (and keep them positive)
        if self.weighted:
            w = self.embedW(X)
            w = torch.exp(w)

        # Step through sequence
        output = []
        for t in torch.arange(
                0, t_max, self.dt, dtype=torch.float32).to(self.device):
            t_idx = ((t <= T) & (T < t+1)).float().unsqueeze(2)
            counts = t_idx.sum(dim=1, keepdim=True)

            if self.weighted:
                w_t = t_idx * w
                X_t = w_t * embedded
            else:
                X_t = t_idx * embedded

            X_t_avg = X_t.sum(dim=1, keepdim=True) / (counts + 1e-6)
            output += [X_t_avg]

        output = torch.cat(output, dim=1)

        return output


# HELPERS
class Embedding(nn.Module):
    def __init__(self, n_tokens, latent_dim,
                 padding_idx=None, init='truncnorm'):
        super(Embedding, self).__init__()
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.n_tokens, \
                    'padding_idx must be within n_tokens'
            elif padding_idx < 0:
                assert padding_idx >= -self.n_tokens, \
                    'padding_idx must be within n_tokens'
        self.padding_idx = padding_idx
        self.init = init

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.init == 'truncnorm':
                t = 1. / (self.n_tokens ** (1 / 2))
                weights = truncnorm.rvs(
                    -t, t, size=[self.n_tokens, self.latent_dim])
                self.weights = Parameter(torch.tensor(weights).float())
            elif self.init == 'zeros':
                self.weights = Parameter(
                    torch.Tensor(self.n_tokens, self.latent_dim))
                self.weights.fill_(1.0)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weights[self.padding_idx].zero_()

    def forward(self, x):
        x = F.embedding(x, self.weights, padding_idx=self.padding_idx)

        return x
