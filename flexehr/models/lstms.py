import math
import torch
import torch.nn as nn


class LSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_norm=True):
		"""
		LSTM class with layer normalization option.

		Parameters
		----------
		input_dim : int
			Dimensionality of LSTM input.

		hidden_dim : int
			Dimensionality of LSTM hidden state.

		layer_norm : bool
			Whether to use layer normalized version of LSTM.
		"""
		super(LSTM, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.layer_norm = layer_norm

		self.lstm_cell = LSTMCell(input_dim, hidden_dim, layer_norm=self.layer_norm)

	def forward(self, x):
		h = torch.zeros(x.size(0), self.hidden_dim, dtype=torch.float32).cuda()
		c = torch.zeros(x.size(0), self.hidden_dim, dtype=torch.float32).cuda()
		x = x.chunk(x.size(1), dim=1)

		O = []
		for x_t in x:
			h, c = self.lstm_cell(x_t.squeeze(1), (h, c))
			O += [h]
		O = torch.stack(O, 1)

		return O, (h, c)


class LSTMCell(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_norm=True):
		"""
		LSTM cell class with layer normalization option.

		Parameters
		----------
		input_dim : int
			Dimensionality of LSTM cell input.

		hidden_dim : int
			Dimensionality of LSTM cell hidden state.

		layer_norm : bool
			Whether to use layer normalized version of LSTM cell.
		"""
		super(LSTMCell, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.layer_norm = layer_norm

		self.i2h = nn.Linear(input_dim, 4 * hidden_dim)
		self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim)
		if self.layer_norm:
			self.ln1 = nn.LayerNorm(4 * hidden_dim)
			self.ln2 = nn.LayerNorm(4 * hidden_dim)
			self.ln3 = nn.LayerNorm(hidden_dim)

		self.reset_parameters()

	def reset_parameters(self):
		# TO-DO: remove as overwritten in overall initialization?
		std = 1.0 / math.sqrt(self.hidden_dim)
		for w in self.parameters():
			w.data.uniform_(-std, std)

	def forward(self, x, hidden):
		h, c = hidden

		# Linear mappings
		if self.layer_norm:
			preact = self.ln1(self.i2h(x)) + self.ln2(self.h2h(h))
		else:
			preact = self.i2h(x) + self.h2h(h)

		# activations
		gates = preact[:, :-self.hidden_dim].sigmoid()
		f_t = gates[:, :self.hidden_dim]
		i_t = gates[:, self.hidden_dim:-self.hidden_dim]
		o_t = gates[:, -self.hidden_dim:]
		g_t = preact[:, -self.hidden_dim:].tanh()

		if self.layer_norm:
			c_t = self.ln3(torch.mul(c, f_t) + torch.mul(i_t, g_t))
		else:
			c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
		h_t = torch.mul(o_t, c_t.tanh())

		return h_t, c_t
