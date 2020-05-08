"""RNN variants module."""

import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn import Parameter


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """LSTM class.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input state.

        hidden_dim : int
            Dimensionality of the hidden state
        """
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm_cell = LSTMCell(input_dim, hidden_dim)

    def forward(self, x):
        h = torch.zeros(
            x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)
        c = torch.zeros(
            x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)

        x = x.chunk(x.size(1), dim=1)

        H = []
        for x_t in x:
            h, c = self.lstm_cell(x_t.squeeze(1), (h, c))
            H += [h]

        return torch.stack(H, 1), (h, c)


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_norm=True):
        """LSTM cell class with layer normalization option.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input space.

        hidden_dim : int
            Dimensionality of the hidden space.

        layer_norm : bool, optional
            Whether to user layer normalisation.
        """
        super(LSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = layer_norm

        # Xavier init input-to-hidden
        W_ii = torch.Tensor(input_dim, hidden_dim)
        W_if = torch.Tensor(input_dim, hidden_dim)
        W_ig = torch.Tensor(input_dim, hidden_dim)
        W_io = torch.Tensor(input_dim, hidden_dim)
        for w in [W_ii, W_if, W_ig, W_io]:
            init.xavier_uniform_(w.data)
        self.W_i = Parameter(torch.cat([W_ii, W_if, W_ig, W_io], dim=1))

        # Orthogonal init hidden-to-hidden
        W_hi = torch.Tensor(hidden_dim, hidden_dim)
        W_hf = torch.Tensor(hidden_dim, hidden_dim)
        W_hg = torch.Tensor(hidden_dim, hidden_dim)
        W_ho = torch.Tensor(hidden_dim, hidden_dim)
        for w in [W_hi, W_hf, W_hg, W_ho]:
            init.orthogonal_(w.data)
        self.W_h = Parameter(torch.cat([W_hi, W_hf, W_hg, W_ho], dim=1))

        # Forget gate init to 1
        bias = torch.Tensor(4*hidden_dim).zero_()
        bias[hidden_dim:2*hidden_dim] = 1
        self.bias = Parameter(bias)

        if layer_norm:
            self.ln1 = nn.LayerNorm(4 * hidden_dim)
            self.ln2 = nn.LayerNorm(4 * hidden_dim)
            self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, hidden):
        h, c = hidden

        if self.layer_norm:
            gates = self.ln1(torch.mm(h, self.W_h)) + \
                    self.ln2(torch.mm(x, self.W_i)) + \
                    self.bias
        else:
            gates = torch.mm(h, self.W_h) + torch.mm(x, self.W_i) + self.bias

        i, f, g, o = gates.chunk(4, dim=1)

        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        if self.layer_norm:
            c = self.ln3(c)
        h = o.sigmoid() * c.tanh()

        return h, c


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_norm=True):
        """
        GRU class with layer normalization option.

        Parameters
        ----------
        input_dim : int
            Dimensionality of GRU input.

        hidden_dim : int
            Dimensionality of GRU hidden state.

        layer_norm : bool
            Whether to use layer normalized version of GRU.
        """
        super(GRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = layer_norm

        self.gru_cell = GRUCell(
            input_dim, hidden_dim, layer_norm=self.layer_norm)

    def forward(self, x):
        h = torch.zeros(
            x.size(0), self.hidden_dim, dtype=torch.float32).to(x.device)

        x = x.chunk(x.size(1), dim=1)

        H = []
        for x_t in x:
            h = self.gru_cell(x_t.squeeze(1), h)
            H += [h]

        return torch.stack(H, 1), h


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_norm=True):
        """
        GRU cell class with layer normalization option.

        Parameters
        ----------
        input_dim : int
            Dimensionality of GRU cell input.

        hidden_dim : int
            Dimensionality of GRU cell hidden state.

        layer_norm : bool
            Whether to use layer normalized version of GRU cell.
        """
        super(GRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = layer_norm

        W_iz = Parameter(torch.Tensor(input_dim, hidden_dim))
        W_ir = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W = Parameter(torch.Tensor(input_dim, hidden_dim))
        # Xavier init input-to-hidden
        for w in [W_iz, W_ir, self.W]:
            init.xavier_uniform_(w.data)
        self.W_i = torch.cat([W_iz, W_ir], dim=1)

        W_hz = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        W_hr = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.U = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        # Orthogonal init hidden-to-hidden
        for w in [W_hz, W_hr, self.U]:
            init.orthogonal_(w.data)
        self.W_h = torch.cat([W_hz, W_hr], dim=1)

        if self.layer_norm:
            self.ln1 = nn.LayerNorm(2 * hidden_dim)
            self.ln2 = nn.LayerNorm(2 * hidden_dim)
            self.ln3 = nn.LayerNorm(hidden_dim)
            self.ln4 = nn.LayerNorm(hidden_dim)

    def forward(self, x, h):
        if self.layer_norm:
            gates = self.ln1(torch.mm(h, self.W_h)) + \
                    self.ln2(torch.mm(x, self.W_i))
        else:
            gates = torch.mm(h, self.W_h) + torch.mm(x, self.W_i)

        z, r = gates.chunk(2, dim=1)

        if self.layer_norm:
            hh = torch.tanh(
                self.ln3(torch.mm(x, self.W)) +
                r.sigmoid() * self.ln4(torch.mm(h, self.U)))
        else:
            hh = torch.tanh(
                torch.mm(x, self.W) +
                r.sigmoid() * torch.mm(h, self.U))

        h = (1 - z.sigmoid()) * h + z.sigmoid() * hh

        return h
