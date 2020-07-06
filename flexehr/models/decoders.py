"""
Module containing the decoders.
"""
from torch import nn


DECODER_DICT = {
    'Mortality': 'Binary',
    'MortalityFFNN': 'BinaryFFNN',
    'Readmission': 'Binary',
    'LOS': 'Regression',
    'LOS7': 'Binary'
}


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
            Dimensionality of LSTM hidden state.
        """
        super(DecoderBinary, self).__init__()

        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        y = self.fc(h).squeeze().sigmoid()

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

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        y = self.fc(h).squeeze()

        return y


class DecoderBinaryFFNN(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        """Hidden state decoder for binary classification tasks.

        Parameters
        ----------
        hidden_dim : int
            Dimensionality of LSTM hidden state.

        n_layers : int
            Number of hidden layers in FFNN decoder.
        """
        super(DecoderBinaryFFNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        assert n_layers > 0
        self.fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        for i in range(1, n_layers):
            self.fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, 1))

    def forward(self, h):
        y = self.fc(h).squeeze().sigmoid()

        return y
