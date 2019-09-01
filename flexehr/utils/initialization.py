import torch
from torch import nn


def get_activation_name(activation):
	"""Given a string or a `torch.nn.modules.activation` return the name of the activation."""
	if isinstance(activation, str):
		return activation

	mapper = {nn.ReLU: 'relu', nn.Tanh: 'tanh', nn.Sigmoid: 'sigmoid'}
	for k, v in mapper.items():
		if isinstance(activation, k):
			return k

	raise ValueError(f'Unknown given activation type: {activation}')


def get_gain(activation):
	"""Given an object of `torch.nn.modules.activation` or an activation name
	return the correct gain."""
	if activation is None:
		return 1

	activation_name = get_activation_name(activation)

	param = None if activation_name != 'leaky_relu' else activation.negative_slope
	gain = nn.init.calculate_gain(activation_name, param)

	return gain


def linear_init(layer, activation='relu'):
	"""Initialize a linear layer.
	Args:
		layer (nn.Linear): parameters to initialize.
		activation (`torch.nn.modules.activation` or str, optional) activation that
			will be used on the `layer`.
	"""
	x = layer.weight

	if activation is None:
		return nn.init.xavier_uniform_(x)

	activation_name = get_activation_name(activation)

	if activation_name == 'relu':
		return nn.init.kaiming_uniform_(x, nonlinearity='relu')
	elif activation_name in ['sigmoid', 'tanh']:
		return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
	if isinstance(module, nn.Linear):
		linear_init(module)
