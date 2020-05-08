import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_all(model_dir):
	pass


def plotter(model_dir):
	df = pd.read_csv(os.path.join(model_dir, 'train_losses.log'))

	# Loss plot
	df['Train Loss'].plot(color='blue', label='Train')
	df['Valid Loss'].plot(color='orange')
	plt.xlabel('Epoch')
	plt.ylabel('BCE Loss')
	plt.legend()

	# AUROC plot
	plt.figure()
	df['AUROC'].plot(color='orange', label='Valid')
	plt.xlabel('Epoch')
	plt.ylabel('AUROC')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Plot loss and auroc info.')
	parser.add_argument('name', type=str,
						help='Name of model.')
	args, _ = parser.parse_known_args()

	plotter(args.name)
