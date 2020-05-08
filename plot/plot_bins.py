import argparse, numpy as np, os
import matplotlib as mpl, matplotlib.cm as cm, matplotlib.pyplot as plt
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rc('font', family='serif')


def plot_bins(var, n_bins, bounds=None):
	# Calculate percentiles
	v = np.load(os.path.join('data', 'value_dict.npy')).item()
	p = []
	for i, vals in enumerate(v.values()):
		p += [np.percentile(vals, np.arange(0, 100+(100//n_bins), 100//n_bins))]
	p = dict(zip(v.keys(), p))

	# Plot
	vals = v[var] if bounds is None else v[var][(bounds[0] < v[var]) & (v[var] < bounds[1])]
	counts, bins = np.histogram(vals, bins=100)
	cols = np.digitize(bins[:-1], p[var]) - 1
	cmap = cm.rainbow(np.linspace(0, 1, n_bins))
	plt.bar(bins[:-1], counts, width=bins[1:]-bins[:-1], color=cmap[cols], align='edge')

	plt.xlim(bins[0], bins[-1])
	plt.xlabel(var, fontsize=15)
	plt.xticks(np.linspace(bins[0], bins[-1], 5), fontsize=12)

	plt.ylim(0, max(counts))
	# plt.ylabel('Frequency', fontsize=15)
	plt.yticks(np.linspace(0, max(counts), 5), fontsize=12)

	plt.savefig(os.path.join('figs', f'{var}_{n_bins}bins_dist.pdf'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
	parser.add_argument('var', type=str,
						help='Variable to visualize.')
	parser.add_argument('-n', '--n_bins', type=int,
						default=20,
						help='Number of bins to visualize')
	parser.add_argument('-b', '--bounds', nargs='+',
						default=None,
						help='Lower and upper bounds for plotting purposes')
	args, _ = parser.parse_known_args()

	bounds = [int(b) for b in args.bounds]
	plot_bins(args.var, args.n_bins, bounds)
