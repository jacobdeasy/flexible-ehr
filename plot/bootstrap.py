import argparse
import numpy as np


def bootstrap(x, low, high, n_samples):
	mu = x.mean()
	n = len(x)
	X = np.random.choice(x, size=n_samples*n).reshape(n_samples, n)
	mu_star = X.mean(axis=1)
	d_star = np.sort(mu_star - mu)

	return mu, mu+d_star[int(low*n_samples)], mu+d_star[int(high*n_samples)]


def bootstrap2d(x, low, high, n_samples):
	MU = np.zeros(x.shape[1])
	LB = np.zeros(x.shape[1])
	UB = np.zeros(x.shape[1])

	for i in range(x.shape[1]):
		MU[i], LB[i], UB[i] = bootstrap(x[:, i], low, high, n_samples)

	return MU, LB, UB


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('arr', type=str,
						help='path to array to be bootstrapped')
	parser.add_argument('save', type=str,
						help='path to save output')
	parser.add_argument('--low', type=float, default=0.025,
						help='lower probability bound')
	parser.add_argument('--high', type=float, default=0.975,
						help='upper probability bound')
	parser.add_argument('--n_samples', type=int, default=10000,
						help='number of times to sample with replacement')
	args = parser.parse_args()


	arr = np.load(args.arr)
	if arr.ndim == 1:
		mu, lb, ub = bootstrap(a,
			low=args.low, high=args.high, n_samples=args.n_samples)
	elif arr.ndim == 2:
		mu, lb, ub = bootstrap2d(arr,
			low=args.low, high=args.high, n_samples=args.n_samples)
	else:
		print('arr has >2 dimensions')

	np.save(args.save, np.stack((lb, mu, ub)))


if __name__ == '__main__':
	main()
