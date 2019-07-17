import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def create_arrays(data, token2index, t_hours=48, n_bins=20, max_len=10000):
	"""Create dictionary containing sequences, outcomes and stay names.

	Parameters
	----------
	data: str
		Path to data directory.

	token2index: dict
		Dictionary mapping tokens to indices.

	t_hours: int
		Max number of hours in timeseries.

	n_bins: int
		Number of discrete bins.

	max_len: 10000
		Maximum number of events to allow in a timeseries.
	"""
	root = os.path.join(data, f'root_{t_hours}_{n_bins}')
	patients = os.listdir(root)

	Y = []
	LOS = []
	tss = []
	ts_paths = []

	for patient in tqdm(patients):
		patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
			os.listdir(os.path.join(root, patient))))

		for i, ts_file in enumerate(patient_ts_files):
			ts = pd.read_csv(os.path.join(root, patient, ts_file),
							 usecols=['Hours', 'TOKEN'])
			ev_file = ts_file.replace('_timeseries', '')
			ev = pd.read_csv(os.path.join(root, patient, ev_file))

			# Map tokens and pad
			ts['TOKEN'] = ts['TOKEN'].map(token2index)
			ts = ts.values
			ts[np.isnan(ts)] = 0

			if ts.shape[0] < max_len:
				t = np.pad(ts[:, :1], ((0, max_len-ts.shape[0]), (0, 0)),
					mode='constant', constant_values=t_hours)
				x = np.pad(ts[:, 1:], ((0, max_len-ts.shape[0]), (0, 0)),
					mode='constant', constant_values=0)
				ts = np.concatenate((t, x), axis=1)
			else:
				print(f'Long timeseries of length: {ts.shape[0]}')
				ts = ts[-max_len:]

			tss += [ts.astype(np.float32)]
			Y += [ev['Mortality'].iloc[0]]
			LOS += [ev['Length of Stay'].iloc[0]]
			ts_paths += [os.path.join(patient, ts_file)]

	# Save
	arrs = {'X':     np.stack(tss),
			'Y':     np.array(Y),
			'LOS':   np.array(LOS),
			'paths': np.array(ts_paths)}

	np.save(os.path.join(data, f'arrs_{t_hours}_{n_bins}'), arrs)


def main():
	parser = argparse.ArgumentParser(description='Array creation for training.')
	parser.add_argument('data',
						help='Processed directory of subjects.')
	parser.add_argument('-t', '--t-hours', type=int,
						default=48,
						help='Maximum number of hours to allow in timeseries.')
	parser.add_argument('-n', '--n-bins', type=int,
						default=20,
						help='Number of percentile bins.')
	parser.add_argument('-m', '--max_len', type=int,
						default=10000,
						help='Maximum number of events to allow in a timeseries.')
	args = parser.parse_args()

	token2index = f'token2index_{args.t_hours}_{args.n_bins}.npy'
	token2index = np.load(os.path.join(args.data, token2index)).item()

	create_arrays(args.data, token2index,
				  t_hours=args.t_hours, n_bins=args.n_bins, max_len=args.max_len)


if __name__ == '__main__':
	main()
