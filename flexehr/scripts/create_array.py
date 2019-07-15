import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def add_idx_to_ts(data, partition, n_hours=48, max_len=10000):
	token2index = np.load(args.dict).item()
	patients = list(filter(str.isdigit, os.listdir(os.path.join(data, partition))))

	Y = []
	LOS = []
	tss = []
	ts_paths = []

	for patient in tqdm(patients):
		patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
			os.listdir(os.path.join(args.data, partition, patient))))
		stays = pd.read_csv(os.path.join(args.data, partition, patient, 'stays.csv'))
		n_stays = stays.shape[0]
		assert n_stays > 0, 'no stay info!'

		for i, ts_file in enumerate(patient_ts_files):
			ts = pd.read_csv(os.path.join(args.data, partition, patient, ts_file))

			# Map tokens and pad
			ts['TOKEN'] = ts['TOKEN'].map(token2index)
			ts = ts.values
			ts[np.isnan(ts)] = 0

			if ts.shape[0] < max_len:
				t = np.pad(ts[:, :1], ((0, max_len-ts.shape[0]), (0, 0)),
					mode='constant', constant_values=n_hours)
				x = np.pad(ts[:, 1:], ((0, max_len-ts.shape[0]), (0, 0)),
					mode='constant', constant_values=0)
				ts = np.concatenate((t, x), axis=1)
			else:
				print('\nLong timeseries of length: {}'.format(ts.shape[0]))
				ts = ts[-max_len:]
			tss += [ts.astype(np.float32)]

			Y += [stays['MORTALITY'].iloc[i]]
			LOS += [stays['LOS'].iloc[i]]
			ts_paths += [os.path.join(patient, ts_file)]

	# Save
	np.save('data/X_{}_{}'.format(n_hours, partition), np.stack(tss))
	np.save('data/Y_{}_{}'.format(n_hours, partition), np.array(Y))
	np.save('data/LOS_{}_{}'.format(n_hours, partition), np.array(LOS))
	np.save('data/paths_{}_{}'.format(n_hours, partition), np.array(ts_paths))


def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-r', '--root',
						default=os.path.join('data', 'root_20_48'),
						help='Path to processed directory containing subject sub-directories.')
	parser.add_argument('-d', '--dict',
						default='token2index_20-bins.npy',
						help='Token to index dictionary name.')
	parser.add_argument('-T', '--max-time', type=int,
						default=48,
						help='Maximum number of hours in timeseries.')
	args = parser.parse_args()

	add_idx_to_ts(args.root, 'test')
	add_idx_to_ts(args.root, 'train')


if __name__ == '__main__':
	main()
