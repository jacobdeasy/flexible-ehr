import argparse
import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm


def quantize_events(data, t_hours, n_bins, V):
	"""Quantize events based on a dictionary labels to values.

	Parameters
	----------
	data: str
		Path to data directory.

	t_hours: int
		Max length of ICU stay data

	n_bins: int
		Number of discrete bins.

	V: dict
		Value dictionary e.g. {'Heart Rate': np.array([72, 84,...]), ...}
	"""
	print('\nGenerating percentiles...')
	P = []
	for i, vals in enumerate(V.values()):
		P += [np.percentile(vals,
							np.arange(0, 100+(100//n_bins), 100//n_bins))]
	P = dict(zip(V.keys(), P))

	def tokenize(row):
		"""Helper function to tokenize a `LABEL` and `VALUE` combination."""
		label = row['LABEL']
		value = row['VALUE']
		if row['CONT']:
			pctiles = P[label]
			posdiff = ((pctiles - float(value)) > 0).astype(int)
			pct = (posdiff[1:] - posdiff[:-1]).argmax()
			return f'{label}_{pct}'
		else:
			return f'{label} {value}'

	print('\nCreating tokens based on percentiles...')
	in_dir = os.path.join(data, f'root_{t_hours}')
	out_dir = os.path.join(data, f'root_{t_hours}_{n_bins}')

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	patients = os.listdir(in_dir)
	token_list = []

	for patient in tqdm(patients):
		if not os.path.exists(os.path.join(out_dir, patient)):
			os.mkdir(os.path.join(out_dir, patient))

		patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
			os.listdir(os.path.join(in_dir, patient))))

		for ts_file in patient_ts_files:
			ev_file = ts_file.replace('_timeseries', '')
			shutil.copy(os.path.join(in_dir, patient, ev_file),
						os.path.join(out_dir, patient, ev_file))

			e = pd.read_csv(os.path.join(in_dir, patient, ts_file))

			e['TOKEN'] = e.apply(tokenize, axis=1)
			del e['CONT']

			e.to_csv(os.path.join(out_dir, patient, ts_file), index=False)
			token_list += list(np.unique(e['TOKEN']))

	# Get the unique set of tokens
	token_list = list(set(token_list))
	print(f'{len(token_list)} tokens overall')

	token2index = dict(zip(token_list, list(range(1, len(token_list)+1))))
	np.save(os.path.join(data, f'token2index_{t_hours}_{n_bins}.npy'), token2index)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Quantize events using value dictionary.')
	parser.add_argument('data', type=str,
						help='Data directory.')
	parser.add_argument('-t', '--t-hours', type=int,
						default=48,
						help='Maximum number of hours to allow in timeseries.')
	parser.add_argument('-n', '--n-bins', type=int,
						default=20,
						help='Number of percentile bins.')
	parser.add_argument('-d', '--dict', type=str,
						default='value_dict.npy',
						help='Dictionary file name.')
	args, _ = parser.parse_known_args()

	value_dict = np.load(os.path.join(args.data, args.dict)).item()

	quantize_events(args.data, args.t_hours, args.n_bins, value_dict)
