import argparse
import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm


def continuous_mask(v):
	try:
		tmp = float(v)
		return True
	except:
		return False


def quantize_events(in_dir, out_dir, data, V, n_bins):
	"""Quantize events based on a dictionary labels to values.

	Parameters
	----------
	in_dir: str
		Directory containing subject sub-directories.

	out_dir: str
		Directory to contain BINNED subject sub-directories.

	data: str
		Data directory to save token to index dictionary.

	V: dict
		Value dictionary (e.g. {'Heart Rate': np.array([72, 84,...]), ...})

	n_bins: int
		Number of bins to quantize 
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
	patients = list(filter(str.isdigit, os.listdir(in_dir)))
	token_list = []

	for patient in tqdm(patients):
		if not os.path.exists(os.path.join(out_dir, patient)):
			os.mkdir(os.path.join(out_dir, patient))
		shutil.copy(os.path.join(in_dir, patient, 'stays.csv'),
					os.path.join(out_dir, patient, 'stays.csv'))

		patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
			os.listdir(os.path.join(in_dir, patient))))

		for ts_file in patient_ts_files:
			lb_file = ts_file.replace('_timeseries', '')
			shutil.copy(os.path.join(in_dir, patient, lb_file),
						os.path.join(out_dir, patient, lb_file))

			e = pd.read_csv(os.path.join(in_dir, patient, ts_file))

			e['TOKEN'] = e.apply(tokenize, axis=1)
			del e['CONT']

			e.to_csv(os.path.join(out_dir, patient, ts_file), index=False)
			token_list += list(np.unique(e['TOKEN']))

	# Get the unique set of tokens
	token_list = list(set(token_list))
	print(f'{len(token_list)} tokens overall')

	token2index = dict(zip(token_list, list(range(1, len(token_list)+1))))
	np.save(os.path.join(data, f'token2index_{n_bins}-bins.npy'), token2index)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
	parser.add_argument('root', type=str,
						help='Directory containing subject sub-directories.')
	parser.add_argument('root_binned', type=str,
						help='Directory to contain BINNED subject sub-directories.')
	parser.add_argument('-n', '--n_bins', type=int,
						default=20,
						help='Number of percentile bins.')
	parser.add_argument('-D', '--data', type=str,
						default=os.path.join(os.path.dirname(__file__), os.pardir,
											 os.pardir, 'data'),
						help='Data directory to read value dictionary from.')
	parser.add_argument('-d', '--dict', type=str,
						default='value_dict.npy',
						help='Dictionary file name.')
	args, _ = parser.parse_known_args()

	if not os.path.exists(args.root_binned):
		os.makedirs(args.root_binned)

	value_dict = np.load(os.path.join(args.data, args.dict)).item()

	quantize_events(args.root, args.root_binned, args.data, value_dict, args.n_bins)
