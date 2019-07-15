import argparse
import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm


def truncate_timeseries(in_dir, partition, n_hours=48, eps=1e-6):
	"""Function to truncate timeseries to desired ICU stay length.

	Parameters
	----------
	in_dir: str
		Path to full timeseries.

	partiion: str
		'train' or 'test' partition

	n_hours: int
		Max length of ICU stay data. 

	eps: float
		Allowed tolerance for truncation boundaries.
	"""
	out_dir = f'{in_dir.rstrip(os.sep)}_{n_hours}'
	in_dir = os.path.join(in_dir, partition)
	out_dir = os.path.join(out_dir, partition)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	patients = os.listdir(in_dir)

	for patient in tqdm(patients):
		stays = pd.read_csv(os.path.join(in_dir, patient, 'stays.csv'))
		patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
			os.listdir(os.path.join(in_dir, patient))))

		for i, ts_file in enumerate(patient_ts_files):
			ts = pd.read_csv(os.path.join(in_dir, patient, ts_file),
				usecols=['Hours', 'TOKEN'])

			# Keep timeseries with LOS > time
			los = 24.0 * stays.iloc[i]['LOS']
			if pd.isnull(los):
				print('\n(length of stay is missing)', patient, ts_file)
				continue
			if los < n_hours - eps:
				continue

			# Clip time series
			ts = ts[(-eps < ts['Hours']) & (ts['Hours'] < (n_hours+eps))]

			if ts.shape[0] == 0:
				print('\n\t(no events in ICU) ', patient, ts_file)
				continue

			# Save
			if not os.path.exists(os.path.join(out_dir, patient)):
				os.mkdir(os.path.join(out_dir, patient))
			if i == 0:
				shutil.copy(os.path.join(in_dir, patient, 'stays.csv'),
							os.path.join(out_dir, patient, 'stays.csv'))
			ts.to_csv(os.path.join(out_dir, patient, ts_file), index=None)


def main():
	parser = argparse.ArgumentParser(
		description="Truncate timeseries to desired ICU stay length.")
	parser.add_argument('root_n', type=str,
						default=os.path.join(os.pardir, os.pardir,
											 'data/root_20'),
						help="Directory containing full timeseries.")
	parser.add_argument('-T', '--max-time', type=int,
						default=48,
						help='Maximum number of hours to allow in timeseries.')
	args = parser.parse_args()

	truncate_timeseries(args.root_n, 'test', n_hours=args.max_time)
	truncate_timeseries(args.root_n, 'train', n_hours=args.max_time)


if __name__ == '__main__':
	main()
