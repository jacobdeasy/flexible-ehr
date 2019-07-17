import argparse
import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm


def truncate_timeseries(in_dir, out_dir, t_hours=48, eps=1e-6):
	"""Function to truncate timeseries to desired ICU stay length.

	Parameters
	----------
	in_dir: str
		Directory of subject directories.

	t_hours: int
		Max length of ICU stay data. 

	eps: float
		Truncation boundaries tolerance.
	"""
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	patients = os.listdir(in_dir)

	for patient in tqdm(patients):
		stays = pd.read_csv(os.path.join(in_dir, patient, 'stays.csv'))
		patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
			os.listdir(os.path.join(in_dir, patient))))

		for i, ts_file in enumerate(patient_ts_files):
			ts = pd.read_csv(os.path.join(in_dir, patient, ts_file),
				usecols=['Hours', 'ITEMID', 'VALUE', 'VALUEUOM'])
			ev_file = ts_file.replace('_timeseries', '')
			events = pd.read_csv(os.path.join(in_dir, patient, ev_file))

			# Keep timeseries with LOS > time
			if events.shape[0] == 0:
				continue
			los = 24.0 * events['Length of Stay'].iloc[0]
			if pd.isnull(los):
				print('\n(length of stay is missing)', patient, ts_file)
				continue
			if los < t_hours - eps:
				continue

			# Clip time series
			ts = ts[(-eps < ts['Hours']) & (ts['Hours'] < (t_hours+eps))]

			if ts.shape[0] == 0:
				print('\n\t(no events in ICU) ', patient, ts_file)
				continue

			# Save
			if not os.path.exists(os.path.join(out_dir, patient)):
				os.mkdir(os.path.join(out_dir, patient))
			shutil.copy(os.path.join(in_dir, patient, ev_file),
						os.path.join(out_dir, patient, ev_file))
			ts.to_csv(os.path.join(out_dir, patient, ts_file), index=None)


def main():
	parser = argparse.ArgumentParser(
		description="Truncate timeseries to desired ICU stay length.")
	parser.add_argument('root', type=str,
						help="Directory containing full timeseries.")
	parser.add_argument('-t', '--t-hours', type=int,
						default=48,
						help='Maximum number of hours to allow in timeseries.')
	args = parser.parse_args()

	out_dir = f'{args.root.rstrip(os.sep)}_{args.t_hours}'

	truncate_timeseries(args.root, out_dir, t_hours=args.t_hours)


if __name__ == '__main__':
	main()
