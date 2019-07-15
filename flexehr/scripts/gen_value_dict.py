import argparse
import numpy as np
import os
import pandas as pd
import regex as re

from tqdm import tqdm


def continuous_mask(v):
	"""Helper function to check is variable is continuous.

	Parameters
	----------
	v: str
		Variable to be checked.
	"""
	try:
		tmp = float(v)
		return True
	except:
		return False


def gen_value_dict(root, data_dir, item2var):
	"""Generate a dictionary of labels to value arrays from `ITEMID`s and values.

	Parameters
	----------
	root: str
		Path of directory of subjects with sub-directory episodes.

	data_dir: str
		Path of data directory to write value dictionary to.

	item2var: str
		Csv file with ITEMID and variable names.
	"""
	item2var = pd.read_csv('../benchmark/resources/itemid_to_variable_map.csv')
	item2label = item2var.set_index('ITEMID')['MIMIC LABEL'].to_dict()
	patients = list(filter(str.isdigit, os.listdir(root)))

	print('\nExtracting numeric labels and values...')
	d = {}

	for patient in tqdm(patients):
		patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
								os.listdir(os.path.join(root, patient))))

		for ts_file in patient_ts_files:
			e = pd.read_csv(os.path.join(root, patient, ts_file),
				usecols=['Hours', 'ITEMID', 'VALUE', 'VALUEUOM'])

			if e.shape[0] == 0:
				print('events file is empty', patient)

			# Remove missing values
			e = e[e['VALUE'].notnull()]

			# Map item IDs to labels
			e['ITEMID'] = e['ITEMID'].map(item2label)
			e = e.rename(columns={'ITEMID': 'LABEL'})

			# Get mask of continuous readings
			e['CONT'] = e['VALUE'].apply(continuous_mask)

			if e.shape[0] == 0:  # Just in case
				print(patient)

			# Save events file as is
			e.to_csv(os.path.join(root, patient, ts_file), index=False)

			# Accumulate continuous labels and values
			e = e[e['CONT']]
			p_vals = e['VALUE'].values.astype(np.float32)
			p_labs = e['LABEL'].unique()

			for label in p_labs:
				if label not in d.keys():
					d[label] = np.empty(0)
				d[label] = np.concatenate((d[label], p_vals[e['LABEL']==label]))

	np.save(os.path.join(data_dir, 'value_dict'), d)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
	parser.add_argument('root', type=str,
						help='Directory containing subject sub-directories.')
	parser.add_argument('-d', '--data', type=str,
						default=os.path.join(os.pardir, os.pardir, 'data'),
						help='Data directory to write value dictionary to.')
	parser.add_argument('-i', '--item2var', type=str,
                    	default=os.path.join(os.pardir, os.pardir,
                    						 'benchmark/resources/itemid_to_variable_map.csv'),
                    	help='csv file with ITEMID and variable names.')
	args, _ = parser.parse_known_args()

	if not os.path.exists(args.root_binned):
		os.makedirs(args.root_binned)

	main(args.root, args.item2var)
