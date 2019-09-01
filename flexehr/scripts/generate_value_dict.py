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


def generate_value_dict(data, t_hours, item2label):
	"""Generate a dictionary of labels to value arrays from `ITEMID`s and values.

	Parameters
	----------
	data: str
		Path to data directory.

	t_hours: str
		Max length of ICU stay data.

	item2label: dict
		Dictionary mapping `ITEMID`s to variable names.
	"""
	root = os.path.join(data, f'root_{t_hours}')
	patients = os.listdir(root)

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

	np.save(os.path.join(data, f'value_dict_{t_hours}'), d)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate value dictionary from subject data.')
	parser.add_argument('data', type=str,
						help='Data directory to write value dictionary to.')
	parser.add_argument('-t', '--t-hours', type=int,
						default=48,
						help='Maximum number of hours to allow in timeseries.')
	parser.add_argument('-i', '--item2var', type=str,
                    	default=os.path.join(os.path.dirname(__file__), os.pardir,
                    						 'benchmark/resources/itemid_to_variable_map.csv'),
                    	help='csv file with ITEMID and variable names.')
	args, _ = parser.parse_known_args()

	item2var = pd.read_csv(args.item2var)
	item2label = item2var.set_index('ITEMID')['MIMIC LABEL'].to_dict()

	generate_value_dict(args.data, args.t_hours, item2label)
