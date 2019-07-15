"""
Extract episode information from validated subject directories.

Adapted from https://github.com/YerevaNN/mimic3-benchmarks

References
----------
Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and
Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series
Data. arXiv:1703.07771
"""


from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
from tqdm import tqdm

from flexehr.benchmark.subject import (read_stays, read_diagnoses, read_events,
									   get_events_for_stay, add_hours_elpased_to_events)
from flexehr.benchmark.preprocessing import assemble_episodic_data


def main(root):
	"""Main function to separate patient stays.

	Parameters
	----------
	root: str
		Directory containing subject sub-directories.
	"""
	for subject_dir in tqdm(os.listdir(root)):
		dn = os.path.join(root, subject_dir)

		try:
			subject_id = int(subject_dir)
			if not os.path.isdir(dn):
				raise Exception
		except:
			continue

		try:
			stays = read_stays(os.path.join(root, subject_dir))
			diagnoses = read_diagnoses(os.path.join(root, subject_dir))
			events = read_events(os.path.join(root, subject_dir))
		except:
			print(f'{subject_id} error reading from disk!\n')
			continue
		else:
			pass

		episodic_data = assemble_episodic_data(stays, diagnoses)

		if events.shape[0] == 0:
			print(f'{subject_id} no valid events!\n')
			continue

		for i in range(stays.shape[0]):
			stay_id = stays.ICUSTAY_ID.iloc[i]
			intime = stays.INTIME.iloc[i]
			outtime = stays.OUTTIME.iloc[i]

			episode = get_events_for_stay(events, stay_id, intime, outtime)
			if episode.shape[0] == 0:
				print(f'{subject_id} (no data!)')
				continue

			episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
			episodic_data.ix[episodic_data.index==stay_id].to_csv(os.path.join(root, subject_dir, 'episode{}.csv'.format(i+1)), index_label='Icustay')
			columns = list(episode.columns)
			columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
			episode = episode[columns_sorted]
			episode.to_csv(os.path.join(root, subject_dir, 'episode{}_timeseries.csv'.format(i+1)), index_label='Hours')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
	parser.add_argument('root', type=str,
						help='Directory containing subject sub-directories.')
	args, _ = parser.parse_known_args()

	main(args.root)
