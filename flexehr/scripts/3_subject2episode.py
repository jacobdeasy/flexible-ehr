"""Extract episode information from validated subject directories."""

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def subject2episode(root):
    for subject_dir in tqdm(os.listdir(root)):
        # Read in tables
        stays = pd.read_csv(
            os.path.join(root, subject_dir, 'stays.csv'),
            header=0, index_col=0)
        for c in ['INTIME', 'OUTTIME', 'DOB', 'DOD', 'DEATHTIME']:
            stays[c] = pd.to_datetime(stays[c])
        stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)

        events = pd.read_csv(
            os.path.join(root, subject_dir, 'events.csv'),
            header=0, index_col=None)
        if events.shape[0] == 0:
            print(subject_dir + ' no valid events!')
            continue
        events['CHARTTIME'] = pd.to_datetime(events['CHARTTIME'])
        events['HADM_ID'] = events['HADM_ID'].fillna(value=-1).astype(int)
        events['ICUSTAY_ID'] = events['ICUSTAY_ID'].fillna(value=-1).astype(int)
        events['VALUEUOM'] = events['VALUEUOM'].fillna('').astype(str)

        # Assemble episodic data
        ep_data = {
            'Icustay':        stays['ICUSTAY_ID'],
            'Age':            stays['AGE'],
            'Length of Stay': stays['LOS'],
            'Mortality':      stays['MORTALITY']
        }
        ep_data.update({
            'Gender': stays['GENDER'].fillna('').apply(
                lambda s: g_map[s] if s in g_map else g_map['OTHER'])})
        ep_data = pd.DataFrame(ep_data).set_index('Icustay')
        ep_data = ep_data[['Gender', 'Age', 'Length of Stay', 'Mortality']]

        for i in range(stays.shape[0]):
            stay_id = stays['ICUSTAY_ID'].iloc[i]
            intime = stays['INTIME'].iloc[i]
            outtime = stays['OUTTIME'].iloc[i]

            idx = events['ICUSTAY_ID'] == stay_id
            if intime is not None and outtime is not None:
                idx = idx | \
                    ((events['CHARTTIME'] >= intime) &
                     (events['CHARTTIME'] <= outtime))
            episode = events.loc[idx]
            del episode['ICUSTAY_ID']

            if episode.shape[0] == 0:
                print(subject_dir + ' (no data!)')
                continue

            episode['HOURS'] = (episode['CHARTTIME'] - intime).apply(
                lambda s: s / np.timedelta64(1, 's')) / 60. / 60
            del episode['CHARTTIME']
            episode = episode.set_index('HOURS').sort_index(axis=0)

            # Save timeseries
            episode.to_csv(
                os.path.join(
                    root, subject_dir, f'episode{i+1}_timeseries.csv'),
                index_label='Hours')
            # Save generic episode data
            ep_data.loc[ep_data.index == stay_id].to_csv(
                os.path.join(root, subject_dir, 'episode%d.csv' % (i+1)),
                index_label='Icustay')


# HELPERS
g_map = {
    'F': 1,
    'M': 2,
    'OTHER': 3,
    '': 0
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract episodes from per-subject data.')
    parser.add_argument(
        'root', type=str,
        help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()

    subject2episode(args.root)
