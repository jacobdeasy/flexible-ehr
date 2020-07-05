"""Module containing the train/test split function."""

import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_train_test(root, seed=0):
    """Function to split training/validation/testing sets.

    Parameters
    ----------
    data: str
        Path to data directory.

    seed : int, optional
        Fixed seed for reproducibility purposes, do not change!
    """

    patients = os.listdir(root)
    if 'numpy' in patients:
        patients = patients[:-1]

    ts_paths = []
    mort = []
    LOS = []

    for patient in tqdm(patients):
        pdir = os.path.join(root, patient)
        patient_ts_files = list(filter(
            lambda x: x.find(f'_48') != -1, os.listdir(pdir)))

        for i, ts_file in enumerate(patient_ts_files):
            ev_file = ts_file.replace(f'_timeseries_48', '')
            ev = pd.read_csv(os.path.join(pdir, ev_file))

            mort += [ev['Mortality'].iloc[0]]
            LOS += [ev['Length of Stay'].iloc[0]]
            ts_paths += [os.path.join(patient, ts_file)]

    # Split training/validation/testing sets
    ts_train, ts_test, m_train, m_test, LOS_train, LOS_test = train_test_split(
        ts_paths, mort, LOS,
        test_size=0.1, stratify=mort, random_state=seed)
    ts_train, ts_valid, m_train, m_valid, LOS_train, LOS_valid = train_test_split(
        ts_train, m_train, LOS_train,
        test_size=1000, stratify=m_train, random_state=seed)

    if not os.path.exists(os.path.join(root, 'numpy')):
        os.makedirs(os.path.join(root, 'numpy'))

    # Save info dataframes
    train_df = pd.DataFrame(
        zip(ts_train, m_train, LOS_train),
        columns=['Paths', 'Mortality', 'LOS'])
    valid_df = pd.DataFrame(
        zip(ts_valid, m_valid, LOS_valid),
        columns=['Paths', 'Mortality', 'LOS'])
    test_df = pd.DataFrame(
        zip(ts_test, m_test, LOS_test),
        columns=['Paths', 'Mortality', 'LOS'])
    train_df.to_csv(os.path.join(root, 'numpy', f'{seed}-train.csv'))
    valid_df.to_csv(os.path.join(root, 'numpy', f'{seed}-valid.csv'))
    test_df.to_csv(os.path.join(root, 'numpy', f'{seed}-test.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train test splitter.')
    parser.add_argument('root',
                        type=str,
                        help='Path to patient folders root.')
    parser.add_argument('-s', '--seeds',
                        nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='seed for data split')
    args = parser.parse_args()

    for s in args.seeds:
        split_train_test(args.root, seed=s)
