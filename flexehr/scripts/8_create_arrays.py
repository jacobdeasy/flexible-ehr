"""Creat patient arrays."""

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def create_arrays(root, token2index, t_hours=48, n_bins=20, max_len=10000):
    """
    Create dictionary containing sequences, outcomes and stay names.

    Parameters
    ----------
    root : str
        Path to root directory.

    token2index : dict
        Dictionary mapping tokens to indices.

    t_hours : int, optional
        Max number of hours in timeseries.

    n_bins : int, optional
        Number of discrete bins.

    max_len: int, optional
        Maximum number of events to allow in a timeseries.
    """
    Y = []
    LOS = []
    tss = []
    ts_paths = []

    for patient in tqdm(os.listdir(root)):
        pdir = os.path.join(root, patient)
        patient_ts_files = list(filter(
            lambda x: x.find(f'_{t_hours}') != -1, os.listdir(pdir)))

        for i, ts_file in enumerate(patient_ts_files):
            ev_file = ts_file.replace(f'_timeseries_{t_hours}', '')
            ts = pd.read_csv(
                os.path.join(pdir, ts_file),
                usecols=['Hours', f'TOKEN_{n_bins}'])
            ev = pd.read_csv(os.path.join(pdir, ev_file))

            # Map tokens and pad
            ts[f'TOKEN_{n_bins}'] = ts[f'TOKEN_{n_bins}'].map(token2index)
            ts = ts.values.astype(np.float32)
            if ts.shape[0] < max_len:
                t = np.pad(
                    ts[:, :1], ((0, max_len-ts.shape[0]), (0, 0)),
                    mode='constant', constant_values=t_hours)
                x = np.pad(
                    ts[:, 1:], ((0, max_len-ts.shape[0]), (0, 0)),
                    mode='constant', constant_values=0)
                ts = np.concatenate((t, x), axis=1)
            else:
                print(f'Long timeseries of length: {ts.shape[0]}')
                ts = ts[-max_len:]

            tss += [ts]
            Y += [ev['Mortality'].iloc[0]]
            LOS += [ev['Length of Stay'].iloc[0]]
            ts_paths += [os.path.join(patient, ts_file)]

    # Save
    arrs = {
        'X':     np.stack(tss),
        'Y':     np.array(Y),
        'LOS':   np.array(LOS),
        'paths': np.array(ts_paths)
    }
    np.save(os.path.join(root+'_dicts', f'{t_hours}_{n_bins}_arrs'), arrs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Array creation for training.')
    parser.add_argument('root',
                        help='Processed directory of subjects.')
    parser.add_argument('-t', '--t-hours',
                        type=int, default=48,
                        help='Maximum number of hours to allow in timeseries.')
    parser.add_argument('-n', '--n-bins',
                        type=int, default=10,
                        help='Number of percentile bins.')
    args = parser.parse_args()

    # Token to index dictionary
    t2i = np.load(
        os.path.join(args.root+'_dicts', f'{args.t_hours}_{args.n_bins}.npy'),
        allow_pickle=True).item()

    create_arrays(args.root, t2i, t_hours=args.t_hours, n_bins=args.n_bins)
