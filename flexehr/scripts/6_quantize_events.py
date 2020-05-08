"""Quantize continuous events by percentile using constructed dict."""

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def quantize_events(root, t_hours, n_bins, V):
    """Quantize events based on a dictionary labels to values.

    Parameters
    ----------
    data : str
        Path to data directory.

    t_hours : int
        Max length of ICU stay data

    n_bins : int
        Number of discrete bins.

    V: dict
        Value dictionary e.g. {'Heart Rate': np.array([72, 84,...]), ...}
    """
    print('\nGenerating percentiles...')
    P = []
    for i, vals in enumerate(V.values()):
        P += [np.percentile(
            vals, np.arange(0, 100+(100//n_bins), 100//n_bins))]
    P = dict(zip(V.keys(), P))

    def tokenize(row):
        if row['CONT']:
            pctiles = P[row['Val_key']]
            posdiff = ((pctiles - float(row['VALUE'])) > 0).astype(int)
            pct = (posdiff[1:] - posdiff[:-1]).argmax()
            return row['Val_key'] + ': ' + str(pct)
        else:
            return row['Val_key'] + ': ' + row['VALUE']

    print('Creating tokens based on percentiles...')
    token_list = []
    for patient in tqdm(os.listdir(root)):
        pdir = os.path.join(root, patient)
        patient_ts_files = list(filter(
            lambda x: x.find(
                'timeseries_%d' % t_hours) != -1, os.listdir(pdir)))

        for ts_file in patient_ts_files:
            ts = pd.read_csv(os.path.join(pdir, ts_file))
            ts[f'TOKEN_{n_bins}'] = ts.apply(tokenize, axis=1)
            ts.to_csv(os.path.join(pdir, ts_file), index=False)
            token_list += list(np.unique(ts[f'TOKEN_{n_bins}']))

    # Get the unique set of tokens
    token_list = list(set(token_list))
    print(f'{len(token_list)} tokens overall')

    # Save
    token2index = dict(zip(token_list, list(range(1, len(token_list)+1))))
    np.save(
        os.path.join(root+'_dicts', '%d_%d' % (t_hours, n_bins)),
        token2index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quantize events using value dictionary.')
    parser.add_argument('root', type=str,
                        help='Root directory.')
    parser.add_argument('-t', '--t-hours', type=int,
                        default=48,
                        help='Maximum number of hours to allow in timeseries.')
    parser.add_argument('-n', '--n-bins', type=int,
                        default=10,
                        help='Number of percentile bins.')
    args, _ = parser.parse_known_args()

    value_dict = np.load(
        os.path.join(args.root+'_dicts', '%d.npy' % args.t_hours),
        allow_pickle=True).item()

    quantize_events(args.root, args.t_hours, args.n_bins, value_dict)
