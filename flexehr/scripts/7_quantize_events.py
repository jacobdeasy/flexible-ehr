"""Quantize continuous events by percentile using constructed dict."""

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def quantize_events(root, n_bins, V, seed=0):
    """Quantize events based on a dictionary labels to values.

    Parameters
    ----------
    data : str
        Path to data directory.

    n_bins : int
        Number of discrete bins.

    V: dict
        Value dictionary e.g. {heart_rate_itemid: np.array([72, 84,...]), ...}

    seed : int
        Random seed.
    """
    # To avoid multiple bins per value,
    # remove all {key: values} pairs where: |unique(values)| < n_bins
    print(len(V))
    V_full = V.copy()
    for key, vals in V_full.items():
        if len(np.unique(vals)) < n_bins:
            del V[key]
    print(len(V))

    # Generate percentiles
    print('\nGenerating percentiles...')
    P = []
    for vals in tqdm(V.values()):
        P += [np.percentile(
            vals, np.arange(0, 100 + (100 // n_bins), 100 // n_bins))]
    P = dict(zip(V.keys(), P))

    def tokenize(row):
        """
        Unseen (ITEM, UOM) pairs mapped to <UNKNOWN> token.
        Seen item_uom pairs binned if continuous.
        """
        if row['ITEMID_UOM'] in V_full:
            if row['CONT'] and row['ITEMID_UOM'] in P:
                pctiles = P[row['ITEMID_UOM']]
                posdiff = ((pctiles - float(row['VALUE'])) > 0).astype(int)
                pct = (posdiff[1:] - posdiff[:-1]).argmax()
                return row['ITEMID_UOM'] + ': ' + str(pct)
            else:
                return row['ITEMID_UOM'] + ': ' + str(row['VALUE'])
        else:
            return '<UNKNOWN>'

    print('Creating tokens based on percentiles...')
    train_files = pd.read_csv(os.path.join(root, 'numpy', f'{seed}-train.csv'))['Paths']
    valid_info = pd.read_csv(os.path.join(root, 'numpy', f'{seed}-valid.csv'))['Paths']
    test_info = pd.read_csv(os.path.join(root, 'numpy', f'{seed}-test.csv'))['Paths']

    ### SOME ARE IN SOME SPLITS, OTHERS AREN'T

    token_list = []
    # for f in tqdm(train_files):
    for f in train_files:
        f = f[:-4]+f'-{seed}.csv'
        print(f)
        ts = pd.read_csv(os.path.join(root, f))
        ts[f'TOKEN_{n_bins}'] = ts.apply(tokenize, axis=1)
        ts.to_csv(os.path.join(root, f), index=False)
        token_list += list(np.unique(ts[f'TOKEN_{n_bins}']))

        assert True == False

    for f in tqdm(valid_files):
        ts = pd.read_csv(os.path.join(root, f))
        ts[f'TOKEN_{n_bins}'] = ts.apply(tokenize, axis=1)
        ts.to_csv(os.path.join(root, f[:-4]+f'-{seed}.csv'), index=False)

    for f in tqdm(test_files):
        ts = pd.read_csv(os.path.join(root, f))
        ts[f'TOKEN_{n_bins}'] = ts.apply(tokenize, axis=1)
        ts.to_csv(os.path.join(root, f[:-4]+f'-{seed}.csv'), index=False)

    # Get the unique set of tokens
    token_list = list(set(token_list))
    print(f'{len(token_list)} tokens overall')

    # Save
    token2index = dict(zip(token_list, list(range(1, len(token_list)+1))))
    np.save(os.path.join(root, 'numpy', f'48-{seed}-{n_bins}'), token2index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quantize events using value dictionary.')
    parser.add_argument('root',
                        type=str,
                        help='Root directory.')
    parser.add_argument('-n', '--n-bins',
                        type=int, default=10,
                        help='Number of percentile bins.')
    parser.add_argument('-s', '--seeds',
                        nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='seed for data split')
    args, _ = parser.parse_known_args()

    for s in args.seeds:
        value_dict = np.load(
            os.path.join(args.root, 'numpy', f'48-{s}.npy'),
            allow_pickle=True).item()

        quantize_events(args.root, args.n_bins, value_dict, seed=s)
