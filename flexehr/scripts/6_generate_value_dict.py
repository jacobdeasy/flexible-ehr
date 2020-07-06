"""Generate dictionary of arrays for each continuous variable."""

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


# def continuous_mask(v):
#     """Function to check is variable is continuous.

#     Parameters
#     ----------
#     v : str
#         Variable to be checked.
#     """
#     if str(v) == 'nan':
#         return False
#     else:
#         try:
#             v = float(v)
#             return True
#         except:
#             return False

def try_to_float(v):
    try:
        return float(v)
    except:
        return str(v)


def generate_value_dict(root, seed=0):
    """Generate a dictionary of Val_keys to arrays for continuous variables.

    Parameters
    ----------
    root: str
        Path to root directory.

    seed: int
        Random seed.
    """
    train_files = pd.read_csv(os.path.join(root, 'numpy', f'{seed}-train.csv'))['Paths']

    d = {}

    for f in tqdm(train_files):
        ts = pd.read_csv(os.path.join(root, f))

        for i, isna in enumerate(ts['VALUE'].isna()):
            k = ts.loc[i, 'ITEMID_UOM']
            v = ts.loc[i, 'VALUE']

            if k not in d.keys():
                d[k] = {}
                d[k]['disc'] = []
                d[k]['cont'] = []

            if isna:
                if 'nan' not in d[k]['disc']:
                    d[k]['disc'] += [str(v)]
            else:
                v = try_to_float(v)
                if isinstance(v, str):
                    d[k]['disc'] += [v]
                else:
                    d[k]['cont'] += [v]

    for k, v in d.items():
        if 'cont' in d[k]:
            d[k]['cont'] = np.array(d[k]['cont'])

    if not os.path.exists(os.path.join(root, 'numpy')):
        os.makedirs(os.path.join(root, 'numpy'))
    np.save(os.path.join(root, 'numpy', f'48-{seed}'), d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate value dictionary from subject data.')
    parser.add_argument('root',
                        type=str,
                        help='Root directory.')
    parser.add_argument('-s', '--seeds',
                        nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='seed for data split')
    args, _ = parser.parse_known_args()

    for s in args.seeds:
        generate_value_dict(args.root, seed=s)
