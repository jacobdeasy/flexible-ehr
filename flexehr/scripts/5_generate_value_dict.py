"""Generate dictionary of arrays for each continuous variable."""

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def continuous_mask(v):
    """Helper function to check is variable is continuous.

    Parameters
    ----------
    v : str
        Variable to be checked.
    """
    if v == np.nan:
        return False
    else:
        try:
            v = float(v)
            return True
        except:
            return False


def generate_value_dict(root, t_hours):
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
    d = {}

    for patient in tqdm(os.listdir(root)):
        pdir = os.path.join(root, patient)
        patient_ts_files = list(filter(
            lambda x: x.find('%d.csv' % t_hours) != -1, os.listdir(pdir)))

        for ts_file in patient_ts_files:
            ts = pd.read_csv(
                os.path.join(pdir, ts_file),
                usecols=['Hours', 'Val_key', 'VALUE'])
            if ts.shape[0] == 0:
                print('timeseries file is empty', patient)
                continue

            # Get mask of continuous readings
            ts['CONT'] = ts['VALUE'].apply(continuous_mask)
            ts.to_csv(os.path.join(pdir, ts_file), index=False)

            # Accumulate continuous labels and values
            ts = ts.loc[ts['CONT']]
            p_labs = ts['Val_key'].unique()
            p_vals = ts['VALUE'].values.astype(np.float32)
            for label in p_labs:
                if label not in d.keys():
                    d[label] = np.empty(0)
                d[label] = np.concatenate(
                    (d[label], p_vals[ts['Val_key'] == label]))

    if not os.path.exists(root+'_dicts'):
        os.makedirs(root+'_dicts')
    np.save(os.path.join(root+'_dicts', str(t_hours)), d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate value dictionary from subject data.')
    parser.add_argument('root',
                        type=str,
                        help='Root directory.')
    parser.add_argument('-t', '--t-hours',
                        type=int, default=48,
                        help='Maximum number of hours allowed in timeseries.')
    args, _ = parser.parse_known_args()

    generate_value_dict(args.root, args.t_hours)
