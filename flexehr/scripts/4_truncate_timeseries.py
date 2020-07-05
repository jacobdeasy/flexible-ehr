"""Extract episode information from validated subject directories."""

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def truncate_timeseries(in_dir, t_hours=48):
    patients = os.listdir(in_dir)

    patient_count = 0
    stay_count = 0
    event_count = 0

    itemids = []
    uoms = []
    itemid_uoms = []

    for patient in tqdm(patients):
        pdir = os.path.join(in_dir, patient)
        patient_ts_files = list(filter(
            lambda x: x.find('timeseries.csv') != -1, os.listdir(pdir)))

        stay_count_inner = 0

        for i, ts_file in enumerate(patient_ts_files):
            ev_file = ts_file.replace('_timeseries', '')
            events = pd.read_csv(os.path.join(pdir, ev_file))

            # Ignore certain patients
            if events.shape[0] == 0:
                print('Events shape is 0.')
                continue
            los = 24.0 * events['Length of Stay'].iloc[0]
            if pd.isnull(los):
                print('Length of stay is missing.', patient, ts_file)
                continue
            if los < t_hours:
                continue

            # Clip time series
            ts = pd.read_csv(
                os.path.join(pdir, ts_file),
                usecols=['Hours', 'ITEMID', 'VALUE', 'VALUEUOM'])
            ts = ts.loc[(0 <= ts['Hours']) & (ts['Hours'] < (t_hours))]
            if ts.shape[0] == 0:
                print('\tNo events in ICU.', patient, ts_file)
                continue

            # Val_key = ITEMID_VALUEUOM
            ts['ITEMID'] = ts['ITEMID'].astype(str)
            ts['VALUEUOM'] = ts['VALUEUOM'].astype(str)
            ts['ITEMID_UOM'] = list(zip(ts['ITEMID'], ts['VALUEUOM']))

            itemids += list(ts['ITEMID'].unique())
            uoms += list(ts['VALUEUOM'].unique())
            itemid_uoms += list(ts['ITEMID_UOM'].unique())

            ts.drop(['ITEMID', 'VALUEUOM'], axis=1, inplace=True)
            ts.to_csv(os.path.join(pdir, ts_file[:-4]+f'_{t_hours}.csv'), index=None)

            stay_count_inner += 1
            stay_count += 1
            event_count += len(ts)

        if stay_count_inner > 0:
            patient_count += 1

    # Output info
    print(f'{patient_count} patients')
    print(f'{stay_count} stays')
    print(f'{event_count} events')

    itemids = list(set(itemids))
    uoms = list(set(uoms))
    itemid_uoms = list(set(itemid_uoms))
    print(f'{len(itemids)} unique ITEMIDs.')
    print(f'{len(uoms)} unique UOMs.')
    print(f'{len(itemid_uoms)} unique pairs (ITEMID, UOM).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Truncate timeseries to desired ICU stay length.")
    parser.add_argument('root', type=str,
                        help="Directory containing full timeseries.")
    parser.add_argument('-t', '--t-hours', type=int,
                        default=48,
                        help='Maximum number of hours to allow in timeseries.')
    args = parser.parse_args()

    truncate_timeseries(args.root, t_hours=args.t_hours)
