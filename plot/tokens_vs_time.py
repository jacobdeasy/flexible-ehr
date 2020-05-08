"""Plotting module for Figure 2 in [1]

References
----------
[1]
"""

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rc('font', family='serif')


def tokens_vs_time(t_hours):
    root = os.path.join('data', f'root_{t_hours}')

    C = np.zeros((3, 48))
    item2var = pd.read_csv(
        os.path.join(
            'flexehr', 'benchmark', 'resources', 'itemid_to_variable_map.csv'))
    item2eventtype = item2var.set_index('MIMIC LABEL')['LINKSTO'].to_dict()
    tables = ['chartevents', 'labevents', 'outputevents']

    for patient in tqdm(os.listdir(root)):
        patient_files = os.listdir(os.path.join(root, patient))
        ts_files = [s for s in patient_files if 'timeseries' in s]

        for ts_file in ts_files:
            ts = pd.read_csv(os.path.join(root, patient, ts_file))
            label_table = ts['LABEL'].map(item2eventtype)

            for i, table in enumerate(tables):
                event_inds = label_table == table

                for j in range(48):
                    C[i, j] += len(ts.loc[(j < ts['Hours']) & (ts['Hours'] < (j+1))].loc[event_inds])

    C /= 21139

    plt.bar(np.arange(48), C[0, :],
        align='edge', width=1, label='Chart Events')
    plt.bar(np.arange(48), C[1, :],
        bottom=C[0, :], align='edge', width=1, label='Lab Events')
    plt.bar(np.arange(48), C[2, :],
        bottom=C[0, :]+C[1,:], align='edge', width=1, label='Output Events')

    plt.xlim(0, 48)
    plt.xticks([0, 12, 24, 36, 48])
    plt.ylim(25, 50)
    plt.xlabel('In-Hospital Hours After Admission', fontsize=15)
    plt.ylabel('Mean Token Count per Patient', fontsize=15)
    plt.legend()

    plt.savefig(os.path.join('figs', f'tokens_vs_time_{t_hours}.pdf'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot dynamic mortity probability.')
    parser.add_argument('-t', '--t-hours', type=int,
                        default=48,
                        help='Dataset hour limit.')
    args = parser.parse_args()

    tokens_vs_time(args.t_hours)
