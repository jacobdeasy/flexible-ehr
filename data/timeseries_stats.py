import argparse
import numpy as np
import os
import pandas as pd
import statistics

from tqdm import tqdm


def los7(args):
    patients = list(filter(str.isdigit, os.listdir(args.root_path)))
    los = []

    for patient in tqdm(patients):
        patient_dir = os.path.join(args.root_path, patient)
        stays = pd.read_csv(os.path.join(patient_dir, 'stays.csv'))
        los += list(stays['LOS'])

    los = np.array(los)
    print((los > 7).sum())
    print(los.shape)


def mean_median_los(args):
    patients = list(filter(str.isdigit, os.listdir(args.root_path)))

    los = []
    ehr_time = []

    for patient in tqdm(patients):
        patient_dir = os.path.join(args.root_path, patient)
        stays = pd.read_csv(os.path.join(patient_dir, 'stays.csv'))
        patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
            os.listdir(patient_dir)))

        los += list(stays['LOS'])

        for i, ts_file in enumerate(patient_ts_files):
            ts = pd.read_csv(os.path.join(patient_dir, ts_file), usecols=['Hours'])
            ehr_time += [(ts['Hours'].iloc[-1] - ts['Hours'][0]) / 24]

    los = [l for l in los if str(l) != 'nan']
    mean_los = sum(los) / len(los)
    median_los = statistics.median(los)
    print('Mean LOS: {:.4f}\tMedian LOS: {:.4f}'.format(mean_los, median_los))

    mean_ehr = sum(ehr_time) / len(ehr_time)
    median_ehr = statistics.median(ehr_time)
    print('Mean EHR: {:.4f}\tMedian EHR: {:.4f}'.format(mean_ehr, median_ehr))


def num_tokens(args):
    n_tokens = []

    for partition in ['test', 'train']:
        patients = list(filter(str.isdigit, os.listdir(
            os.path.join(args.root_path, partition))))
        for patient in tqdm(patients):
            patient_dir = os.path.join(args.root_path, partition, patient)
            patient_ts_files = list(filter(lambda x: x.find('timeseries') != -1, 
                os.listdir(patient_dir)))
            for ts_file in patient_ts_files:
                ts = pd.read_csv(os.path.join(patient_dir, ts_file), usecols=['Hours'])
                n_tokens += [ts.shape[0]]

    mean_n_tokens = sum(n_tokens) / len(n_tokens)
    median_n_tokens = statistics.median(n_tokens)
    print('Mean n_tokens: {:.4f}\tMedian n_tokens: {:.4f}'.format(
        mean_n_tokens, median_n_tokens))


def main():
    parser = argparse.ArgumentParser(
        description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str,
        help="Path to root folder containing train and test sets.")
    args = parser.parse_args()

    los7(args)
    # mean_median_los(args)
    # num_tokens(args)


if __name__ == '__main__':
    main()
