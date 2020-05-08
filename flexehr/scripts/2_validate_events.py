"""Validate events have sufficient information to be used.

Adapted from [1] (https://github.com/YerevaNN/mimic3-benchmarks).

References
----------
Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and
Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series
Data. arXiv:1703.07771
"""

import os
import argparse
import pandas as pd


def is_subject_folder(x):
    """Verify directory contains subject information and not metadata.

    Parameters
    ----------
    x: str
        Directory name.
    """
    return str.isdigit(x)


def main(args):
    """Main function to validate events for HADM_ID and ICUSTAY_ID.
    """

    n_events = 0                   # total number of events
    empty_hadm = 0                 # HADM_ID is empty in events.csv.
    no_hadm_in_stay = 0            # HADM_ID does not appear in stays.csv.
    no_icustay = 0                 # ICUSTAY_ID is empty in events.csv.
    recovered = 0                  # empty ICUSTAY_IDs are recovered
    could_not_recover = 0          # empty ICUSTAY_IDs that are not recovered.
    icustay_missing_in_stays = 0   # ICUSTAY_ID does not appear in stays.csv.

    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))

    for (index, subject) in enumerate(subjects):
        if index % 100 == 0:
            print("processed {} / {} {}\r".format(
                index+1, len(subjects), ' '*10))

        stays_df = pd.read_csv(
            os.path.join(args.subjects_root_path, subject, 'stays.csv'),
            index_col=False, dtype={'HADM_ID': str, "ICUSTAY_ID": str})
        stays_df.columns = stays_df.columns.str.upper()

        # assert that there is no row with empty ICUSTAY_ID or HADM_ID
        assert(not stays_df['ICUSTAY_ID'].isnull().any())
        assert(not stays_df['HADM_ID'].isnull().any())

        # assert there are no repetitions of ICUSTAY_ID or HADM_ID
        # since admissions with multiple ICU stays were excluded
        assert(len(
            stays_df['ICUSTAY_ID'].unique()) == len(stays_df['ICUSTAY_ID']))
        assert(len(
            stays_df['HADM_ID'].unique()) == len(stays_df['HADM_ID']))

        events_df = pd.read_csv(
            os.path.join(args.subjects_root_path, subject, 'events.csv'),
            index_col=False, dtype={'HADM_ID': str, "ICUSTAY_ID": str})
        events_df.columns = events_df.columns.str.upper()
        n_events += events_df.shape[0]

        # drop all events for them HADM_ID is empty
        empty_hadm += events_df['HADM_ID'].isnull().sum()
        events_df = events_df.dropna(subset=['HADM_ID'])

        merged_df = events_df.merge(
            stays_df,
            left_on=['HADM_ID'], right_on=['HADM_ID'], how='left',
            suffixes=['', '_r'], indicator=True)

        # drop all events for which HADM_ID is not listed in stays.csv
        # since there is no way to know the targets of that stay
        no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
        merged_df = merged_df[merged_df['_merge'] == 'both']

        # if ICUSTAY_ID is empty in stays.csv, we try to recover it
        # we exclude all events for which we could not recover ICUSTAY_ID
        cur_no_icustay = merged_df['ICUSTAY_ID'].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, 'ICUSTAY_ID'] = \
            merged_df['ICUSTAY_ID'].fillna(merged_df['ICUSTAY_ID_r'])
        recovered += cur_no_icustay - merged_df['ICUSTAY_ID'].isnull().sum()
        could_not_recover += merged_df['ICUSTAY_ID'].isnull().sum()
        merged_df = merged_df.dropna(subset=['ICUSTAY_ID'])

        # when ICUSTAY_ID is present in events.csv, but not in stays.csv
        # ICUSTAY_ID in events.csv =/= that of stays.csv for the same HADM_ID
        # drop all such events
        icustay_missing_in_stays += (
            merged_df['ICUSTAY_ID'] != merged_df['ICUSTAY_ID_r']).sum()
        merged_df = merged_df[
            (merged_df['ICUSTAY_ID'] == merged_df['ICUSTAY_ID_r'])]

        to_write = merged_df[[
            'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID',
            'VALUE', 'VALUEUOM']]
        to_write.to_csv(
            os.path.join(args.subjects_root_path, subject, 'events.csv'),
            index=False)

    assert(could_not_recover == 0)
    print(f'n_events: {n_events}')
    print(f'empty_hadm: {empty_hadm}')
    print(f'no_hadm_in_stay: {no_hadm_in_stay}')
    print(f'no_icustay: {no_icustay}')
    print(f'recovered: {recovered}')
    print(f'could_not_recover: {could_not_recover}')
    print(f'icustay_missing_in_stays: {icustay_missing_in_stays}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'subjects_root_path',
        type=str, help='Directory containing subject subdirectories.')
    args = parser.parse_args()
    print(args)

    main(args)
