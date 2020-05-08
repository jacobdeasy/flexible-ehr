"""Extract subject information and events from MIMIC-III csvs."""

import argparse
import csv
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument(
    'mimic3_path', type=str,
    help='Directory containing MIMIC-III CSV files.')
parser.add_argument(
    'output_path', type=str,
    help='Directory where per-subject data should be written.')
parser.add_argument(
    '--event_tables', '-e', type=str, nargs='+',
    default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'],
    help='Tables from which to read events.')
args, _ = parser.parse_known_args()


if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)


# Read in tables
print('Reading in tables...')
pats = pd.read_csv(
    os.path.join(args.mimic3_path, 'PATIENTS.csv'),
    header=0, index_col=0, usecols=['SUBJECT_ID', 'GENDER', 'DOB', 'DOD'])
pats['DOB'] = pd.to_datetime(pats['DOB'])
pats['DOD'] = pd.to_datetime(pats['DOD'])

admits = pd.read_csv(
    os.path.join(args.mimic3_path, 'ADMISSIONS.csv'),
    header=0, index_col=0, usecols=[
        'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME',
        'DEATHTIME', 'ADMISSION_TYPE', 'ETHNICITY', 'DIAGNOSIS'])
admits['ADMITTIME'] = pd.to_datetime(admits['ADMITTIME'])
admits['DISCHTIME'] = pd.to_datetime(admits['DISCHTIME'])
admits['DEATHTIME'] = pd.to_datetime(admits['DEATHTIME'])

stays = pd.read_csv(
    os.path.join(args.mimic3_path, 'ICUSTAYS.csv'),
    header=0, index_col=0)
stays['INTIME'] = pd.to_datetime(stays['INTIME'])
stays['OUTTIME'] = pd.to_datetime(stays['OUTTIME'])

print(
    len(stays['ICUSTAY_ID'].unique()),
    len(stays['HADM_ID'].unique()),
    len(stays['SUBJECT_ID'].unique()))


# Remove icustays with transfers
print('Removing icustays with transfers...')
stays = stays.loc[
    (stays['FIRST_WARDID'] == stays['LAST_WARDID']) &
    (stays['FIRST_CAREUNIT'] == stays['LAST_CAREUNIT'])]
stays = stays.drop(
    ['FIRST_WARDID', 'LAST_WARDID', 'FIRST_CAREUNIT'], axis=1)

print(
    len(stays['ICUSTAY_ID'].unique()),
    len(stays['HADM_ID'].unique()),
    len(stays['SUBJECT_ID'].unique()))


# Merge on subject admission and subject
stays = stays.merge(
    admits,
    left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
stays = stays.merge(
    pats,
    left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


# Filter admissions on number of ICU stays and age
print('Filtering admissions on number of ICU stays and age...')
to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
to_keep = to_keep.loc[to_keep.ICUSTAY_ID == 1][['HADM_ID']]
stays = stays.merge(to_keep, left_on='HADM_ID', right_on='HADM_ID')

stays['AGE'] = (
    stays['INTIME'].values.astype('datetime64[D]') -
    stays['DOB'].values.astype('datetime64[D]')).astype(float) / 365
stays.loc[stays.AGE < 0, 'AGE'] = 90
stays = stays.loc[(stays['AGE'] >= 18) & (stays['AGE'] <= np.inf)]

print(
    len(stays['ICUSTAY_ID'].unique()),
    len(stays['HADM_ID'].unique()),
    len(stays['SUBJECT_ID'].unique()))


# Add mortality info
print('Adding mortality info...')
mortality = (
    stays['DOD'].notnull() &
    (stays['INTIME'] <= stays['DOD']) &
    (stays['OUTTIME'] >= stays['DOD']))
mortality = (
    mortality |
    (stays.DEATHTIME.notnull()) &
    (stays['INTIME'] <= stays['DEATHTIME']) &
    (stays.OUTTIME >= stays.DEATHTIME))
stays['MORTALITY_INUNIT'] = mortality.astype(int)

mortality = (stays['DOD'].notnull()) & \
            (stays['ADMITTIME'] <= stays['DOD']) & \
            (stays['DISCHTIME'] >= stays['DOD'])
mortality = (mortality) | \
            (stays['DEATHTIME'].notnull()) & \
            (stays['ADMITTIME'] <= stays['DEATHTIME']) & \
            (stays['DISCHTIME'] >= stays['DEATHTIME'])
stays['MORTALITY'] = mortality.astype(int)


# Break up stays by subject
print('Breaking up stays by subject...')
subjects = stays['SUBJECT_ID'].unique()
for subject_id in tqdm(subjects):
    dn = os.path.join(args.output_path, str(subject_id))
    if not os.path.exists(dn):
        os.makedirs(dn)
    stays.loc[stays['SUBJECT_ID'] == subject_id].sort_values(
        by='INTIME').to_csv(os.path.join(dn, 'stays.csv'), index=False)


# Read events table and break up by subject
print('Reading events table and breaking up by subject...')
subjects = set([str(s) for s in subjects])
nb_rows = {
    'CHARTEVENTS':  330712484,
    'LABEVENTS':    27854056,
    'OUTPUTEVENTS': 4349219
}

for table in ['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS']:
    tn = os.path.join(args.mimic3_path, table+'.csv')
    obs_header = [
        'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE',
        'VALUEUOM']

    curr_obs = []
    curr_subject_id = ''

    for row in tqdm(csv.DictReader(open(tn, 'r')), total=nb_rows[table]):
        if row['SUBJECT_ID'] not in subjects:
            continue
        row_out = {
            'SUBJECT_ID': row['SUBJECT_ID'],
            'HADM_ID':    row['HADM_ID'],
            'ICUSTAY_ID': row['ICUSTAY_ID'] if 'ICUSTAY_ID' in row else '',
            'CHARTTIME':  row['CHARTTIME'],
            'ITEMID':     row['ITEMID'],
            'VALUE':      row['VALUE'],
            'VALUEUOM':   row['VALUEUOM']
        }
        if curr_subject_id != '' and curr_subject_id != row['SUBJECT_ID']:
            dn = os.path.join(args.output_path, str(curr_subject_id))
            if not os.path.exists(dn):
                os.makedirs(dn)
            fn = os.path.join(dn, 'events.csv')
            if not os.path.exists(fn) or not os.path.isfile(fn):
                f = open(fn, 'w')
                f.write(','.join(obs_header) + '\n')
                f.close()
            w = csv.DictWriter(
                open(fn, 'a'),
                fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
            w.writerows(curr_obs)
            curr_obs = []
        curr_obs.append(row_out)
        curr_subject_id = row['SUBJECT_ID']

    if curr_subject_id != '' and curr_subject_id != row['SUBJECT_ID']:
        dn = os.path.join(args.output_path, str(curr_subject_id))
        if not os.path.exists(dn):
            os.makedirs(dn)
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(
            open(fn, 'a'),
            fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(curr_obs)
        curr_obs = []
