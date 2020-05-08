"""Check the maximum length of stay of donors in MIMIC-III.
If it is <48 it does not affect our study."""

import os
import pandas as pd

df = pd.read_csv('ADMISSIONS.csv')
df = df.loc[(df['DIAGNOSIS'].str.lower() == 'organ donor') | (df['DIAGNOSIS'].str.lower() == 'organ donor account')]

files = os.listdir('root')
ods = list(df['SUBJECT_ID'])

los_list = []

for od in ods:
    try:
        df_tmp = pd.read_csv(os.path.join('root', str(od), 'stays.csv'))
        los_list += list(df_tmp['LOS'].values)
    except:
        pass

print(max(los_list))

"""
Result: 37.2832
"""
