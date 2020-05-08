
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
