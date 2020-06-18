"""Generate summary statistics."""

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


VARIABLE_NAMES = [
    'Anion gap',
    'Albumin',
    # 'BANDS',
    'Bicarbonate',
    'Bilirubin',
    'Creatinine',
    'Chloride',
    'Diastolic BP',
    'GCS Total',
    'Glucose',
    'Heart rate',
    'Hematocrit',
    'Hemoglobin',
    'FiO2',
    'Lactate',
    'Mean ABP',
    'Platelet',
    'Potassium',
    'PTT',
    # 'INR (PT)',
    'PT',
    'Sodium',
    'Oxygen saturation',
    'Respiratory rate',
    'Systolic BP',
    'Temperature (F)',
    'BUN',
    'WBC'
]


UNITS = [
    'mEq/L',
    'g/L',
    # '',
    'mEq/L',
    'mg/dL',
    'mg/dL',
    'mEq/L',
    'mmHg',
    '',
    'mmol/L',
    '/minute',
    '\% of blood volume',
    'g/L',
    '\%',
    'mmol/L',
    'mmHg',
    '1000/mm$^{3}$',
    'mEq/L',
    'seconds',
    # '',
    'seconds',
    'mEq/L',
    '%',
    '/minute',
    'mmHg',
    '$^{\circ}$F',
    'mg/dL',
    '1000/mm$^{3}$'
]


ITEMIDS_LIST = [
    [50868],  # ANION GAP | CHEMISTRY | BLOOD | 769895
    [50862],  # ALBUMIN | CHEMISTRY | BLOOD | 146697
    # [51144],  # BANDS - hematology
    [50882],  # BICARBONATE | CHEMISTRY | BLOOD | 780733
    [50885],  # BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
    [50912],  # CREATININE | CHEMISTRY | BLOOD | 797476
    [
        50902,  # CHLORIDE | CHEMISTRY | BLOOD | 795568
        50806,  # CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
    ],
    [
        8368,   # Arterial BP [Diastolic] | 2088638
        8440,   # Manual BP [Diastolic] | 2152
        8441,   # NBP [Diastolic] | 1580652
        8555,   # Arterial BP #2 [Diastolic] | 18820
        220180, # Non Invasive Blood Pressure diastolic | Routine Vital Signs | 1289885
        220051  # Arterial Blood Pressure diastolic | Routine Vital Signs | 1149537
    ],
    [198],    # GCS TOTAL | 948148
    [
        50931,  # GLUCOSE | CHEMISTRY | BLOOD | 748981
        50809,  # GLUCOSE | BLOOD GAS | BLOOD | 196734
    ],
    [
        211,    # Heart Rate | 5190683
        220045, # Heart Rate | Routine Vital Signs | 2762225
    ],
    [
        51221,  # HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
        50810,  # HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
    ],
    [
        51222,  # HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
        50811,  # HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
    ],
    [223835], # Inspired O2 Fraction | Respiratory | 558061
    [50813],  # LACTATE | BLOOD GAS | BLOOD | 187124
    [
        456,    # NBP MEAN | 1560440
        52,     # Arterial BP Mean | 2075364
        6702,   # Arterial BP Mean #2 | 18946
        443,    # Manual BP Mean(calc) | 2346
        220052, # Arterial Blood Pressure mean | Routine Vital Signs | 1156173
        220181, # Non invasive Blood Pressure mean | Routine Vital Signs | 1292916
        225312  # ART BP mean | Routine Vital Signs | 87182
    ],
    [51265],  # PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
    [
        50971,  # POTASSIUM | CHEMISTRY | BLOOD | 845825
        50822,  # POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
    ],
    [51275],  # PTT | HEMATOLOGY | BLOOD | 474937
    # [51237],  # INR(PT) | HEMATOLOGY | BLOOD | 471183
    [51274],  # PT | HEMATOLOGY | BLOOD | 469090
    [
        50983,  # SODIUM | CHEMISTRY | BLOOD | 808489
        50824,  # SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
    ],
    [
        646,    # SpO2 | 3428011
        220277, # O2 saturation pulseoxymetry | Respiratory | 2671816
    ],
    [
        618,    # Respiratory Rate | 3395998
        615,    # Resp Rate (Total) | 417512
        220210, # Respiratory Rate | Respiratory | 2737105
        224690  # Respiratory Rate (Total) | Respiratory | 399752
    ],
    [
        51,     # Arterial BP [Systolic] | 2099353
        442,    # Manual BP [Systolic] | 2565
        455,    # NBP [Systolic] | 1586769
        6701,   # Arterial BP #2 [Systolic] | 19232
        220179, # Non Invasive Blood Pressure systolic | Routine Vital Signs | 1290488
        220050  # Arterial Blood Pressure systolic | Routine Vital Signs | 1149788
    ],
    [
        # 223762, # "Temperature Celsius" | Routine Vital Signs | 74144
        # 676,	# "Temperature C" | 379232
        223761, # "Temperature Fahrenheit" | Routine Vital Signs | 522143
        678     # "Temperature F" | 775928
    ],
    [51006],  # UREA NITROGEN | CHEMISTRY | BLOOD | 791925
    [
        51301,  # WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
        51300,  # WBC COUNT | HEMATOLOGY | BLOOD | 2371
    ]
]


def gen_stats(hours):
    patients = os.listdir('root')[:-1]

    total_event_count = 0
    variable_counts = np.zeros(len(ITEMIDS_LIST))
    arrs = [[] for _ in range(len(ITEMIDS_LIST))]

    for patient in tqdm(patients):
        df = pd.read_csv(os.path.join('root', patient, 'events.csv'))

        total_event_count += len(df)

        for i, itemids in enumerate(ITEMIDS_LIST):
            s = df.loc[df['ITEMID'].isin(itemids)]['VALUE']

            variable_counts[i] += len(s)

            s = pd.to_numeric(s, errors='coerce')  # convert to float and force errors to be nan
            s = s.loc[s.notnull()]  # now remove the nans

            arrs[i].extend(s.tolist())

    q1 = np.zeros(len(ITEMIDS_LIST))
    q2 = np.zeros(len(ITEMIDS_LIST))
    q3 = np.zeros(len(ITEMIDS_LIST))
    for i, itemids in enumerate(ITEMIDS_LIST):
        q1[i], q2[i], q3[i] = np.quantile(arrs[i], q=[0.25, 0.5, 0.75])

    # Save DataFrame and create LaTeX table
    df = pd.DataFrame(zip(VARIABLE_NAMES, variable_counts, q1, q2, q3, UNITS),
        columns=['Variable', 'n', 'q1', 'q2', 'q3', 'Unit of measurement'])
    df.to_csv('variable_stats.csv', index=None)
    df.to_latex(buf='table.txt', index=False, float_format='{:.2f}'.format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate summary statistics for patients.")
    parser.add_argument('-t', '--t-hours', type=int,
                        default=48,
                        help='Maximum number of hours to allow in timeseries.')
    args = parser.parse_args()

    gen_stats(hours=args.t_hours)
