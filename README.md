Time-Sensitive Deep Learning for ICU Outcome Prediction Without Variable Selection or Cleaning
=========

Python suite to reproduce the results presented in **Time-Sensitive Deep Learning for ICU Outcome Prediction Without Variable Selection or Cleaning** from the MIMIC-III clinical database.

## News

* **2019 June 14**: This work was accepted as a conference abstract and chosen for a talk the 32nd Annual congress of the European Society of Intensive Care Medicine in Berlin ([ESICM LIVES 2019](https://www.esicm.org/events/32nd-annual-congress-berlin/)).

## Citation
If you use this code or these benchmarks in your research, please cite the following publication: *Jacob Deasy, Pietro Liò, and Ari Ercole. Time-Sensitive Deep Learning for ICU Outcome Prediction Without Variable Selection or Cleaning. **ARXIV CODE TBC*** which is now available at **LINK TBC**.

## TLDR
Simply run:

        ./bin/preprocessing.sh
        ./bin/train_20bins_48hr.sh

## Motivation

**TBC**

## Structure
The content of this repository can be divided into four big parts:
* Processing tokenized datasets.  
* Model training and evaluation scripts.
* Plotting scripts

The `mimic3benchmark/scripts` directory contains the scripts for converting the raw MIMIC-III csv files to our flexible data structure).
All evaluation scripts are stored in the `mimic3benchmark/evaluation` directory.
The `mimic3models` directory contains the baselines models along with some helper tools.
Those tools include discretizers, normalizers and functions for computing metrics.

## Dataset generation
----
Here are the required steps to replicate our results:
1. Clone the repo:

        git clone https://github.com/jacobdeasy/flexEHR/
        cd flexEHR/
        pip install -r requirements.txt

2. Separate patients as described in the [MIMIC-III benchmark](https://arxiv.org/abs/1703.07771):

        python -m flexehr.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
        python -m flexehr.scripts.validate_events data/root/

3. Extract individual stays from each patient. but **DO NOT** select or clean variables.
        a. **No variable selection.**
        b. **No variable cleaning.**

        python -m flexehr.scripts.extract_episodes_from_subjects data/root/

4. Generate a dictionary of labels to value arrays from `ITEMID`s and values

        python -m flexehr.scripts.gen_value_dict data/root/

5. Quantize events into `n` bins based on dictionary labels to values.

        python -m flexehr.scripts.quantize_events data/root/ data/root_n

6. Split train and test set

        python -m flexehr.benchmark.split_train_and_test data/root_n

7. Truncate timeseries to `T` hours (and match patient inclusion in benchmark dataset)

        python -m flexehr.scripts.truncate_timeseries data/root_n data/root_n_T

8. Create final arrays for training

        python -m flexehr.scripts.create_array data/root_n_T

## Model training

sh scripts to train particular models...

## Plotting

### AUROC plot

### AUROC over time plot
