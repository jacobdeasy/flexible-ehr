Time-Sensitive Deep Learning for ICU Outcome Prediction Without Variable Selection or Cleaning
=========

Python suite to reproduce the results presented in **Time-Sensitive Deep Learning for ICU Outcome Prediction Without Variable Selection or Cleaning** from the MIMIC-III clinical database.

## News

* **2019 June 14:** This work was accepted as a conference abstract and chosen for a talk the 32nd Annual congress of the European Society of Intensive Care Medicine in Berlin ([ESICM LIVES 2019](https://www.esicm.org/events/32nd-annual-congress-berlin/)).
* **2019 October 01:** This work was present at the ESICM conference in Berlin as part of the track *From bytes to bedside: Improving intensive care with Data*.

## Citation
If you use this code or the paper in your research, please cite the following publication: *Jacob Deasy, Pietro Liò, and Ari Ercole. Time-Sensitive Deep Learning for ICU Outcome Prediction Without Variable Selection or Cleaning. arXiv preprint arXiv:1909.08981*

## TLDR
Clone the repo and simply run the following (NB. this may take several hours and requires 30GB storage on top of the space required for the MIMIC III csv files):

        ./bin/collect_tokens.sh  {PATH TO MIMIC-III CSVs}
        ./bin/train_48_20.sh
        ./bin/plot_48_20.sh

## Plots

**TBC**

## Motivation

**TBC**

## Structure
The content of this repository can be divided into:
* Processing and tokenizing MIMIC III.  
* Model training and evaluation scripts.
* Plotting.

The `flexehr/scripts` directory contains the scripts for converting the raw MIMIC-III csv files to tokenized arrays in-line with ourdata structure.
Training and evaluation can be ran using `main.py` which relies on the scripts within the `flexehr` directory.

Running 

## Dataset generation
Here are the required steps to replicate our results:
1. Clone the repo:

        git clone https://github.com/jacobdeasy/flexEHR/
        cd flexEHR/
        pip install -r requirements.txt

2. Separate patients as described in the [MIMIC-III benchmark](https://arxiv.org/abs/1703.07771):

        python -m flexehr.scripts.1_subject_events {PATH TO MIMIC-III CSVs} data/root/
        python -m flexehr.scripts.2_validate_events data/root/

3. Extract individual stays from each patient. Patient data is simply separated by admission information.

    a. **No variable selection.**

    b. **No variable cleaning.**

        python -m flexehr.scripts.3_subject2episode data/root/

4. Truncate timeseries to `t` hours and store in `data/root_t`

        python -m flexehr.scripts.4_truncate_timeseries data/root -t 48

5. Generate a dictionary of labels to value arrays from `ITEMID`s and values

        python -m flexehr.scripts.6_generate_value_dict data/root

6. Quantize events into `n` bins based on dictionary labels to values.

        python -m flexehr.scripts.7_quantize_events data/root -t 48 -n 20

7. Create final arrays for training

        python -m flexehr.scripts.create_arrays data -t 48 -n 20

## Model training
To start we split the train and test set with a fixed seed for reproducibility:

        python -m flexehr.scripts.split_train_test data -t 48 -n 20

Our model is now ready to train.

        python main.py
