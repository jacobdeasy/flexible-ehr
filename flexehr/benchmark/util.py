"""
Helper function for converting csv to pandas.DataFrame.

Adapted from https://github.com/YerevaNN/mimic3-benchmarks

References
----------
Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and
Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series
Data. arXiv:1703.07771
"""


from __future__ import absolute_import
from __future__ import print_function

import pandas as pd


def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col)
