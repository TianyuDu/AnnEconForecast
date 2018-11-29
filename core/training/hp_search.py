"""
The default hyper-parameter searching program.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from typing import Dict

import sys
sys.path.append(".../")

import constants
from core.tools.metrics import *
from core.tools.visualize import *
from core.tools.time_series import *
from core.tools.data_import import *
import core.tools.rnn_prepare as rnn_prepare

import core.models.stacked_lstm as stacked_lstm


# data preparation phase.
pprint(constants.DATA_DIR)
choice = None
while choice is None or choice not in constants.DATA_DIR.keys():
    if choice is not None:
        print("Invalid data location received, try again...")
    choice = input("Select Dataset >>> ")
FILE_DIR = constants.DATA_DIR[choice]

print(f"Dataset chosen: \n{FILE_DIR}")


# Configuration:
default_data_para = {
    PERIODS: 1,
    ORDER: 1,
    LAGS: 12,
    TRAIN_RATIO: 0.8
}

if input("Specify data processing parameters? [y/n]>>>").upper() == "Y":
    PERIODS = int(input("Periods[int] >>>"))
    ORDER = int(input("Order[int] >>>"))
    LAGS = int(input("Lags[int] >>>"))
    TRAIN_RATIO = float(input("Ratio of training set[float] >>>"))
else:
    globals().update(default_data_para)

EXPERIMENT_NAME = input()
