"""
The default hyper-parameter searching program.
This is a control script.
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
from typing import Dict, List
from datetime import datetime

import sys
sys.path.append("../")
sys.path.append(
    "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast")
core_dir = input("Directory of core files >>> ")
if not core_dir.endswith("/"):
    core_dir += "/"
sys.path.append(core_dir)

import constants
from core.tools.metrics import *
import core.tools.visualize as visualize
from core.tools.time_series import *
from core.tools.data_import import *
import core.tools.rnn_prepare as rnn_prepare

import core.models.stacked_lstm as stacked_lstm

import core.training.hps_methods as hps_methods


# data preparation phase.
pprint(constants.DATA_DIR)
choice = None
while choice is None or choice not in constants.DATA_DIR.keys():
    if choice is not None:
        print("Invalid data location received, try again...")
    choice = input("Select Dataset >>> ")
FILE_DIR = constants.DATA_DIR[choice]

print(f"Dataset chosen: \n\t{FILE_DIR}")

config_name = input("Name of configuration file to load >>> ")

exec(f"import core.training.configs.{config_name} as config")

for att in dir(config):
    if att.endswith("_config"):
        print(f"Loading: {att}")
        exec(f"globals().update(config.{att})")


parameter_collection = hps_methods.gen_hparam_set(config.train_param)


def individual_train(para) -> None:
    prepared_df = rnn_prepare.prepare_dataset(
        file_dir=FILE_DIR,
        periods=PERIODS,
        order=ORDER,
        remove=None,
        verbose=False
    )
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = rnn_prepare.generate_splited_dataset(
        raw=prepared_df,
        train_ratio=0.8,
        val_ratio=0.1,
        lags=para["num_time_steps"]
    )
    data_collection = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    def checkpoints(z): return [
        z*x for x in range(1, para["epochs"] // z)] + [-1]
    
    (metrics_dict, predictions) = stacked_lstm.exec_core(
        parameters=para,
        data_collection=data_collection,
        prediction_checkpoints=checkpoints(
            para["epochs"] // 10
        )
    )
    plt.close()
    fig = visualize.plot_checkpoints(predictions, y_test, "test")
    plt.savefig(para["fig_path"]+"pred_records.svg")


for (i, para) in enumerate(parameter_collection):
    print("================================")
    print(f"Executing [{i}/{len(parameter_collection) - 1}] hyper-parameter searching session...")
    start = datetime.now()
    individual_train(para)
    print(f"Time taken for session [{i}]: {str(datetime.now() - start)}.")

