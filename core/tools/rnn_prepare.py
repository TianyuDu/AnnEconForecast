"""
This file contains data preparation methods for RNN.
"""
import sys
import os.path
from pprint import pprint
from typing import Dict, Union, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import *
from core.models.baseline_rnn import *
from core.models.stat_models import *
import core.tools.data_import as data_import
from core.tools.metrics import *
import core.tools.time_series as time_series

sys.path.extend(["../"])


def prepare_dataset(
    file_dir: str,
    periods: int=1,
    order: int=1,
    verbose: bool=True
) -> pd.DataFrame:
    """
    Prepare the dataset for RNN Training.

    Args:
        file_dir:
            A string locating the directory of *.csv file to be loaded.
        periods:
            An integer representing the number of periods of looking back
            when taking the difference on raw dataset.
        order:
            An integer representing the (recursive) times of differencing is
            computed on the raw dataset.
            * Analogue to the "d" parameter in ARIMA(p,d,q)

        ** In most cases, use periods=1 and order=1 should be sufficient.

        verbose:
            A bool indicating whether report should be printed during the process.

    Returns:
        A dataframe of prepared dataset.
    """
    # ======== Args Check ========
    assert os.path.exists(file_dir), f"File {file_dir} cannot be found."

    assert isinstance(periods, int), "Periods arg should be an integer."
    assert periods >= 1, "Periods arg should be at least 1."

    assert isinstance(order, int), "Order arg should be an integer."
    assert order >= 1, "Order arg should be at least 1."

    assert isinstance(verbose, bool), "Verbose arg should be a bool."

    # ======== Core ========
    if verbose:
        print(f"Retrieving raw data from {file_dir}...")

    df = data_import.load_dataset(file_dir)
    if verbose:
        print(f"Processing data, taking (periods, order)=({periods}, {order})...")
    prepared_df = time_series.differencing(
        df,
        periods=periods,
        order=order
    )
    prepared_df.head()
    if verbose:
        print("Dropping Nan observations...")
    prepared_df.dropna(inplace=True)

    if verbose:
        print("First few rows of dataset loaded:")
        print(prepared_df.head())
    return prepared_df


# Normalize the sequence
def normalize(
    df: pd.DataFrame,
    train_ratio: float,
    lags: int
) -> Tuple[np.ndarray]:
    scaler = StandardScaler().fit(
        df[:int(train_ratio*len(df))].values)
    df.iloc[:, 0] = scaler.transform(df.values)

    X_raw, y_raw = gen_supervised_sequence(
        df, lags, df.columns[0], sequential_label=False)

    (X_train, X_test, y_train, y_test) = train_test_split(
        X_raw, y_raw,
        test_size=1 - train_ratio,
        shuffle=False)

    (X_train, X_val, y_train, y_val) = train_test_split(
        X_train, y_train,
        test_size=0.1,
        shuffle=False
    )

    def trans(x): return x.reshape(-1, 1)

    y_train = trans(y_train)
    y_test = trans(y_test)
    y_val = trans(y_val)

    print(
        f"Training and testing set generated,\
        \nX_train shape: {X_train.shape}\
        \ny_train shape: {y_train.shape}\
        \nX_test shape: {X_test.shape}\
        \ny_test shape: {y_test.shape}\
        \nX_validation shape: {X_val.shape}\
        \ny_validation shape: {y_val.shape}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
