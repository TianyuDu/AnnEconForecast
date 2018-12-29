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
    verbose: bool=True,
    remove: object=None
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
        remove:
            Any observation with value equals the "remove" object will be dropped.
    Returns:
        A dataframe of prepared dataset.
    """
    # ======== Args Check ========
    assert os.path.exists(file_dir), f"File {file_dir} cannot be found."

    assert type(periods) in [int, np.int_], f"Periods arg should be an integer, received: {periods}"
    assert periods >= 1, f"Periods arg should be at least 1, received: {periods}"

    assert type(order) in [int, np.int_], f"Order arg should be an integer, received: {order}"
    assert order >= 1, f"Order arg should be at least 1, received: {order}"

    assert isinstance(verbose, bool), f"Verbose arg should be a bool, received: {verbose}"

    # ======== Core ========
    if verbose:
        print(f"Retrieving raw data from {file_dir}...")

    df = data_import.load_dataset(
        file_dir,
        remove=remove
    )
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


def normalize(
    raw: pd.DataFrame,
    train_ratio: float
) -> pd.DataFrame:
    """
    Normalize the dataset based on a training subset and apply the scaler to the whole set.
    Args:
        raw:
            A dataframe of raw data.
        train_ratio:
            The ratio of all observations that should be considered as the training set.
            The scaler will be fit on the training subset only.
    Returns:
        A normalized dataframe with the same shape as raw dataframe.
    """
    # ======== Args Check ========
    assert isinstance(raw, pd.DataFrame), f"Raw dataset should be a pandas DataFrame, received type: {type(raw)}"
    assert type(train_ratio) in [float, np.float_], f"Training set ratio should be a float, received: {train_ratio}"
    assert 0 < train_ratio <= 1, f"Training set ratio should be positive and at most 1, received: {train_ratio}"
    # ======== Core ========
    df = raw.copy()
    scaler = StandardScaler().fit(
        df[:int(train_ratio*len(df))].values
    )
    print(f"StandardScaler applied, scaling based on the first {int(train_ratio*len(df))} observations.")
    df.iloc[:, 0] = scaler.transform(df.values)
    return df


def split_dataset(
    raw: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    lags: int
) -> Tuple[np.ndarray]:
    """
    Generate and split the prepared dataset for RNN training into training, testing and validation sets.
    Args:
        raw:
            A dataframe containing the prepared dataset returned from prepare_dataset method.
        train_ratio:
            A float representing the ratio of the whole dataset to be used as training set.
        val_ratio:
            A float denoting the ratio of the whole dataset to be used as validation set.
        ** Note the sum of train_ratio and val_ratio should between 0 and 1.
        lags:
            An integer indicating 
    Returns:
        return a 6-tuple.
        Univariate case format:
            X: (num_samples, lags, 1)
            y: (num_samples, 1)
        Tuple format:
            (X_train, X_val, X_test, y_train, y_test, y_val)
    """
    # ======== Args Check ========
    assert isinstance(raw, pd.DataFrame), "Raw dataset should be a pandas dataframe."
    assert type(train_ratio) in [float, np.float_]\
    and 0 < train_ratio <= 1,\
    f"train_ratio should be a float within range (0,1], received: {train_ratio}"

    assert type(val_ratio) in [float, np.flaot_]\
    and 0 < val_ratio <= 1,\
    f"val_ratio should be a float within range (0,1], received: {val_ratio}"

    assert type(lags) in [int, np.int_]\
    and lags >= 1,\
    f"lags should be an integer at least 1, received: {lags}"

    # ======== Core ========
    test_ratio = 1 - train_ratio - val_ratio
    df = normalize(
        raw,
        train_ratio
    )

    X_raw, y_raw = gen_supervised_sequence(
        df, lags, df.columns[0], sequential_label=False)

    (X_train, X_test, y_train, y_test) = train_test_split(
        X_raw, y_raw,
        test_size=test_ratio,
        shuffle=False)

    (X_train, X_val, y_train, y_val) = train_test_split(
        X_train, y_train,
        test_size=val_ratio / (val_ratio + train_ratio),
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
