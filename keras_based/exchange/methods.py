"""
Methods for CPI prediction model.
"""
import datetime
import warnings

import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def load_dataset(dir: str) \
        -> pd.Series:
    """
        Read csv file, by default load exchange rate data
        CNY against USD (1 USD = X CNY)
    """
    series = pd.read_csv(
        dir,
        header=0,
        index_col=0,
        squeeze=True
    )
    # In Fred CSV file, nan data is represented by "."
    series = series.replace(".", np.nan)
    series = series.astype(np.float32)

    print(
        f"Found {np.sum(series.isna())} Nan data point(s), linear interpolation is applied.")
    series = series.interpolate(method="linear")
    print("Summary on Data:")
    print(series.describe())
    return series


def gen_sup_learning(data: np.ndarray, lag: int=1, nafill: object=0.0) \
        -> pd.DataFrame:
    """
        Generate superized learning problem.
        Transform the time series problem into a supervised learning
        with lag values as the training input and current value
        as target.
    """
    df = pd.DataFrame(data)
    # Each shifting creates a lag var: shift(n) to get lag n var.
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns = [df] + columns
    df = pd.concat(columns, axis=1)
    df.fillna(nafill, inplace=True)
    col_names = ["L0/current/target"] + [f"L{i}" for i in range(1, lag+1)]
    df.columns = col_names
    return df


def difference(dataset: np.ndarray, lag: int=1) \
        -> pd.Series:
    diff = list()
    for i in range(lag, len(dataset)):
        value = dataset[i] - dataset[i - lag]
        diff.append(value)
    return pd.Series(diff)


def inverse_difference(history, yhat, lag=1):
    return yhat + history[-lag]


def gen_scaler(train, test, tar_idx: int=0):
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_in = sklearn.preprocessing.StandardScaler()
    scaler_out = sklearn.preprocessing.StandardScaler()

    idx = list(range(train.shape[1]))
    idx.remove(tar_idx)

    train_X = train[:, idx]
    train_y = train[:, tar_idx].reshape(-1, 1)

    scaler_in = scaler_in.fit(train_X)
    scaler_out = scaler_out.fit(train_y)

    train_scaled_X = scaler_in.transform(train_X)
    train_scaled_y = scaler_out.transform(train_y)

    train_scaled = np.concatenate(
        [train_scaled_y, train_scaled_X],
        axis=1
    )

    test_X = test[:, idx]
    test_y = test[:, tar_idx].reshape(-1, 1)

    test_scaled_X = scaler_in.transform(test_X)
    test_scaled_y = scaler_out.transform(test_y)

    test_scaled = np.concatenate(
        [test_scaled_y, test_scaled_X],
        axis=1
    )

    return scaler_in, scaler_out, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    ar = np.array(new_row)
    ar = ar.reshape(1, len(ar))
    inverted = scaler.inverse_transform(ar)
    return inverted[0, -1]


def fit_lstm(train, batch_size, epoch, neurons) \
        -> keras.Sequential:
    """
    """
    # The first column is
    X, y = train[:, 1:], train[:, 1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=neurons[0],
        batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
        stateful=True,
        name="lstm_input"
    ))

    model.add(keras.layers.Dense(
        units=1,
        name="dense_output"
    ))

    model.compile(
        loss="mean_squared_error",
        optimizer="adam"
    )

    model.fit(
        X, y,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
        shuffle=False
    )
    return model


def forecast_lstm(
        model: keras.Sequential, batch_size: int, X: np.ndarray) -> float:
    # Single step prediction method.
    # Transform to array with shape (1 obs) * (1 batch) * (n features)
    X = X.reshape(1, 1, -1)
    yhat = model.predict(X, batch_size=batch_size, verbose=0)
    # return yhat[0, 0] <- original method, replace the current one if it does not work.
    return float(yhat)


def reshape_and_split(data: np.ndarray, tar_idx: int=0) \
        -> (np.ndarray, np.ndarray):
    """
    Reshaped dataset into shape (*, 1, *) to fit in the input
    layer of model.
    tar_idx is the index of column (one output sequence in this model)
    of target sequence.
    """
    obs, fea = data.shape
    reshaped = data.reshape(obs, 1, fea)
    idx = list(range(fea))
    idx.remove(tar_idx)

    res_X = reshaped[:, :, idx]
    res_y = reshaped[:, :, [tar_idx]]
    return res_X, res_y

def visualize3(
        raw_values: np.ndarray,
        train_pred: np.ndarray,
        test_pred: np.ndarray,
        dir: str=None) -> None:
    """
    If dir is set to None, graphic output will be shown in pyplot GUI. Otherwise graphic output will
    be saved to directory provided.
    """
    # Make sure dir string is a directory/folder.
    if dir is not None:
        assert dir.endswith("/"), "Target directory provided should be ended with /"
    # By default, training set comes before the test set and there's no shuffle.
    total_step, train_step, test_step = len(
        raw_values), len(train_pred), len(test_pred)

    print(f"Visualization info: \
    \n\ttotal step {total_step}, \
    \n\ttraining step: {train_step}, \
    \n\ttesting step: {test_step} \
    \n\toutput directory: {dir}")

    test_pred = [None] * train_step + list(np.squeeze(test_pred))
    test_pred = np.array(test_pred)

    plt.plot(np.squeeze(test_pred), alpha=0.6, linewidth=0.6)
    plt.plot(np.squeeze(train_pred), alpha=0.6, linewidth=0.6)
    plt.plot(np.squeeze(raw_values), alpha=0.6, linewidth=0.6)

    plt.legend(["Prediction on Testing Set", "Prediction on Training Set", "Actual Values"])

    now = str(datetime.datetime.now())
    if dir is not None:
        file_name = "Output" + now + ".svg"
        try:
            plt.savefig(dir + file_name)
            print(f"Visualization result is saved to {dir + file_name}")
        except FileNotFoundError:
            warnings.warn(
                f"Cannot save graphic output to {dir}, no action taken.",
                RuntimeWarning
            )
    else:
        plt.show()


def uni_visualize(
        actual_val: np.ndarray,
        pred_val: np.ndarray,
        break_point: int,
        dir: str=None) -> None:
    now = str(datetime.datetime.now())
    if dir is not None:
        assert type(dir) is str, "directory should be a string."
        assert dir.endswith(
            "/"), "Target directory provided should be ended with /"

    assert len(actual_val) == len(pred_val), \
        f"Actual values and model prediction values should have the same length, \
        got actual: {actual_val.shape} and pred: {pred_val.shape} instead."

    assert break_point in range(len(actual_val)), \
    f"Break point {break_point} beyond boundary (0, {len(actual_val)})"

    print(f""" Uni Visualization:
    Total observations: {len(actual_val)}
    With training-testing break point at {break_point}
    Save to directory: {dir}
    """)
    
    for val in [actual_val, pred_val]:
        plt.plot(val, alpha=0.6, linewidth=0.6)
    
    plt.axvline(x=break_point, alpha=0.3, linewidth=5.0)

    plt.legend([
        "Actual Observations",
        "Values Generated by Neural Network",
        "Train-test break point"
    ])

    # Save figure or show.
    now = str(datetime.datetime.now())
    if dir is not None:
        file_name = "Output" + now + ".svg"
        try:
            plt.savefig(dir + file_name)
            print(f"Visualization result is saved to {dir + file_name}")
        except FileNotFoundError:
            warnings.warn(
                f"Cannot save graphic output to {dir}, no action taken.",
                RuntimeWarning
            )
    else:
        plt.show()

def reshape_data(c) -> Tuple[np.ndarray]:
    train_X_reshaped = c.train_X_scaled.reshape(
        c.train_size,
        1,
        c.sup_fea
    )
    train_y_reshaped = c.train_y_scaled.reshape(
        c.train_size,
        1,
        c.sup_num_target
    )

    test_X_reshaped = c.test_X_scaled.reshape(
        c.test_size,
        1,
        c.sup_fea
    )
    test_y_reshaped = c.test_y_scaled.reshape(
        c.test_size,
        1,
        c.sup_num_target
    )
    return (
        train_X_reshaped,
        train_y_reshaped,
        test_X_reshaped,
        test_y_reshaped
    )
