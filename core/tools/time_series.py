"""
This package contains methods to manipulate time series 
object.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def differencing(
    src_df: pd.DataFrame,
    order: int=1,
    periods: int=1
) -> pd.DataFrame:
    df = src_df.copy()

    for od in range(order):
        df = df.diff(periods=periods)

    new_cols = [col+f"_od{periods}_pd{order}" for col in df.columns]
    df.columns = new_cols
    return df


def invert_diff(
    src_df: pd.DataFrame,
    order: int=1,
    periods: int=1
) -> pd.DataFrame:
    raise NotImplementedError


def gen_supervised_dnn(
    src_df: pd.DataFrame,
    predictors: List[int]
) -> Tuple[pd.DataFrame]:
    """
    predictors format: (order, period)
    generate supervised learning problem.
    # Customized predictors. (Non-consecutive)
    """
    df = src_df.copy()
    main_name = df.columns[0]
    df.columns = [f"{main_name}_target"]

    cols = list()
    for p in predictors:
        predictor = df.shift(periods=p)
        predictor.columns = [f"{main_name}_lag{p}"]
        cols.append(predictor)

    X = pd.concat(cols, axis=1)
    return X, df


def gen_supervised_sequence(
    src_df: pd.DataFrame,
    lags: int,
    target_column: str,
    sequence_label: bool=False,
) -> Tuple[np.ndarray]:
    """
    Args:
        src_df: source Series or DataFrame,
        lags: total number of lags used in one observation.
    Return 
    """
    assert lags > 0, "Lags(Periods of looking back) must be positive."
    X = src_df.copy()  # Predictor(s).

    # Copy the one-period-future value of target series and
    # use it as the label.
    target = X[target_column].shift(-1)

    # Append the label column to predictors.
    # So the last column of df is the future(shifted) target.
    df = pd.concat([X, target], axis=1)
    df.dropna(inplace=True)

    observations = [df[t:t+lags].values for t in range(len(df)-lags)]

    observations = np.array(observations)
    num_obs = len(observations)
    print(f"Total {num_obs} observations generated.")

    if sequence_label:
        X, y = (observations[:, :, :-1],
                observations[:, :, -1].reshape(num_obs, -1, 1))
    else:
        X, y = (observations[:, :, :-1],
                observations[:, -1, -1].reshape(num_obs, 1, 1))

    print("Note: shape format: (num_obs, time_steps, num_inputs/outputs)")
    print(f"X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def clean_nan(
    X: pd.DataFrame,
    y: pd.DataFrame
) -> Tuple[pd.DataFrame]:
    target_col = list(y.columns)
    aggreagate = pd.concat([X, y], axis=1)
    ori_len = len(aggreagate)

    aggreagate.dropna(inplace=True)
    new_len = len(aggreagate)
    print(f"{ori_len - new_len} ({(ori_len - new_len) / ori_len * 100:0.2f}%) observations with Nan are dropped.")

    return (
        aggreagate.drop(columns=target_col),
        aggreagate[target_col]
    )
