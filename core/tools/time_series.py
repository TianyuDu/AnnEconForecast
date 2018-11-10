"""
This package contains methods to manipulate time series 
object.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


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


def gen_supervised(
    src_df: pd.DataFrame,
    predictors: List[int]
) -> Tuple[pd.DataFrame]:
    """
    predictors format: (order, period)
    generate supervised learning problem.
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


def gen_supervised_rnn(
    src_df: pd.DataFrame,
    lags: int,
) -> Tuple[np.ndarray]:
    """
    Args:
        src_df: source Series or DataFrame,
        lags: total number of lags used in one observation.
    Return 
    """
    assert lags > 0
    target = src_df.copy()
    X = target.shift(1)
    df = pd.concat([X, target], axis=1)
    df.dropna(inplace=True)

    samples = list()

    for t in range(len(df)-lags):
        obs = df[t:t+lags].values
        samples.append(obs)

    samples = np.array(samples)

    return samples[:,:,0], samples[:,:,1]


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

