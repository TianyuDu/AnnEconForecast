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
    pass


def gen_supervised(
    src_df: pd.DataFrame,
    predictors: List[int]
) -> Tuple[pd.DataFrame]:
    """
    predictors format: (order, period)
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


def clean_nan(
    X: pd.DataFrame,
    y: pd.DataFrame
) -> Tuple[pd.DataFrame]:
    # target_col = [y.columns] if y.columns is not list else y.columns
    target_col = y.columns[0]
    aggreagate = pd.concat([X, y], axis=1)
    ori_len = len(aggreagate)

    aggreagate.dropna(inplace=True)
    new_len = len(aggreagate)
    print(f"{ori_len - new_len} ({(ori_len - new_len) / ori_len * 100:0.2f}%) observations with Nan are dropped.")
    
    return (
        aggreagate.drop(columns=target_col),
        aggreagate[target_col].to_frame()
    )

