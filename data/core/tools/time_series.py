"""
This package contains methods to manipulate time series 
object.
"""

import numpy as np
import pandas as pd


def differencing(
    src_df = pd.DataFrame,
    order: int=1,
    periods: int=1
) -> pd.DataFrame:
    df = src_df.copy()
    
    for od in range(order):
        df = df.diff(periods=periods)

    new_cols = [col+f"_d{periods}_x{order}" for col in df.columns]
    df.columns = new_cols
    return df


def invert_diff(
    src_df = pd.DataFrame,
    order: int=1,
    periods: int=1
) -> pd.DataFrame:
    pass
