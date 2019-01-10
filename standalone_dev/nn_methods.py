import numpy as np
import pandas as pd
from typing import Tuple, List


# Instance data type, a single training example.
Instance = Tuple[np.ndarray, np.ndarray, pd.Timestamp]


def gen_slp_seq(
    df: pd.DataFrame,
    num_time_steps: int,
    label_col: str = None
) -> List[Instance]:
    """
    Generate the supervised learning problem with
    sequence-valued label in each instance.
    Sliding Window Method
    data.shape = (num_obs, 1)
    """
    X_set = df.copy()
    if label_col is None:
        # The next-observed values of ALL features are interpreted as label.
        y_set = df.copy()
    else:
        y_set = df[label_col].copy()

    instances = list()
    for t in range(len(X_set)):
        try:
            feature = X_set.iloc[t: t+num_time_steps, :]
            label = y_set.iloc[t+1: t+num_time_steps+1, :]
            assert len(feature) == len(label)
            instances.append((
                feature.values,
                label.values,
                label.index[-1]
            ))
        except AssertionError:
            print(f"Failed time step ignored: {t}")

    return instances


def gen_slp_pt(df, num_time_steps):
    """
    Generate the supervised learning problem with
    point-valued label in each instance.
    """
    slp_sequential = gen_slp_seq(
        df,
        num_time_steps=num_time_steps
    )
    instances = [
        (x, y[-1], t)
        for x, y, t in slp_sequential
    ]
    return instances


def gen_slp(
    df,
    num_time_steps,
    sequential: bool,
    label_col: str = None
    ):
    if sequential:
        return gen_slp_seq(df, num_time_steps)
    else:
        return gen_slp_pt(df, num_time_steps)

























