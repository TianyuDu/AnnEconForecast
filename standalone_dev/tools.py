import numpy as np
import pandas as pd
from typing import List, Tuple


def progbar(curr, total, full_progbar):
    """
    Progress bar used in training process.
    Reference: https://geekyisawesome.blogspot.com/2016/07/python-console-progress-bar-using-b-and.html
    """
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
#     print('\r', '#'*filled_progbar + '-'*(
#         full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    print('\r', '#'*filled_progbar + '-'*(
        full_progbar-filled_progbar), f"[{curr}/{total}, {frac:>7.2%}]", end='')



def inv_diff(
    diff: pd.DataFrame,
    src: pd.DataFrame,
    order: int=1
) -> pd.DataFrame:
    col_name = src.columns[0]
    inv = pd.DataFrame([np.nan] * len(src))  # Inverted series
    inv.columns = src.columns
    inv.index = src.index

    for (idx, date) in enumerate(diff.index):
        # For each date in the differenced series.
        if idx - order < 0:
            inv[col_name][date] = np.nan
            continue
        
        base_date = diff.index[idx - order]
        base_val = src[col_name][base_date]
        inv[col_name][date] = base_val + diff[col_name][date]

    return inv


Instance = Tuple[np.ndarray, np.ndarray, pd.Timestamp]
def gen_slp_sequential(
    df: pd.DataFrame,
    num_time_steps: int,
    label_col: str = None
) -> List[Instance]:
    """
    GENerate Supervised Learning Problem with SEQUENTIAL label.
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


def format_instances(
    instances: List[Instance]
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Return shape:
    X: (num_instances, num_time_steps, num_inputs)
    y: (num_instances, num_time_steps, num_outputs)
    time_index: len=num_instances, list.
    """
    num_instances = len(instances)
    typical_X, typical_y, t = instances[0]
    num_time_steps, num_inputs = typical_X.shape
    _, num_outputs =  typical_y.shape

    print(f"num_inputs={num_inputs}, num_outputs={num_outputs}, num_time_steps={num_time_steps}")

    X_lst = [z[0] for z in instances]
    y_lst = [z[1] for z in instances]
    ts = [z[2] for z in instances]
    X = np.squeeze(X_lst)
    y = np.squeeze(y_lst)

    X = X.reshape(num_instances, num_time_steps, num_inputs)
    y = y.reshape(num_instances, num_time_steps, num_outputs)

    return X, y, ts















