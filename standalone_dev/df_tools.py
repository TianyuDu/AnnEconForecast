import numpy as np
import pandas as pd
import os
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

# Instance data type, a single training example.
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

    print(f"num_instances={num_instances}, num_inputs={num_inputs}, num_outputs={num_outputs}, num_time_steps={num_time_steps}")

    X_lst = [z[0] for z in instances]
    y_lst = [z[1] for z in instances]
    ts = [z[2] for z in instances]
    X = np.squeeze(X_lst)
    y = np.squeeze(y_lst)

    X = X.reshape(num_instances, num_time_steps, num_inputs)
    y = y.reshape(num_instances, num_time_steps, num_outputs)

    return X, y, ts


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    ts: List[pd.Timestamp],
    ratio={"train": 0.8, "val": 0.1, "test": 0.1},
    shuffle=False
):
    if shuffle:
        raise NotImplementedError()
    train_end = int(ratio["train"] * X.shape[0])
    val_end = train_end + int(ratio["val"] * X.shape[0])
    
    X_train = X[:train_end, :, :]
    y_train = y[:train_end, :, :]
    ts_train = ts[:train_end]
    
    X_val = X[train_end: val_end, :, :]
    y_val = y[train_end: val_end, :, :]
    ts_val = ts[train_end: val_end]
    
    X_test = X[val_end:, :, :]
    y_test = y[val_end:, :, :]
    ts_test = ts[val_end:]
    
    return (X_train, y_train, ts_train,
            X_val, y_val, ts_val,
            X_test, y_test, ts_test
           )


def mkdirs(base: str) -> dict:
    path_col = {
        "BASE_PATH": base,
        "TB_PATH": base + "/tensorboard",
        "FIG_PATH": base + "/figures",
        "NUM_PATH": base + "/numericals"
    }
    for p in path_col.values():
        print(f"Creating directory: {p}")
        os.mkdir(p)
    return path_col









