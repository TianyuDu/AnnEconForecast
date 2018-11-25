"""
This script contains various types of loss metric report function.
"""

import numpy as np
import pandas as pd
from typing import Dict


def score(
    self,
    actual: pd.DataFrame,
    pred: pd.DataFrame
) -> Dict[str, float]:
    acceptable_types = (
        float, int, np.float32, np.float64, np.int32, np.int64)

    def check_type(df): return all(
        type(x) in acceptable_types for x in df.values.reshape(-1))

    def get_types(df): return set([type(x) for x in df.values.reshape(-1)])

    assert check_type(actual),\
        f"Invalid types found in actual series,\nTypes found {get_types(actual)}"
    assert check_type(pred),\
        f"Invalid types found in prediction series,\nTypes found {get_types(pred)}"

    metric_dict = dict()

    act_val = actual.values.reshape(-1,)
    pred_val = pred.values.reshape(-1,)

    MAE = lambda x, y: np.sum(np.abs(x - y))
    MSE = lambda x, y: np.sum((x - y)**2)
    RMSE = lambda x, y: np.sqrt(MSE(x, y))
    MAPE = lambda x, y: np.sum(np.abs((x - y) / y))

    # mean absolute error
    metric_dict["mae"] = MAE(pred_val, act_val)
    # mean squared error
    metric_dict["mse"] = MSE(pred_val, act_val)
    # root mean squared error
    metric_dict["rmse"] = RMSE(pred_val, act_val)
    # mean absolute percentage error
    metric_dict["mape"] = MAPE(pred_val, act_val)

    return metric_dict

