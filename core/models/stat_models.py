"""
Basic statistical models 
"""
import numpy as np
import pandas as pd
from typing import Dict

class persistence_model:
    def __init__(
        self
    ) -> None:
        self.acceptable_types = (
            float, int, np.float32, np.float64, np.int32, np.int64)
    
    def predict(
        self,
        series: pd.DataFrame
    )-> pd.DataFrame:
        pred = series.shift(1)
        new_column_names = [col+"_pred" for col in pred.columns]
        pred.columns = new_column_names
        pred.fillna(value=0.0, inpalce=True)
        return pred
    
    def score(
        self,
        actual: pd.DataFrame,
        pred: pd.DataFrame
    ) -> Dict[str, float]:
        check_type = lambda df: all(type(x) in self.acceptable_types for x in df.values.reshape(-1))
        get_types = lambda df: set([type(x) for x in df.values.reshape(-1)])

        assert check_type(actual),\
        f"Invalid types found in actual series,\nTypes found {get_types(actual)}"
        assert check_type(pred),\
        f"Invalid types found in prediction series,\nTypes found {get_types(pred)}"
        # TODO: stop here.


