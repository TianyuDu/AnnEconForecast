"""
Basic statistical models 
"""
import numpy as np
import pandas as pd
from typing import Dict
import sklearn

class persistence_model:
    def __init__(
        self
    ) -> None:
        pass

    def predict(
        self,
        series: pd.DataFrame
    )-> pd.DataFrame:
        pred = series.shift(1)
        new_column_names = [col+"_pred" for col in pred.columns]
        pred.columns = new_column_names
        pred.fillna(value=0.0, inplace=True)
        return pred

