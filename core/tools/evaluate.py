"""
Containing statistical models used as baseline models
to evaluate the relative performance of neural networks
in time series prediction.
Created: Nov. 25 2018
"""
from constants import *
from core.tools.metrics import *
from core.models.stat_models import *
from core.models.baseline_rnn import *
from core.tools.visualize import *
from core.tools.time_series import *
from core.tools.data_import import *
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from typing import Dict, Union

import sys
sys.path.extend(["../"])

# ======== Start Test Code ========
pprint(DATA_DIR)

# Pre-processing Parameters
PERIODS = 1
ORDER = 1
LAGS = 12
df = load_dataset(DATA_DIR["0"])

# ======== End Test Code ========

def run_persistence_model(
    test_series: pd.DataFrame
) -> Dict[str, float]:
    model = PersistenceModel()
    pred = model.predict(df)
    metrics = merged_score(actual=df, pred=pred)
    print(f"Persistence prediction on {len(test_series)} observations.")
    for m, v in zip(metrics.keys(), metrics.values()):
        print(f"\t{m}={v}")
    return metrics

