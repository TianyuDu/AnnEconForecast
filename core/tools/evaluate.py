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
from typing import Dict, Union, Tuple
import statsmodels

import sys
sys.path.extend(["../"])

# ======== Start Test Code ========
pprint(DATA_DIR)

# Pre-processing Parameters
PERIODS = 1
ORDER = 1
LAGS = 12
df = load_dataset(DATA_DIR["0"])
train, test = df[:int(0.8*len(df))], df[int(0.8*len(df)):]

# ======== End Test Code ========


def run_persistence_model(
    test_series: pd.DataFrame
) -> Dict[str, float]:
    model = PersistenceModel()
    pred = model.predict(df)
    metrics = merged_scores(actual=df, pred=pred)
    print(f"Persistence prediction on {len(test_series)} observations.")
    for m, v in zip(metrics.keys(), metrics.values()):
        print(f"\t{m}={v}")
    return metrics


def run_arima(
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    order: Tuple[int]
) -> Dict[str, float]:
    print(f"Evaluating ARIMA performance on time series.\
    \nMode: Simple Forecasting\
    \nConfig:\
    \n\tOrder(p,d,q)={order}\
    \n\tTraining set size: {len(train_series)}\
    \n\tTesting set (to be forecasted) size: {len(test_series)}\
    ")
    model = statsmodels.tsa.arima_model.ARIMA(test_series, order=order)
    model_fit = model.fit(disp=1)
    pred = model_fit.forecast(steps=len(train_series))[0]
    pred = pd.DataFrame(pred)

    print(f"ARIMA{order} prediction on {len(test_series)} observations.")
    metrics = merged_scores(actual=test_series, pred=pred)
    return metrics


def run_arima_rolling_forecast(
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    order: Tuple[int],
    verbose: bool=False
) -> Dict[str, float]:
    print(f"Evaluating ARIMA performance on time series.\
    \nMode: Rolling Forecasting\
    \nConfig:\
    \n\tOrder(p,d,q)={order}\
    \n\tTraining set size: {len(train_series)}\
    \n\tTesting set (to be forecasted) size: {len(test_series)}\
    ")

    def f(x): return x.values.reshape(-1,)
    train, test = f(train_series), f(test_series)
    history = [x for x in train]
    pred = list()
    for t in range(len(test_series)):
        model = statsmodel.tsa.arima_model.ARIMA(
            history, order=order
        )
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        pred.append(yhat)
        obs = test[t]
        hist.append(obs)
        if verbose:
            print(f"Test step [{t}]: Predicted={yhat}, expected={obs}")
    error = sklearn.metrics.mean_squared_error(test, pred)
    if verbose:
        print("Test MSE: {error}")
    metrics = merged_scores(
        actual=test_series,
        pred=pd.DataFrame(pred),
        verbose=True)
    return metrics


# ==== Test Code ====
pred = run_arima(
    train,
    test,
    (14, 1, 1)
)

run_persistence_model(test)
