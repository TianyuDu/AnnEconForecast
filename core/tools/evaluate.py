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


# ======== Start Test Code ========
pprint(DATA_DIR)

# Pre-processing Parameters
PERIODS = 1
ORDER = 1
LAGS = 12
df = load_dataset(DATA_DIR["0"])
train, test = df[:int(0.8*len(df))], df[int(0.8*len(df)):]
train_short, test_short = train[:200], test[:20]

# ======== End Test Code ========


def run_persistence_model(
    test_series: pd.DataFrame
) -> Dict[str, float]:
    model = PersistenceModel()
    pred = model.predict(df)
    print(f"Persistence prediction on {len(test_series)} observations.")
    metrics = merged_scores(actual=df, pred=pred, verbose=True)
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

    print(f"ARIMA{order} simple prediction on {len(test_series)} observations.")
    metrics = merged_scores(actual=test_series, pred=pred, verbose=True)
    return metrics


def run_arima_rolling_forecast(
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    order: Tuple[int],
    verbose: bool = False
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
        model = statsmodels.tsa.arima_model.ARIMA(
            history, order=order
        )
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        pred.append(yhat)
        obs = test[t]
        history.append(obs)
        if verbose:
            print(f"Test Step: [{t}/{len(test_series)}]: Predicted={yhat}, expected={obs}")
    
    error = sklearn.metrics.mean_squared_error(test, pred)
    if verbose:
        print(f"Test MSE: {error}")
    print(f"ARIMA{order} rolling prediction on {len(test_series)} observations.")

    pred_df = pd.DataFrame(pred)
    metrics = merged_scores(
        actual=test_series,
        pred=pred_df,
        verbose=True)
    return metrics


# ==== Test Code ====
arima = run_arima(
    train,
    test,
    (14, 1, 1)
)

persistence = run_persistence_model(test)

arima_rolling = run_arima_rolling_forecast(
    train_short,
    test_short,
    order=(6, 1, 1),
    verbose=True
)
