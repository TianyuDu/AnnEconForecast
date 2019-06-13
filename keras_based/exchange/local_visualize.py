"""
Multivariate Version of prediciton.
"""
import sys
import datetime

import keras
import pandas as pd
import numpy as np
import matplotlib
# TODO: for mac OS: os.name == "posix" and sys.platform == "darwin"
# Use this identifier to automatically decide the following.
on_server = bool(
    int(input("Training on server wihtout graphic output? [0/1] >>> ")))
if on_server:
    matplotlib.use(
        "agg",
        warn=False,
        force=True
    )
from matplotlib import pyplot as plt
import sklearn

from typing import Union, List

import config
import containers
import methods
from containers import *
from methods import *
from models import *
from multi_config import *

file_dir = "./data/exchange_rates/exchange_rates_Daily.csv"

container = MultivariateContainer(
    file_dir,
    "DEXCAUS",
    load_multi_ex,
    CON_config)

# Create empty model


# Testing Data

yhat = model.predict(model.container.test_X)
yhat = model.container.invert_difference(
    yhat, range(5130-len(yhat), 5130), fillnone=True
)

train_yhat = model.predict(model.container.train_X)
train_yhat = model.container.invert_difference(
    train_yhat, range(len(train_yhat)), fillnone=True
)

plt.close()
plt.plot(yhat, linewidth=0.6, alpha=0.6, label="Test set yhat")
plt.plot(train_yhat, linewidth=0.6, alpha=0.6, label="Train set yhat")
plt.plot(model.container.ground_truth_y,
         linewidth=1.2, alpha=0.3, label="actual")
plt.legend()
plt.show()

# Training Data

yhat = model.predict(model.container.train_X)
acty = model.container.scaler_y.inverse_transform(model.container.train_y)
yhat = model.container.invert_difference(yhat, range(4617), fillnone=False)

plt.close()
plt.plot(yhat, linewidth=0.6, alpha=0.6, label="yhat")
# plt.plot(acty, linewidth=0.6, alpha=0.6, label="actual")
plt.plot(model.container.ground_truth_y,
         linewidth=0.6, alpha=0.6, label="actual")
plt.legend()

time_stamp = str(datetime.datetime.now())

if not on_server:
    if bool(int(input("Show plot result? [0/1] >>> "))):
        plt.show()
    else:
        plt.savefig(f"./figure/{time_stamp}_train.svg")
else:
    plt.savefig(f"./figure/{time_stamp}_train.svg")


def visualize_raw(data: pd.DataFrame, action: Union["save", "show"]) -> None:
    plt.close()
    plt.figure()
    values = data.values
    num_series = values.shape[1]
    wid = int(np.ceil(np.sqrt(num_series)))
    for i in range(num_series):
        plt.subplot(wid, wid, i+1)
        name = data.columns[i]
        plt.plot(values[:, i], alpha=0.6, linewidth=0.6)
        plt.title(name, y=0.5, loc="right")
    if action == "show":
        plt.show()
    elif action == "save":
        plt.savefig("raw.svg")
