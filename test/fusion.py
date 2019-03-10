"""
Test codes for ARIMA and RNN fusion model.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use("seaborn-dark")


def paser(x):
    return datetime.strptime(x, "%Y")

df = pd.read_csv("./data/sunspots.csv", index_col=0, date_parser=paser, verbose=True)
df.plot()
plt.show()

if __name__ == "__main__":
    pass
