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

import sys
sys.path.extend(["../"])


pprint(DATA_DIR)

# Pre-processing Parameters
PERIODS = 1
ORDER = 1
LAGS = 12

df = load_dataset(DATA_DIR["0"])

pmodel