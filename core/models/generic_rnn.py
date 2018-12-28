"""
This file contains the generic recurrent neural network type.
"""
import sys
from pprint import pprint
from typing import Dict, Iterable, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import *
from core.models.baseline_rnn import *
from core.models.stat_models import *
from core.tools.data_import import *
import core.tools.metrics as metrics
from core.tools.rnn_prepare import *
from core.tools.time_series import *
from core.tools.visualize import *


def GenericRNN:
    """
    The generic recurrent network.
    """
    def __init__(
        self
        param: Dict[str, object],
        prediction_checkpoints: Iterable[int]=[-1],
        verbose: bool=True
    ) -> Nones:
        pass
