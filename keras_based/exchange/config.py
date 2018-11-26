import pandas as pd
import numpy as np

"""
Configuration file for univariate time series model.
"""
# Configuration for neural network
neural_network_config = {
    "batch_size": 1,  # Batch size for training
    "epoch": 10,  # Training epochs
    "neuron": [128],  # Unit of neurons in layers in LSTM model.
}

# Configuration for data processing
# Configuration dictionary to create DataContainer Object
data_proc_config = {
    # Method to remove non-stationarity of time series
    # By default, create stationary time series using first
    # differencing.
    "method": "diff",  # {str}
    # Avaiable method:
    # "diff": remove non-stationarity by differncing.
    "diff.lag": 1,  # {int > 0}
    # Lag of differencing,
    # Notice: If set to zero, the 0 series will be returned (x[i] - x[i - 0] == 0)
    "diff.order": 1,  # {int >= 0}
    # Order of differencing,
    # Notice: 0 order differencing will return the original series.
    # For k order differncing, the generated series will have length N - k
    "test_ratio": 0.2,  # {0 <= float < 1}
    # Ratio of test set to evaluate neural network.
    "lag_for_sup": 3,  # {int > 0}
    # Total number of lag values used to generate
    # supervised learning problem (SLP).
    # Using m then the i-1 to i-m total m lagged variables
    # will be used to predict the i-index variable in the SLP.
}
