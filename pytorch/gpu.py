"""
Jun. 7 2019
This is the script for training with GPUs on server.
"""
from pprint import pprint
from datetime import datetime
import sys
sys.path.extend(["./core", "./core/tools"])

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

import tqdm
import torch

import main_lstm
from param_set_generator import gen_hparam_set

CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA_E = "/home/ec2-user/environment/AnnEconForecast/data/sunspots.csv"
SUNSPOT_DATA_EG = "/home/ec2-user/AnnEconForecast/data/sunspots.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"

SRC_PROFILE = {
    "TRAIN_SIZE": 231,  # Include both training and validation sets.
    "TEST_SIZE": 58,
    "LAGS": 12,
    "VAL_RATIO": 0.2,  # Validation ratio.
    "LEARNING_RATE": 0.03,
    "NEURONS": (512, 1024),
    "EPOCHS": 100,
    "LOG_NAME": "lastout",
    "TASK_NAME": "LastOutLSTM on Sunspot",
    "DATA_DIR": SUNSPOT_DATA_EG
}

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == "__main__":
    print("CUDA Device Avaiable: ", torch.cuda.is_available())
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    start = datetime.now()
    # PROFILE = SRC_PROFILE
    main_lstm.core(**SRC_PROFILE, profile_record=SRC_PROFILE, verbose=False)
    print(f"\nTotal time taken: {datetime.now() - start}")
