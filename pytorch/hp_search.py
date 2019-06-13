"""
TODO: write documents
"""
from pprint import pprint
from datetime import datetime
import sys
sys.path.extend(["./core", "./core/tools"])

import matplotlib
c = input("Use Agg as matplotlib (avoid tkinter)[y/n]: ")
if c.lower() == "y":
    matplotlib.use("agg")
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import tqdm
import torch

import lstm_controls
from param_set_generator import gen_hparam_set
import DIRS

# For sunspot, train_size = 231, test_size = 58

SRC_PROFILE = {
    "TRAIN_SIZE": 0.8,  # Include both training and validation sets.
    "TEST_SIZE": 0.2,
    "LAGS": [9, 12],
    "VAL_RATIO": 0.2,  # Validation ratio.
    "BATCH_SIZE": 32,
    "LEARNING_RATE": [0.03, 0.01, 0.3],
    "NEURONS": (256, 512),
    "EPOCHS": [300, 500],
    "NAME": "_"
}

SRC_PROFILE = {
    "TRAIN_SIZE": 0.8,  # Include both training and validation sets.
    "TEST_SIZE": 0.2,
    "LAGS": 36,
    "VAL_RATIO": 0.2,  # Validation ratio.
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 0.1,
    "NEURONS": (256, 512),
    "EPOCHS": [300],
    "NAME": "_"
}

def df_loader() -> pd.DataFrame:
    df = pd.read_csv(
        "/home/ec2-user/AnnEconForecast/data/CPIAUCSL.csv",
        index_col=0,
        date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
        engine="c"
    )
    # df[df[df.columns[0]] == "."] = np.nan
    # df.fillna(method="ffill")
    # df = df[df != "."]
    # df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    profile_set = gen_hparam_set(SRC_PROFILE)
    print("Cuda avaiable: ", torch.cuda.is_available())
    start = datetime.now()
    raw_df = df_loader()
    for i in tqdm.trange(len(profile_set), desc="Hyper-Param Profile"):
        PROFILE = profile_set[i]
        lstm_controls.core(
            **PROFILE, 
            profile_record=PROFILE,
            raw_df=raw_df,
            verbose=True
        )
    print(f"\nTotal time taken: {datetime.now() - start}")
