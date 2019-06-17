"""
TODO: write documents
"""
from pprint import pprint
from datetime import datetime
import sys
sys.path.extend(["./core", "./core/tools"])

import matplotlib
c = input("Use Agg as matplotlib (avoid tkinter)[y/n]: ")
if c.lower() in ["y", ""]:
    matplotlib.use("agg")
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import tqdm
import torch

import lstm_controls
import nnar_controls
from param_set_generator import gen_hparam_set
import DIRS
from ProfileLoader import ProfileLoader

# For sunspot, train_size = 231, test_size = 58

LSTM_PROFILE = {
    "TRAIN_SIZE": 0.8,  # Include both training and validation sets.
    "TEST_SIZE": 0.2,
    "LAGS": [20, 32],
    "VAL_RATIO": 0.2,  # Validation ratio.
    "BATCH_SIZE": [32, 128],
    "LEARNING_RATE": [0.003, 0.001],
    "NEURONS": [(2048, 1024), (1024, 2048), (2048, 4096)],
    "EPOCHS": [100, 300],
    "NAME": "high_complex_lstm"
}

ANN_PROFILE = {
    "TRAIN_SIZE": 0.8,  # Include both training and validation sets.
    "TEST_SIZE": 0.2,
    "LAGS": [6, 12, 20, 32],
    "VAL_RATIO": 0.2,  # Validation ratio.
    "BATCH_SIZE": 32,
    "LEARNING_RATE": [0.03, 0.01, 0.003, 0.001],
    "NEURONS": [(64, 128), (128, 256), (256, 512)],
    "EPOCHS": [500, 1000],
    "NAME": "NNAR"
}

SRC_PROFILE = ANN_PROFILE

def df_loader() -> pd.DataFrame:
    df = pd.read_csv(
        "/home/ec2-user/AnnEconForecast/data/CPIAUCSL_monthly_change.csv",
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
    if input("Create a new experiment?[y/n] ").lower() in ["y", ""]:
        profile_set = gen_hparam_set(SRC_PROFILE)
    else:
        path = input("Directory of profiles: ")
        loader = ProfileLoader(path)
        profile_set = loader.get_all()
    print("Cuda avaiable: ", torch.cuda.is_available())
    
    # Expand lstm profiles
    # profile_set_expand = list()
    # for p in profile_set:
    #     if "lstm" in p["NAME"]:
    #         for i in [True, False]:
    #             p_expand = p.copy()
    #             p_expand["POOLING"] = i
    #             profile_set_expand.append(p_expand)
    #     elif "nnar" in p["NAME"]:
    #         profile_set_expand.append(p)
    #     else:
    #         raise Exception
    
    # Reset validation ratio to 0
    profile_set_expand = list()
    for p in profile_set:
        p_modified = p.copy()
        p_modified["VAL_RATIO"] = 0.05
        profile_set_expand.append(p_modified)
    
    start = datetime.now()
    raw_df = df_loader()
    with tqdm.trange(len(profile_set_expand)) as prg:
        for i in prg:
            PROFILE = profile_set_expand[i]
            prg.set_description(
                f"n={PROFILE['NEURONS']};l={PROFILE['LAGS']};a={PROFILE['LEARNING_RATE']};Total")
            if "lstm" in PROFILE["NAME"]:
                lstm_controls.core(
                    **PROFILE, 
                    profile_record=PROFILE,
                    raw_df=raw_df,
                    verbose=False
                )
            elif "nnar" in PROFILE["NAME"]:
                nnar_controls.core(
                    **PROFILE, 
                    profile_record=PROFILE,
                    raw_df=raw_df,
                    verbose=False
                )
    print(f"\nTotal time taken: {datetime.now() - start}")
