import numpy as np
import pandas as pd

# ============ Configuration for multivariate model ============
file_dir = "./data/exchange_rates/exchange_rates_Daily.csv"
target = "DEXCAUS"

# Function to load data from spread sheet.


def load_multi_ex(file_dir: str) -> pd.DataFrame:
    dataset = pd.read_csv(file_dir, delimiter="\t", index_col=0)
    # Cleaning Data
    dataset.dropna(how="any", axis=0, inplace=True)
    dataset.replace(to_replace=".", value=np.NaN, inplace=True)
    dataset.fillna(method="ffill", inplace=True)
    dataset = dataset.astype(np.float32)
    # DEXVZUS behaved abnomally
    dataset.drop(columns=["DEXVZUS"], inplace=True)
    return dataset


CON_config = {
    "max_lag": 3,
    "train_ratio": 0.9,
    "time_steps": 14,
    "drop_target": False
}

NN_config = {
    "batch_size": 32,
    "validation_split": 0.1,
    "nn.lstm1": 256,
    "nn.lstm2": 128,
    "nn.dense1": 64
}
