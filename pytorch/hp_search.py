"""
TODO: write documents
"""
from pprint import pprint
from datetime import datetime
import sys
sys.path.extend(["./core", "./core/tools"])

import matplotlib
c = input("Use Agg as matplotlib (avoid tkinter)[y/n]:")
if c.lower() == "y":
    matplotlib.use("agg")
from matplotlib import pyplot as plt

import tqdm
import torch

import main_lstm
from param_set_generator import gen_hparam_set
import DIRS

# For sunspot, train_size = 231, test_size = 58

SRC_PROFILE = {
    "TRAIN_SIZE": 0.8,  # Include both training and validation sets.
    "TEST_SIZE": 0.2,
    "LAGS": [12, 36],
    "VAL_RATIO": 0.2,  # Validation ratio.
    "LEARNING_RATE": [0.03, 0.1, 0.3],
    "NEURONS": [(128, 126), (256, 512)],
    "EPOCHS": [300, 500],
    "LOG_NAME": "lastout",
    "TASK_NAME": "Exchange rate",
    "DATA_DIR": DIRS.DEXCAUS["ec2_gpu"]
}

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == "__main__":
    profile_set = gen_hparam_set(SRC_PROFILE)
    print("====Sample Configuration====")
    pprint(profile_set[0])
    print("============================")
    print("Cuda avaiable: ", torch.cuda.is_available())
    # TODO: manage cuda devices.
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    start = datetime.now()
    for i in tqdm.trange(len(profile_set), desc="Hyper-Param Profile"):
        PROFILE = profile_set[i]
        main_lstm.core(**PROFILE, profile_record=PROFILE, verbose=False)
    print(f"\nTotal time taken: {datetime.now() - start}")
    