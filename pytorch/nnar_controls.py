"""
Main model for the fully-connected ANN
NNAR, neural network auto-regression.
"""
import json
from datetime import datetime
from typing import Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import torch
import tqdm
from tensorboardX import SummaryWriter

import FcModel
import LogUtility
import SlpGenerator

plt.style.use("seaborn-dark")


# Settings
CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"
SUNSPOT_DATA_EC2 = "/home/ec2-user/environment/AnnEconForecast/data/sunspots.csv"

# Let's call hyper-parameters profile.
PROFILE = {
    "TRAIN_SIZE": 0.8,  # Include both training and validation sets.
    "TEST_SIZE": 0.2,
    "LAGS": 6,
    "VAL_RATIO": 0.2,  # Validation ratio.
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.05,
    "NEURONS": (64, 128),
    "EPOCHS": 100,
    "NAME": "example"  # Name for tensorboard logs.
}


def core(
    profile_record: dict,
    raw_df: pd.DataFrame,
    verbose: bool,
        # set verbose=False when running hyper-parameter search.
    # To ensure progress bar work correctly
    # ==== Parameter from profile ====
    TRAIN_SIZE: Union[int, float],
    TEST_SIZE: Union[int, float],
    LAGS: int,
    VAL_RATIO: float,
    BATCH_SIZE: int,
    LEARNING_RATE: float,
    NEURONS: Tuple[int],
    EPOCHS: int,
    NAME: str    
    ) -> None:
    if verbose:
        try:
            input_name = input("Log name ([Enter] for default name): ")
            assert input_name != ""
            LOG_NAME = input_name
        except AssertionError:
            print(f"Default name: {LOG_NAME} is used.")


    # ======== Prepare the Dataset ========
    df = raw_df.copy()

    if TRAIN_SIZE < 1 and TEST_SIZE:
        assert TRAIN_SIZE + TEST_SIZE == 1
        TRAIN_SIZE = int(len(df) * TRAIN_SIZE)
        TEST_SIZE = len(df) - TRAIN_SIZE
    
    df_train, df_test = df[:TRAIN_SIZE], df[-TEST_SIZE:]
    
    if verbose:
        print(f"Training set: {TRAIN_SIZE}; test set: {TEST_SIZE}")

    gen = SlpGenerator.SlpGenerator(df_train, verbose=verbose)
    fea, tar = gen.get_many_to_one(lag=LAGS)
    train_dl, val_dl, train_ds, val_ds = gen.get_tensors(
        mode="Nto1", lag=LAGS, shuffle=True, batch_size=BATCH_SIZE, validation_ratio=VAL_RATIO
    )

    if verbose:
        print(f"Training Set @ {len(train_ds)}\nValidation Set: @ {len(val_ds)}")

    # ==== Move everything to GPU (if avaiable) ====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if verbose:
        print("Device selected: ", device)
    
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # ======== Build the Model ========
    net = FcModel.Net(num_fea=LAGS, num_tar=1, neurons=NEURONS)
    net.float()  # Cast all floating point parameters and buffers to double datatype
    net = net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    with tqdm.trange(EPOCHS) as prg, SummaryWriter(comment=NAME) as writer:
        for i in prg:
            # ======== Training Phase ========
            if verbose:
                print(f"======== Training Phase @ Epoch {i} ========")
            train_loss = []
            for batch_idx, (data, target) in enumerate(train_dl):
                # Move 
                data = data.to(device).float()
                target = target.to(device).float()
                
                optimizer.zero_grad()
                out = net(data)
                loss = criterion(out, target)
                train_loss.append(loss.data.item())
                loss.backward()
                optimizer.step()
                # raise Exception("Debug")
            # MSE Loss
            writer.add_scalars("loss/mse", {"Train": np.mean(train_loss)}, i)
            # RMSE Loss
            _rmse = lambda x: np.sqrt(np.mean(x))
            writer.add_scalars("loss/rmse", {"Train": _rmse(train_loss)}, i)
            
            # ======== Validation Phase ========
            if verbose:
                print(f"======== Validation Phase @ Epoch {i} ========")
            if i % 10 == 0:
                val_loss = []
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_dl):
                        data = data.to(device).float()
                        target = target.to(device).float()
                        out = net(data)
                        loss = criterion(out, target)
                        val_loss.append(loss.data.item())
                writer.add_scalars(
                    "loss/mse", {"Validation": np.mean(val_loss)}, i)
                writer.add_scalars(
                    "loss/rmse", {"Validation": _rmse(val_loss)}, i)
            prg.set_description(
                f"TrainMSE:{np.mean(train_loss): 0.7f}, ValMSE:{np.mean(val_loss): 0.7f}")
        
        # Save the training profile.
        with open("./" + writer.logdir + "/profile.json", "a") as f:
            encoded = json.dumps(profile_record)
            f.write(encoded)

        # Begin to predict on test set, no need to track gradient here.
        # ======== Testing Phase ========
        if verbose:
            print("======== Testing Phase ========")
        with torch.no_grad():
            gen_test = SlpGenerator.SlpGenerator(df_test, verbose=False)
            fea_df, tar_df = gen_test.get_many_to_one(lag=LAGS)
            assert len(fea_df) == len(tar_df)

            pred = list()
            for i in range(len(fea_df)):
                f = lambda x, idx: torch.Tensor(x.iloc[idx].values)
                feature = f(fea_df, i)
                feature = feature.view(1, feature.shape[0])
                # TODO: check this.
                # feature = feature.double()
                # feature = feature.to(device).float()
                feature = feature.to(device)
                pred.append(net(feature))
            
            pred_df = pd.DataFrame(
                data=np.array(pred),
                index=tar_df.index
            )
        # ==== visualize test set prediction ====
            rcParams["lines.linewidth"] = 0.2
            total = pd.concat([tar_df, pred_df], axis=1)
            total.columns = ["Actual", "Forecast"]
            mse = np.mean((total["Actual"] - total["Forecast"])**2)
            fig = plt.figure(dpi=700)

            plt.plot(total)
            plt.grid()
            plt.title(f"Test Set of {NAME} After {EPOCHS} Epochs")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend(["Actual", f"Forecast, MSE={mse}"])
            writer.add_figure(
                "Test set predictions", fig, global_step=EPOCHS
            )
