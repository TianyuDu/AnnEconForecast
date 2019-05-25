"""
Main model for the fully-connected ANN
NNAR, neural network auto-regression.
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark")

import tqdm
import torch
from tensorboardX import SummaryWriter

import FcModel
import SlpGenerator
import LogUtility

# Settings 
CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"

# Let's call hyper-parameters profile.
PROFILE = {
    "MODEL": "NNAR",
    "TRAIN_SIZE": 231, # Include both training and validation sets.
    "TEST_SIZE": 58,
    "LAGS": 6,
    "VAL_RATIO": 0.2, # Validation ratio.
    "NEURONS": (64, 128),
    "EPOCHS": 100,
    "LOG_NAME": "null" # Name for tensorboard logs.
}

if __name__ == "__main__":
    globals().update(PROFILE)
    try:
        input_name = input("Log name ([Enter] for default name): ")
        assert input_name != ""
        LOG_NAME = input_name
    except AssertionError:
        print(f"Default name: {LOG_NAME} is used.")
    
    df = pd.read_csv(
        SUNSPOT_DATA,
        index_col=0,
        date_parser=lambda x: datetime.strptime(x, "%Y")
    )
    df_train, df_test = df[:TRAIN_SIZE], df[-TEST_SIZE:]
    print(f"Train&validation size: {len(df_train)}, test size: {len(df_test)}")

    gen = SlpGenerator.SlpGenerator(df_train, verbose=False)
    fea, tar = gen.get_many_to_one(lag=LAGS)
    train_dl, val_dl, train_ds, val_ds = gen.get_tensors(
        mode="Nto1", lag=LAGS, shuffle=True, batch_size=32, validation_ratio=VAL_RATIO
    )

    net = FcModel.Net(num_fea=LAGS, num_tar=1, neurons=NEURONS)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    # train_log = LogUtility.Logger()
    # val_log = LogUtility.Logger()

    with tqdm.trange(EPOCHS) as prg, SummaryWriter(comment=LOG_NAME) as writer:
        for i in prg:
            train_loss = []
            for batch_idx, (data, target) in enumerate(train_dl):
                data, target = map(torch.Tensor, (data, target))
                optimizer.zero_grad()
                out = net(data)
                loss = criterion(out, target)
                train_loss.append(loss.data.item())
                loss.backward()
                optimizer.step()
            # train_log.add(i, np.mean(train_loss))
            # Write MSE
            writer.add_scalars(
                "loss/mse", {"Train": np.mean(train_loss)}, i)
            # Write RMSE
            func = lambda x: np.sqrt(np.mean(x))
            writer.add_scalars(
                "loss/rmse", {"Train": func(train_loss)}, i)
            if i % 5 == 0:
                val_loss = []
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_dl):
                        data, target = map(torch.Tensor, (data, target))
                        out = net(data)
                        loss = criterion(out, target)
                        val_loss.append(loss.data.item())
                # val_log.add(i, np.mean(val_loss))
                writer.add_scalars(
                    "loss/mse", {"Validation": np.mean(val_loss)}, i)

                writer.add_scalars(
                    "loss/rmse", {"Validation": func(val_loss)}, i)
            prg.set_description(
                f"Epoch [{i+1}/{EPOCHS}]: TrainLoss={np.mean(train_loss): 0.3f}, ValLoss={np.mean(val_loss): 0.3f}")
        writer.add_graph(net, (torch.zeros(LAGS)))
        encoded = json.dumps(PROFILE)
        with open(writer.logdir + "/profile.json", "a") as f:
            f.write(encoded)

    # Create plot
    # plt.close()
    # plt.plot(train_log.get_df(ln=True), label="Train Loss")
    # plt.plot(val_log.get_df(ln=True), label="Validation Loss")
    # plt.legend()
    # plt.grid()
    # plt.show()
