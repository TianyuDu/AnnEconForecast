import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

# import Logger
import LSTM
import SlpGenerator

plt.style.use("seaborn-dark")

# import ptc.data_proc as data_proc
# from ptc.model import *


CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"

PROFILE = {
    "TRAIN_SIZE": 231,  # Include both training and validation sets.
    "TEST_SIZE": 58,
    "LAGS": 12,
    "VAL_RATIO": 0.2,  # Validation ratio.
    "NEURONS": (32, 64),
    "EPOCHS": 100,
    "LOG_NAME": "null"
}

# if __name__ == '__main__':
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

# TODO: preprocessing date, and write reconstruction script.

df_train, df_test = df[:TRAIN_SIZE], df[-TEST_SIZE:]

gen = SlpGenerator.SlpGenerator(df_train, verbose=False)
fea, tar = gen.get_many_to_one(lag=LAGS)
train_dl, val_dl, train_ds, val_ds = gen.get_tensors(
    mode="Nto1", lag=LAGS, shuffle=True, batch_size=32, validation_ratio=VAL_RATIO
)
# build the model
net = LSTM.PoolingLSTM(
    lags=LAGS,
    neurons=NEURONS
)

net.double()  # Cast all floating point parameters and buffers to double datatype
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.03)

with tqdm.trange(EPOCHS) as prg, SummaryWriter(comment=LOG_NAME) as writer:
    for i in prg:
        train_loss = []
        # TODO: rename all data to feature
        for batch_idx, (data, target) in enumerate(train_dl):
            # data, target = Variable(data), Variable(target)
            data, target = map(torch.Tensor, (data, target))
            data, target = data.double(), target.double()
            optimizer.zero_grad()
            out = net(data)
            loss = criterion(out, target)
            train_loss.append(loss.data.item())
            loss.backward()
            optimizer.step()
        # # train_log.add(i, np.mean(train_loss))
        # MSE Loss
        writer.add_scalars(
            "loss/mse", {"Train": np.mean(train_loss)}, i)
        # RMSE Loss
        func = lambda x: np.sqrt(np.mean(x))
        writer.add_scalars(
            "loss/rmse", {"Train": func(train_loss)}, i)

        if i % 10 == 0:
            val_loss = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_dl):
                    data, target = map(torch.Tensor, (data, target))
                    data, target = data.double(), target.double()
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
        # print(f"Epoch: {i}\tTotal Loss: {train_loss:0.6f}\tLatest Val Loss: {val_loss:0.6f}")
    # TODO: deal with the add graph function here.
    # writer.add_graph(net, (torch.zeros(32, LAGS)))
    with open(writer.logdir + "/profile.json", "a") as f:
        encoded = json.dumps(PROFILE)
        f.write(encoded)

# begin to predict, no need to track gradient here
# with torch.no_grad():
#     pred_train = net(train_ds.tensors[0], future=True)
#     loss = criterion(pred_train[:, :-future], train_ds.tensors[1])
#     print("train loss", loss.item())

# with torch.no_grad():
#     future = 1
#     pred_test = seq(val_ds.tensors[0], future=future)
#     loss = criterion(pred_test[:, :-future], val_ds.tensors[1])
#     print('test loss:', loss.item())
#     # y = pred.detach().numpy()  # Fetch the result.
