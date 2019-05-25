from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
plt.style.use("seaborn-dark")

import SlpGenerator
import Logger
import ManyToOneLSTM
# import ptc.data_proc as data_proc
# from ptc.model import *


CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"

PROFILE = {
    "train_size": 231,  # Include both training and validation sets.
    "test_size": 58,
    "lags": 8,
    "vr": 0.2,  # Validation ratio.
    "neurons": (64, 128),
    "epochs": 1000
}

# if __name__ == '__main__':
globals().update(PROFILE)
# load data and make training set
# data = torch.load('traindata.pt')
df = pd.read_csv(
    SUNSPOT_DATA,
    index_col=0,
    date_parser=lambda x: datetime.strptime(x, "%Y")
)

# TODO: preprocessing date, and write reconstruction script.

df_train, df_test = df[:train_size], df[-test_size:]

gen = SlpGenerator.SlpGenerator(df_train)
fea, tar = gen.get_many_to_one(lag=lags)
train_dl, val_dl, train_ds, val_ds = gen.get_tensors(
    mode="Nto1", lag=lags, shuffle=True, batch_size=32, validation_ratio=vr
)
# build the model
net = ManyToOneLSTM.ManyToOneLSTM(
    neurons=neurons
)
net.double()  # Cast all floating point parameters and buffers to double datatype
criterion = torch.nn.MSELoss()
# use LBFGS as optimizer since we can load the whole data to train
optimizer = torch.optim.Adam(net.parameters(), lr=0.03)

train_log = Logger.TrainLogger()
val_log = Logger.TrainLogger()
# begin to train
with tqdm.trange(epochs) as prg:
    for i in prg:
        train_loss = []
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
        train_log.add(i, np.mean(train_loss))
        if i % 10 == 0:
            val_loss = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_dl):
                    data, target = map(torch.Tensor, (data, target))
                    data, target = data.double(), target.double()
                    out = net(data)
                    loss = criterion(out, target)
                    val_loss.append(loss.data.item())
            val_log.add(i, np.mean(val_loss))
        prg.set_description(
            f"Epoch [{i}/{epochs}]: TrainLoss={np.mean(train_loss): 0.3f}, ValLoss={np.mean(val_loss): 0.3f}")
        # print(f"Epoch: {i}\tTotal Loss: {train_loss:0.6f}\tLatest Val Loss: {val_loss:0.6f}")


# # begin to predict, no need to track gradient here
# with torch.no_grad():
#     future = 1
#     pred_train = seq(train_ds.tensors[0], future=future)
#     loss = criterion(pred_train[:, :-future], train_ds.tensors[1])
#     print("train loss", loss.item())

# with torch.no_grad():
#     future = 1
#     pred_test = seq(val_ds.tensors[0], future=future)
#     loss = criterion(pred_test[:, :-future], val_ds.tensors[1])
#     print('test loss:', loss.item())
#     # y = pred.detach().numpy()  # Fetch the result.
