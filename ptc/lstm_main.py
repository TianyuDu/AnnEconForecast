from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
plt.style.use("seaborn-dark")

# import ptc.data_proc as data_proc
import data_proc
# from ptc.model import *
from model import *


CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"

# if __name__ == '__main__':
EPOCHS = 1000
# load data and make training set
# data = torch.load('traindata.pt')
df = pd.read_csv(
    SUNSPOT_DATA,
    index_col=0,
    date_parser=lambda x: datetime.strptime(x, "%Y")
)

# TODO: preprocessing date, and write reconstruction script.

train_dl, val_dl, train_ds, val_ds = data_proc.gen_data_tensors(
    df,
    lag=8,
    batch_size=32,
    validation_ratio=0.2
)

# NOTE: many to many forecasting

# build the model
try:
    del(seq)
except NameError:
    pass 
seq = SingleLayerLSTM(neurons=[64])
seq.double()  # Cast all floating point parameters and buffers to double datatype
criterion = torch.nn.MSELoss()
# use LBFGS as optimizer since we can load the whole data to train
optimizer = torch.optim.Adam(seq.parameters(), lr=0.1)
# begin to train
with tqdm.trange(EPOCHS) as prg:
    for i in prg:
        optimizer.zero_grad()  # Reset gradients in optimizer record, otherwsie grad will be accumualted.
        out = seq(train_ds.tensors[0])  # Equivalently, seq.forward(inputs)
        loss = criterion(out, train_ds.tensors[1])

        prg.set_description(f"Epoch [{i:0.3f}], loss: {loss.item()}")

        loss.backward()
        optimizer.step()

# begin to predict, no need to track gradient here
with torch.no_grad():
    future = 1
    pred_train = seq(train_ds.tensors[0], future=future)
    loss = criterion(pred_train[:, :-future], train_ds.tensors[1])
    print("train loss", loss.item())

with torch.no_grad():
    future = 1
    pred_test = seq(val_ds.tensors[0], future=future)
    loss = criterion(pred_test[:, :-future], val_ds.tensors[1])
    print('test loss:', loss.item())
    # y = pred.detach().numpy()  # Fetch the result.

# Actual forecast is the first element of every array.
# extract = lambda x: x.detach().numpy()[..., 0]
# forecast = np.concatenate(
#     [extract(pred_train), extract(pred_test)])
# actual = np.concatenate(
#     [extract(train_ds.tensors[1]), extract(val_ds.tensors[1])])
# plt.plot(forecast, label="Predicted")
# plt.plot(actual, label="Actual")
# plt.title("One step forecast")
# plt.legend()
# plt.show()

# # draw the result
# plt.figure(figsize=(30, 10))
# plt.title(
#     'Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
# plt.xlabel('x', fontsize=20)
# plt.ylabel('y', fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)


# def draw(yi, color):
#     plt.plot(np.arange(inputs.size(1)),
#             yi[:inputs.size(1)], color, linewidth=2.0)
#     plt.plot(np.arange(inputs.size(1), inputs.size(1) + future),
#             yi[inputs.size(1):], color + ':', linewidth=2.0)
#     # plt.show()
# draw(y[0], 'r')
# draw(y[1], 'g')
# draw(y[2], 'b')
# # plt.savefig('predict%d.pdf' % i)
# plt.close()
# plt.show()
