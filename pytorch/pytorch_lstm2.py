from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt

import data
from model import *


if __name__ == '__main__':
    EPOCHS = 200
    # load data and make training set
    # data = torch.load('traindata.pt')
    df = pd.read_csv(
        "./sunspot.csv",
        index_col=0,
        date_parser=lambda x: datetime.strptime(x, "%Y")
    )
    X, Y = data.gen_sup(df, lag=24)
    num_train = int(0.8 * X.shape[0])
    inputs = torch.from_numpy(X.values[:num_train, ...])
    target = torch.from_numpy(Y.values[:num_train, ...])
    test_inputs = torch.from_numpy(X.values[num_train:, ...])
    test_targets = torch.from_numpy(Y.values[num_train:, ...])
    assert inputs.shape == target.shape
    assert test_inputs.shape == test_targets.shape
    # inputs = torch.from_numpy(data[3:, :-1])
    # target = torch.from_numpy(data[3:, 1:])  # One step shifting.
    # test_inputs = torch.from_numpy(data[:3, :-1])
    # test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    seq = Sequence()
    seq.double()  # Cast all floating point parameters and buffers to double datatype
    criterion = torch.nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = torch.optim.Adam(seq.parameters(), lr=0.03)
    # begin to train
    with tqdm.trange(EPOCHS) as prg:
        for i in prg:
            optimizer.zero_grad()  # Reset gradients in optimizer record, otherwsie grad will be accumualted.
            out = seq(inputs)  # Equivalently, seq.forward(inputs)
            loss = criterion(out, target)
            prg.set_description(f"Epoch [{i}], loss: {loss.item()}")
            loss.backward()
            optimizer.step()

    # begin to predict, no need to track gradient here
    with torch.no_grad():
        future = 1
        pred_train = seq(inputs, future=future)
        loss = criterion(pred_train[:, :-future], target)
        print("train loss", loss.item())

    with torch.no_grad():
        future = 1
        pred_test = seq(test_inputs, future=future)
        loss = criterion(pred_test[:, :-future], test_targets)
        print('test loss:', loss.item())
        # y = pred.detach().numpy()  # Fetch the result.

    # Actual forecast is the first element of every array.
    extract = lambda x: x.detach().numpy()[..., 0]
    forecast = np.concatenate(
        [extract(pred_train), extract(pred_test)])
    actual = np.concatenate(
        [extract(target), extract(test_targets)])
    plt.plot(forecast, label="Predicted")
    plt.plot(actual, label="Actual")
    plt.title("One step forecast")
    plt.legend()
    plt.show()


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
