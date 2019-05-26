import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
import matplotlib
c = input("Use Agg as matplotlib (avoid tkinter)[y/n]:")
if c.lower() == "y":
    matplotlib.use("agg")
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

# import Logger
import LstmModels
import SlpGenerator

plt.style.use("seaborn-dark")

# import ptc.data_proc as data_proc
# from ptc.model import *

# Default directories for data
CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA = "/home/ec2-user/environment/AnnEconForecast/data/sunspots.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"

# if __name__ == '__main__':
def core(
    TRAIN_SIZE,
    TEST_SIZE,
    LAGS,
    VAL_RATIO,
    LEARNING_RATE,
    NEURONS,
    EPOCHS,
    LOG_NAME,
    TASK_NAME
    ) -> None:
    # globals().update(PROFILE)
    # locals().update(PROFILE)
    # print(locals())
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
    # net = LstmModels.PoolingLSTM(lags=LAGS, neurons=NEURONS)

    net = LstmModels.LastOutLSTM(neurons=NEURONS)

    net.double()  # Cast all floating point parameters and buffers to double datatype
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

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

        # Save the training profile.
        with open("./" + writer.logdir + "/profile.json", "a") as f:
            encoded = json.dumps(PROFILE)
            f.write(encoded)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            gen_test = SlpGenerator.SlpGenerator(df_test, verbose=False)
            fea_df, tar_df = gen_test.get_many_to_one(lag=LAGS)
            assert len(fea_df) == len(tar_df)

            pred = list()
            for i in range(len(fea_df)):
                f = lambda x, idx: torch.Tensor(x.iloc[idx].values)
                feature = f(fea_df, i)
                feature = feature.view(1, feature.shape[0])
                feature = feature.double()
                pred.append(net(feature))
            
            pred_df = pd.DataFrame(
                data=np.array(pred),
                index=tar_df.index
            )
            total = pd.concat([tar_df, pred_df], axis=1)
            total.columns = ["Actual", "Forecast"]
            mse = np.mean((total["Actual"] - total["Forecast"])**2)
            fig = plt.figure(dpi=200)
            # plt.plot(np.random.rand(10), linewidth=0.7, alpha=0.6)
            plt.plot(total)
            plt.grid()
            plt.title(f"Test Set of {TASK_NAME} After {EPOCHS} Epochs")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend(["Actual", f"Forecast, MSE={mse}"])
            writer.add_figure(
                "Test set predictions", fig, global_step=EPOCHS
            )

# with torch.no_grad():
#     future = 1
#     pred_test = seq(val_ds.tensors[0], future=future)
#     loss = criterion(pred_test[:, :-future], val_ds.tensors[1])
#     print('test loss:', loss.item())
#     # y = pred.detach().numpy()  # Fetch the result.
