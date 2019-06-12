import json
from datetime import datetime
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
import matplotlib
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

# import Logger
import LstmModels
import SlpGenerator

plt.style.use("seaborn-dark")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)

def core(
    DATA_DIR: str,
    TRAIN_SIZE: Union[int, float],
    TEST_SIZE: Union[int, float],
    LAGS: int,
    VAL_RATIO: float,
    LEARNING_RATE: float,
    NEURONS: Tuple[int],
    EPOCHS: int,
    LOG_NAME: str,
    TASK_NAME: str,
    profile_record: dict,
    df_loader: Callable,
    verbose: bool=True # set verbose=False when running hyper-parameter search.
    # To ensure progress bar work correctly
    ) -> None:
    if verbose:
        try:
            input_name = input("Log name ([Enter] for default name): ")
            assert input_name != ""
            LOG_NAME = input_name
        except AssertionError:
            print(f"Default name: {LOG_NAME} is used.")
    
    # ==== TODO ====
    # Extract this portion to an external function used as 
    # an arg in the profile -> Data cleaning call
    # or just use a DataFrame as an arg, so that the csv file
    # is read from hard drive only once.
    
    # ==== END ====
    # TODO: preprocessing date, and write reconstruction script.
    if TRAIN_SIZE < 1 and TEST_SIZE:
        assert TRAIN_SIZE + TEST_SIZE == 1
        TRAIN_SIZE = int(len(df) * TRAIN_SIZE)
        TEST_SIZE = len(df) - TRAIN_SIZE
    
    df_train, df_test = df[:TRAIN_SIZE], df[-TEST_SIZE:]
    
    gen = SlpGenerator.SlpGenerator(df_train, verbose=verbose)
    fea, tar = gen.get_many_to_one(lag=LAGS)
    train_dl, val_dl, train_ds, val_ds = gen.get_tensors(
        mode="Nto1", lag=LAGS, shuffle=True, batch_size=256, validation_ratio=VAL_RATIO,
        pin_memory=False
    )
    # build the model
    # net = LstmModels.PoolingLSTM(lags=LAGS, neurons=NEURONS)
    net = LstmModels.LastOutLSTM(neurons=NEURONS)
    
    # ==== Move everything to GPU ====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print("Device detected: ", device)
    move = lambda x: x.to(device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    
    net.float()  # Cast all floating point parameters and buffers to double datatype
    net = net.to(device)
    net = to_device(net, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    with tqdm.trange(EPOCHS) as prg, SummaryWriter(comment=LOG_NAME) as writer:
        for i in prg:
            train_loss = []
            # TODO: rename all data to feature
            for batch_idx, (data, target) in enumerate(train_dl):
                # data, target = map(torch.Tensor, (data, target))
                # data, target = data.double(), target.double()
                # ========
                # print("data.shape", data.shape)
                # print("data.device: ", data.device)
                # print("target.device: ", target.device)
                # ==== GPU ====
                data = data.to(device).float()
                target = target.to(device).float()
                # ==== END ====
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
            # print(f">>>> Training phase done: {i}")
            if i % 10 == 0:
                val_loss = []
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_dl):
                        # data, target = map(torch.Tensor, (data, target))
                        # data, target = data.double(), target.double()
                        data = data.to(device).float()
                        target = target.to(device).float()
                        out = net(data)
                        loss = criterion(out, target)
                        val_loss.append(loss.data.item())
                # val_log.add(i, np.mean(val_loss))
                writer.add_scalars(
                    "loss/mse", {"Validation": np.mean(val_loss)}, i)
                writer.add_scalars(
                    "loss/rmse", {"Validation": func(val_loss)}, i)
            prg.set_description(
                f"TrainLoss:{np.mean(train_loss): 0.7f}, ValLoss:{np.mean(val_loss): 0.7f}")
            # print(f"Epoch: {i}\tTotal Loss: {train_loss:0.6f}\tLatest Val Loss: {val_loss:0.6f}")
        # TODO: deal with the add graph function here.
        # writer.add_graph(net, (torch.zeros(32, LAGS)))

        # Save the training profile.
        with open("./" + writer.logdir + "/profile.json", "a") as f:
            encoded = json.dumps(profile_record)
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
                feature = feature.to(device).float()
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
