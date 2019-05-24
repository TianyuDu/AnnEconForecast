"""
Main model for the fully-connected ANN
NNAR, neural network auto-regression.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark")

import tqdm
import torch
import FcModel
import SlpGenerator
import LogUtility

# Settings 
CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"

# Let's call hyper-parameters profile.
PROFILE = {
    "train_size": 231, # Include both training and validation sets.
    "test_size": 58,
    "lags": 6,
    "vr": 0.2, # Validation ratio.
    "neurons": (64, 258),
    "epochs": 100
}

# if __name__ == "__main__":
globals().update(PROFILE)
df = pd.read_csv(
    SUNSPOT_DATA,
    index_col=0,
    date_parser=lambda x: datetime.strptime(x, "%Y")
)
df_train, df_test = df[:train_size], df[-test_size:]
print(f"Train&validation size: {len(df_train)}, test size: {len(df_test)}")

gen = SlpGenerator.SlpGenerator(df_train)
fea, tar = gen.get_many_to_one(lag=lags)
train_dl, val_dl, train_ds, val_ds = gen.get_tensors(
    mode="Nto1", lag=lags, shuffle=True, batch_size=32, validation_ratio=vr
)

net = FcModel.Net(num_fea=lags, num_tar=1, neurons=neurons)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
train_log = LogUtility.Logger()
val_log = LogUtility.Logger()

with tqdm.trange(epochs) as prg:
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
        train_log.add(i, np.mean(train_loss))
        if i % 5 == 0:
            val_loss = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_dl):
                    data, target = map(torch.Tensor, (data, target))
                    out = net(data)
                    loss = criterion(out, target)
                    val_loss.append(loss.data.item())
            val_log.add(i, np.mean(val_loss))
        prg.set_description(
            f"Epoch [{i+1}/{epochs}]: TrainLoss={np.mean(train_loss): 0.3f}, ValLoss={np.mean(val_loss): 0.3f}")

# Create plot
plt.close()
plt.plot(train_log.get_df(ln=True), label="Train Loss")
plt.plot(val_log.get_df(ln=True), label="Validation Loss")
plt.legend()
plt.grid()
plt.show()