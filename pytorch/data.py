"""
Mar. 12 2019
Generate one step batch training from fred dataset.
"""
import numpy as numpy
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
plt.style.use("seaborn-dark")
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Summarized time series information.
    """
    print(">>>> Dataset Received <<<<")
    print(f"* Date range from {df.index[0]} to {df.index[-1]}")
    print(f"* Number of observations: {len(df)}")
    try:
        print(f"* Features: Number of features: {df.shape[1]}")
    except IndexError:
        print(f"* Features: Univariate series.")


def generate_supervised(
    df: pd.DataFrame,
    lag: int = 6
) -> (pd.DataFrame, pd.DataFrame):
    summarize_dataset(df)
    lagged = [df.shift(i) for i in range(lag + 1)]
    col_names = [f"lag[{i}]" for i in range(lag + 1)]
    frame = pd.concat(lagged, axis=1)
    frame.columns = col_names
    frame.dropna(inplace=True)
    # In N-to-N models, 
    features = frame.iloc[:, 1:]
    target = frame.iloc[:, :-1]
    assert features.shape == target.shape, "Something went wrong."
    print(f"X@{features.shape}, Y@{target.shape}")
    return features, target


def gen_data_tensors(
    df: pd.DataFrame,
    lag: int = 6,
    batch_size: int = 32,
    validation_ratio: float = 0.2
) -> (DataLoader, DataLoader, TensorDataset, TensorDataset):
    """
    Primary goal: create dataloader object.
    """
    x_train, y_train = generate_supervised(df, lag=lag)
    # Transform DataFrame to NumpyArray.
    x_train, y_train = map(lambda x: x.values, (x_train, y_train))
    # Generating Validation Set.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=validation_ratio, shuffle=True
    )

    # Transform to Tensor
    x_train, y_train, x_val, y_val = map(
        torch.tensor, (x_train, y_train, x_val, y_val)
    )

    assert batch_size <= x_train.shape[0] and batch_size <= x_val.shape[0],\
        "Batch size cannot be greater than number of training instances."
    
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = TensorDataset(x_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl, train_ds, val_ds


if __name__ == "__main__":
    df = pd.read_csv(
        "./sunspot.csv",
        index_col=0,
        date_parser=lambda x: datetime.strptime(x, "%Y")
    )
    # diff = df.diff()
    # diff.dropna(inplace=True)
    # diff.plot()
    # plt.show()
    X, Y = gen_sup(df, lag=60)

    # plt.plot(X.iloc[10, :].values, label="Features")
    # plt.plot(Y.iloc[10, :].values, label="Target")
    # plt.legend()
    # plt.show()
