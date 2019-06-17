"""
Converting time series to suprevised learning problems.
"""
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import data_proc


class GenericGenerator():
    """
    Baseline supervised learning problem generator,
    refer to the structure of methods in this class to write document.
    """

    def __init__(self, main_df: pd.DataFrame, verbose=True):
        self.df = main_df.copy()
        self.v = verbose
        if self.v:
            data_proc.summarize_dataset(self.df)

    def get_many_to_many(self):
        raise NotImplementedError()

    def get_many_to_one(self):
        raise NotImplementedError()


class SlpGenerator(GenericGenerator):
    def __init__(self, main_df: pd.DataFrame, verbose=True):
        super().__init__(main_df=main_df, verbose=verbose)

    def get_many_to_many(
        self,
        lag: int=6
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Generate the many-to-many (shift) supervised learning problem.
        For one period forecasting.
        For each time step t, the associated fea, tar are
            fea: [t-lag, t-lag+1, ..., t-1]
            tar: [t-lag+1, t-lag+2, ..., t]
        """
        lagged = [self.df.shift(i) for i in range(lag + 1)]
        col_names = [f"lag[{i}]" for i in range(lag + 1)]
        frame = pd.concat(lagged, axis=1)
        frame.columns = col_names
        frame.dropna(inplace=True)
        # In N-to-N models.
        fea = frame.iloc[:, 1:]
        tar = frame.iloc[:, :-1]
        assert fea.shape == tar.shape, \
            f"The shape of features and targets in the N-to-N supervised \
                learning problem should be the same. \
                Shapes received: X@{fea.shape}, Y@{tar.shape}."
        if self.v:
            print(f"X@{fea.shape}, Y@{tar.shape}")

        # Cast the datatype to float 32, and swap order.
        c = lambda x: x.astype(np.float32)
        swap = lambda x: x[x.columns[::-1]]
        return swap(c(fea)), swap(c(tar))

    def get_many_to_one(
        self,
        lag: int=6
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Generate the many-to-one (last) supervised learning problem.
        For each time step t, the associated fea, tar are
            fea: [t-lag, t-lag+1, ..., t-1]
            tar: [t]
        """
        indices = list(self.df.index)
        fea_lst, tar_lst, idx_lst= [], [], []
        for (i, t) in enumerate(indices):
            # ==== Debug ====
            assert t == indices[i]
            # ==== End ====
            cur_fea = self.df.iloc[i-lag:i]
            if cur_fea.empty:
                if self.v:
                    print(f"\tAt timestep {t}, insufficient lags, dropped.")
                continue
            cur_tar = self.df.iloc[[i]]
            # Extract
            e = lambda x: np.squeeze(x.values)
            fea_lst.append(e(cur_fea))
            tar_lst.append(e(cur_tar))
            idx_lst.append(t)
        col_names = [f"lag[{i}]" for i in range(1, lag+1)][::-1]
        fea = pd.DataFrame(
            data=fea_lst,
            index=idx_lst,
            columns=col_names
            )
        tar = pd.DataFrame(
            data=tar_lst,
            index=idx_lst,
            columns=["Target"]
            )

        c = lambda x: x.astype(np.float32)
        swap = lambda x: x[x.columns[::-1]]
        fea, tar = swap(c(fea)), swap(c(tar))
        assert len(fea) == len(tar), \
            "The number of observations in feature and target \
            data frame do not agree."
        if self.v:
            print(f"X@{fea.shape}, Y@{tar.shape}")
        return swap(c(fea)), swap(c(tar))


    def get_tensors(
        self,
        mode: Union["NtoN", "Nto1"],
        lag: int=6,
        shuffle: bool=True,
        batch_size: int=32,
        validation_ratio: float=0.2,
        pin_memory: bool=False
    ) -> (DataLoader, DataLoader, TensorDataset, TensorDataset):
        """
        Primary goal: create dataloader object.
        """
        if mode == "NtoN":
            x_train, y_train = self.get_many_to_many(lag=lag)
        elif mode == "Nto1":
            x_train, y_train = self.get_many_to_one(lag=lag)
        else:
            raise ValueError(
                f"Undefined mode, avaiable: NtoN, Nto1. Received {mode}.")
        # Transform DataFrame to NumpyArray.
        x_train, y_train = map(lambda x: x.values, (x_train, y_train))
        # Generating Validation Set.
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=validation_ratio, shuffle=shuffle
        )
        # Transform to Tensor
        x_train, y_train, x_val, y_val = map(
            torch.tensor, (x_train, y_train, x_val, y_val)
        )
        # if validation_ratio > 0:
        #     assert batch_size <= x_train.shape[0] and batch_size <= x_val.shape[0],\
        #         "Batch size cannot be greater than number of training or validation instances."

        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

        val_ds = TensorDataset(x_val, y_val)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

        return train_dl, val_dl, train_ds, val_ds


if __name__ == "__main__":
    # The artificial time series
    df2 = pd.DataFrame(
        data=[x+0.1 for x in list(range(30))],
        index=range(1980, 2010))
    g = SlpGenerator(df2)
    fea, tar = g.get_many_to_one()
    print("features")
    print(fea.head())
    print(fea.tail())
    print("targets")
    print(tar.head())
    print(tar.tail())
