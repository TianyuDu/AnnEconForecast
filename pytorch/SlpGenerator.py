"""
Converting time series to suprevised learning problems.
"""
import numpy as np
import pandas as pd

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
                    print(f"At timestep {t}, not sufficient lags, dropped.")
                continue
            cur_tar = self.df.iloc[[i]]
            # Extract
            e = lambda x: np.squeeze(x.values)
            fea_lst.append(e(cur_fea))
            tar_lst.append(e(cur_tar))
            idx_lst.append(t)
        col_names = [f"lag[{i}]" for i in range(1, lag+1)][::-1]
        fea = pd.DataFrame(
            data=fea_lst, index=idx_lst,
            columns=col_names
            )
        tar = pd.DataFrame(data=tar_lst, index=idx_lst,
        columns=["Target"])

        c = lambda x: x.astype(np.float32)
        swap = lambda x: x[x.columns[::-1]]
        fea, tar = swap(c(fea)), swap(c(tar))
        assert len(fea) == len(tar), \
            "The number of observations in feature and target \
            data frame do not agree."
        if self.v:
            print(f"X@{fea.shape}, Y@{tar.shape}")
        return swap(c(fea)), swap(c(tar))


if __name__ == "__main__":
    df2 = pd.DataFrame(list(range(30)))
    g = SlpGenerator(df2)
    fea, tar = g.get_many_to_many()