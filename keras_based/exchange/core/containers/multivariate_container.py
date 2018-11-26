"""
This file contains data container instance for multivariate time series (panels)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Union


from base_container import BaseContainer


class MultivariateContainer(BaseContainer):
    """
        Multivariate container for RNN prediction problem.
    """

    def __init__(
            self,
            file_dir: str,
            target_col: str,
            load_data: callable,
            config: dict={
                "max_lag": 3,
                "train_ratio": 0.9,
                "time_steps": 14,
                "drop_target": False
            }):
        # ======== Pre-requiste ========
        self.__check_config(config)
        self.config = config  # Load configuration.

        # ======== Preprocessing Data ========
        self.dataset = load_data(file_dir)
        assert type(
            self.dataset) is pd.DataFrame, \
            f"Illegal object returned by data retrieving method, expected: \
            pd.DataFrame, got: {type(self.dataset)} instead."

        assert target_col in self.dataset.columns, f"Target column received: {target_col} \
        cannot be found in DataFrame loaded."

        self.target_col = target_col
        print(f"\tTarget variable received: {self.target_col}")

        print("[Done]Dataset loaded succesfully.")

        # Actual Dataset
        self.values = self.dataset.values
        self.num_obs, self.num_fea = self.values.shape
        print(
            f"\tDataset with {self.num_obs} observations and {self.num_fea} variables. \
            \n\tDataset shape={self.dataset.shape}"
        )

        # Differencing to remove non-stationarity.
        self.diff_dataset = self.dataset.diff()
        self.diff_dataset.fillna(0.0, inplace=True)

        self.X, self.y = self.generate_supervised_learning(
            data=self.diff_dataset,
            time_steps=self.config["time_steps"],  # Time step of look back.
            drop_target=self.config["drop_target"]
        )

        self.train_size = int(
            self.config["train_ratio"] * self.num_obs)  # Get training set size

        (self.train_X, self.train_y, self.test_X, self.test_y) \
            = self.split_data(self.X, self.y, train_size=self.train_size)

        # Scaler run over training the training set.
        # (self.scaled_train_X, self.scaled_train_y, self.scaler_X, 
        #  self.scaler_y) = self.scale_data(self.train_X, self.train_y)

        # self.scaled_test_X = self.scaler_X.transform(self.test_X)
        # self.scaled_test_y = self.scaler_y.transform(self.test_y)

    def __str__(self) -> str:
        return f"""Multivariate Container:
        ============Pre-processing Method============
        Pre-processing method: Differencing
        ============Raw Data============
        Aggregate dataset shape: {self.dataset.shape}
        # Observations: {self.num_obs}
        # Features: {self.num_fea}
        ** Differenced dataset shape: {self.diff_dataset.shape}
        ============Supervised Learning Summary============
        ++shape format X: (sample * length_series * num_fature)
        ++shape format y: (sample * 1)
        Time steps of lagged variables: {self.config["time_steps"]}
        Total predictor(X) shape: {self.X.shape}
        Total response(y) shape: {self.y.shape}
        ============Training & Testing Set Splits============
        ++shape format X: (sample * length_series * num_fature)
        ++shape format y: (sample * 1)
        Training ratio: {self.config["train_ratio"]}
        Training set predictor(train_X) shape: {self.train_X.shape}
        Training set response(train_y) shape: {self.train_y.shape}
        Testing set predictor(test_X) shape: {self.test_X.shape}
        Testing set response(test_y) shape: {self.test_y.shape}
        """

    def __repr__(self) -> str:
        return self.__str__()

    # TODO: Improve check config method
    def __check_config(self, config: dict) -> None:
        """
        Assert the configuration dictionary passed into 
        the model has all its attributes leagel.
        """
        print("[IPR]Checking configurations...")
        assert type(config["max_lag"]) is int, \
            "(Illegal Config) \
            Max lag of supervised learning should be an integer."

        assert config["max_lag"] >= 1, \
            "(Illegal Config) Max lag of supervised learning should \
            be greater than or equal to 1."

        assert type(config["train_ratio"]) is float, \
            "(Illegal Config) Train ratio should be a float."

        assert 0.0 < config["train_ratio"] < 1.0, \
            "(Illegal Config) Traio ratio should be on range (0, 1)"

        assert type(config["time_steps"]) is int, \
            "(Illegal Config) Time steps of supervised \
            learning should be an integer."

        assert config["time_steps"] >= 1.0, \
            "(Illegal Config) Time steps of supervised \
            learning should be greater than or equal to 1."
        print("[Done]Passed.")

    def generate_supervised_learning(
            self,
            data: pd.DataFrame,
            time_steps: int,
            drop_target: bool=False) -> (
                np.array,
                np.array):
        """
        This method converts data fram with shape
        (n_sample, n_feature)
        into a supervised time series learning problem.

        The generated input X array should have shape
        (n_sample, time_step, n_feature)

        The generated output y array should have shape
        (n_sample, 1)

        which is the next period value of target variable.
        ============================================================
        Args:
            data: the raw data in format of data frame.

            time_steps: the number of lagged values to be used as
                input X in supervised learning problem.

            drop_target: whether to include the lagged values of value
                to be predicted (y[t-1]) in the input X for SLP.
        Returns:
            X: the array containing all training data X
                (as bulk of series of past/lagged variables)
            y: the corresponding target[t] value for each
                sample generated in the SLP.
        """

        num_obs, num_fea = data.shape

        print(
            f"[IPR]Generating supervise learning problem with {num_fea} variables and total {time_steps} lagged variables.")

        y = data[self.target_col]

        if drop_target:
            data.drop(columns=[self.target_col], inplace=True)
        y = pd.DataFrame(y.values.reshape(-1, 1))

        value = data.values

        # X = [None] * num_obs  
        # If numpy array does not work, use the python built-in list

        X = np.array([None] * num_obs)
        # Create placeholder for predictors.

        for t in range(num_obs):  # Iterating over all raw sample
            if t - time_steps < 0:
                # If there are not enough look back values.
                X[t] = None
                # X[t] = np.zeros([time_steps, value.shape[1]])
            else:
                # Retrive past observations on all concurrent series
                # (including the target).
                X[t] = value[t - time_steps: t, :]

        # X = [sub for sub in X if sub is not None]  # Drop training
        # data without enough look back values.

        empty = np.zeros_like(X[-1])
        X = [empty if sub is None else sub for sub in X]
        X = np.array(X)

        # Check return shape.
        if drop_target:
            print("\tPrevious values of target(y) is NOT included in input(X) set.")
            assert X.shape == (num_obs, time_steps, num_fea - 1), \
                f"Expected shape = {(num_obs, time_steps, num_fea - 1)} \
            Shape received = {X.shape}"
        else:
            print("\tPrevious values of target(y) is included in input(X) set.")
            assert X.shape == (num_obs, time_steps, num_fea)

        # Drop first few target data.
        # So that shape y is the in the same length of X.
        # y = y[-(X.shape[0]):]
        # Above lines seems to be redundant.
        # If no further issues present, delete above comments.

        y = np.array(y).reshape(-1, 1)

        print(
            f"[Done]Supervised Learning Set Generated: X = {X.shape}, y = {y.shape}")
        return X, y

    def split_data(self, X, y, train_size: int) -> Tuple[np.array]:
        """
        Generate training and testing data, both input X and target y.
        """
        # scaler = sklearn.preprocessing.StandardScaler()
        # X = scaler.fit_transform(X)
        print(f"[IPR]Spliting data: train_size {train_size}")

        train_X = X[:train_size, :, :]
        train_y = y[:train_size, :]

        test_X = X[train_size:, :, :]
        test_y = y[train_size:, :]

        print(f"[Done]Split data into training and testing sets \
        \n\ttrain_X = {train_X.shape} \
        \n\ttrain_y = {train_y.shape} \
        \n\ttest_X = {test_X.shape} \
        \n\ttest_y = {test_y.shape}")

        return (train_X, train_y, test_X, test_y)

    def invert_difference(
        self,
        delta: np.ndarray,
        stamps: np.ndarray,
        fillnone: bool=False
            ) -> np.ndarray:
        """
        This function reconstruct the predicted series from differenced series and (past) ground truth values.
        Args:
            delta:
                array containing the predicted differencing value (y[t] - y[t-1]).
                Thus, by definition, delta[t] = y[t] - y[t-1]
                ==> reconstructed y[t] <- delta[t] + y[t-1] for all t provided.

            stamps:
                array containing time index to locate the delta array.

            fillnone:
                bool to indicate if fill the time step t not in stamps with none/nan value.
                filling the blank with none would preserve the index and would make
                visualization easier.
                If passed as True, a full length time series will be returned and
                time stamps that are not in STAMPS will be filled with Nan.
                t in stamps starts with 0.
        """

        assert len(delta) == len(stamps), \
            "Differenced series and time stamp series must have the same length."

        assert all([t in range(self.num_obs) for t in stamps]), \
            f"Some time stamps passed in exceed the limit. Problems caught: {stamps[stamps >= self.num_obs]}"

        # The ground truth value for y sequence.
        hist_y = self.dataset[self.target_col].values

        if fillnone:
            recon = [None] * self.num_obs
            for d, t in zip(delta, stamps):
                recon[t] = hist_y[t - 1] + d
        else:
            recon = list()
            for d, t in zip(delta, stamps):
                recon.append(hist_y[t - 1] + d)

        return recon

    def get_true_y(self) -> np.ndarray:
        return self.dataset[self.target_col].values