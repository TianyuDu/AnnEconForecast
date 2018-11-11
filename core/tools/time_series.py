"""
This file contains all operation methods on time series data.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def differencing(
    src_df: pd.DataFrame,
    order: int=1,
    periods: Union[int, List[int]]=1
) -> pd.DataFrame:
    """
    Generate the differenced data frame of the given data frame.
    With order=1:
    After[t] = Original[t] - Original[t-periods].
    The differencing will be called again if order > 1.

    Args:
        src_df:
            A time-series or panel dataframe.
        order:
            Order of differencing, the total times of recursive differencing will be called.
        periods:
            Nummber of periods of looking back while each recursive differncing stage is executed.
            To implement different periods of looking back in each stage of differencing, pass 
            a list of integers with length equals order of differencing.
            To use the same periods of looking back in each stage, pass an integer.
    Returns:
        [0] A data frame containing the result, some of data points are Nan.
    """
    assert isinstance(periods, list) or isinstance(
        periods, int), "periods should be an integer or a list of integers."
    if isinstance(periods, list):
        assert len(
            periods) == order, "list periods should have the length of order."
        assert all([isinstance(p, int) for p in periods]
                   ), "all elements in list of periods should be integers."

    df = src_df.copy()

    # Alternatively, the for loop can be implement as a recursion.
    for i in range(order):
        lookback = periods[i] if isinstance(periods, list) else periods
        df = df.diff(periods=lookback)

    # Rename column.
    new_cols = [col+f"_period{periods}_order{order}" for col in df.columns]
    df.columns = new_cols
    return df


def invert_diff(
    src_df: pd.DataFrame,
    order: int=1,
    periods: int=1
) -> pd.DataFrame:
    raise NotImplementedError


def gen_supervised_dnn(
    src_df: pd.DataFrame,
    predictor_lags: List[int],
    predictor_orders: List[int]
) -> Tuple[np.ndarray]:
    """
    Generate Supervised Learning Problem for basic deep neural networks.
    Predictor is NOT a time series. 
    Predictors are non-consecutive.

    Args:
        src_df:
            A time-series or panel dataframe.
        predictor_lags:
            List of integers, lagged values with degree given that would 
            be used as predictors of the value at each time period.
        predictor_orders:
            An integer or a list of integersorders that will be applied for each lagged predictor above.
            If an integer is passed in, all lagged will be applied with the same order.
            If a list is passed, it should have the same length as predictor_lags.

            Example: predictor_lags = [1, 2, 3, 5] and predictor_orders = 1
                x[t-1], x[t-2], x[t-3] and x[t-5] will be used as predictors to predict x[t].
                And those five values above contribute one observation in data.
    Returns:
        [0] Predictor array with shape (num_obs, len(predictor_index))
        [1] Response array with shape (num_obs, 1)
    """
    assert all([isinstance(i, int) for i in predictor_lags]
               ), "all elements in predictor lags should be integers."

    if isinstance(predictor_orders, list):
        assert len(predictor_orders) == len(
            predictor_lags), "order and lag lists should have the same length."
        assert all([(i > 0 and isinstance(i, int)) for i in predictor_orders]
                   ), "all elements in predictor orders should be positive integers."
        orders = predictor_orders
    elif isinstance(predictor_orders, int):
        assert predictor_orders > 1, "predictor order should be a positive integer."
        orders = [predictor_orders] * len(predictor_lags)
    else:
        raise TypeError(
            "predictor_orders should be either an integer or a list of integers.")

    df = src_df.copy()
    main_name = df.columns[0]
    df.columns = [f"{main_name}_target"]

    cols = list()
    for (o, l) in zip(orders, predictor_lags):
        predictor = differencing(df, order=o, periods=l)
        predictor.columns = [f"{main_name}_periods{l}_order{o}"]
        cols.append(predictor)

    X = pd.concat(cols, axis=1)
    return X.values, df.values


def gen_supervised_sequence(
    src_df: pd.DataFrame,
    lags: int,
    target_column: str,
    sequential_label: bool=False,
) -> Tuple[np.ndarray]:
    """
    CURRENT VERSION: SINGLE OUTPUT SERIES, SO NUM_OUTPUTS = 1.
    AND ONE-PERIOD FORECASTING ONLY. IF MULTIPLE PREDICTION IS NEEDED, COULD RUN PREDICTION RECURSIVELY.
    The supervised learning problem generator for recurrent neural networks,
    training predictors is a consecutive series.
    Args:
        src_df:
            Source time series or panel data in DataFrame format.
        lags:
            Total number of lags used as predictors.
            Example: lags=10
                Then sequence (x[t-10], x[t-9], ..., x[t-1]) will be used as training predictor
                to predict x[t].
        target_column:
            Column name of target column in source panel data.
        sequence_label:
            See returns section below.    
    Returns:
        X: The array with shape (num_obs, lags, num_inputs), where num_inputs is the number of columns in src_df.
        y: if sequential_label is True, then with shape (num_obs, lags, 1).
        if False, with shape (num_obs, 1, 1)
        Example: with lags=10,
            If sequential_label=True, a typical element in y would be (x[t-9], x[t-8], ..., x[t]).
            If sequential_label=False, a typical element wooudl be (x[t]).
    """
    assert lags > 0, "Lags(Periods of looking back) must be positive."
    X = src_df.copy()  # Predictor(s).

    # Copy the one-period-future value of target series and
    # use it as the label.
    target = X[target_column].shift(-1)

    # Append the label column to predictors.
    # So the last column of df is the future(shifted) target.
    df = pd.concat([X, target], axis=1)
    df.dropna(inplace=True)

    observations = [df[t:t+lags].values for t in range(len(df)-lags)]

    observations = np.array(observations)
    num_obs = len(observations)
    print(f"Total {num_obs} observations generated.")

    if sequential_label:
        X, y = (observations[:, :, :-1],
                observations[:, :, -1].reshape(num_obs, lags, 1))
    else:
        X, y = (observations[:, :, :-1],
                observations[:, -1, -1].reshape(num_obs, 1, 1))

    print("Note: shape format: (num_obs, time_steps, num_inputs/outputs)")
    print(f"X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def clean_nan(
    X: pd.DataFrame,
    y: pd.DataFrame
) -> Tuple[pd.DataFrame]:
    target_col = list(y.columns)
    aggreagate = pd.concat([X, y], axis=1)
    ori_len = len(aggreagate)

    aggreagate.dropna(inplace=True)
    new_len = len(aggreagate)
    print(f"{ori_len - new_len} ({(ori_len - new_len) / ori_len * 100:0.2f}%) observations with Nan are dropped.")

    return (
        aggreagate.drop(columns=target_col),
        aggreagate[target_col]
    )
