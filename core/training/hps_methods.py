"""
This script contains helper functions implemented
for hyper-parameter searching / grid searching
"""
import os
import copy
import itertools
from typing import Dict, List, Union

import matplotlib
from matplotlib import pyplot as plt

import core.tools.rnn_prepare as rnn_prepare
import core.tools.visualize as visualize


def checkpoints(period, total):
    """
    A helpful function for individual train method.
    to generate checkpoint list with integers 
    for every PERIOD time steps and TOTAL time steps in total.
    """
    return [
        period * x for x in range(1, total // period)
    ]


def individual_train(
    param: Dict[str, object],
    exec_core: "function",
    file_dir: str
) -> None:
    """
    TODO: doc string
    """
    # Generate the dataset.
    df_ready = rnn_prepare.prepare_dataset(
        file_dir=file_dir,
        periods=param["PERIODS"],
        order=param["ORDER"],
        remove=None,
        verbose=False
    )

    # Split dataset.
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = rnn_prepare.generate_splited_dataset(
        raw=df_ready,
        train_ratio=param["TRAIN_RATIO"],
        val_ratio=param["VAL_RATIO"],
        lags=param["LAGS"]
    )
    data_collection = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    (metrics_dict, predictions) = exec_core(
        param=param,
        data=data_collection,
        prediction_checkpoints=checkpoints(
            param["epochs"] // 10
        ) + [-1]
    )

    # Visualize
    for s in ["train", "val", "test"]:
        plt.close()
        fig = visualize.plot_checkpoints(
            predictions,
            data_collection["y_" + s],
            s)

        if not os.path.exists(param["fig_path"]):
            os.makedirs(param["fig_path"])
        assert not param["fig_path"].endswith("/")
        plt.savefig(param["fig_path"] + "/" + f"pred_record_{s}.svg")
        plt.close()
