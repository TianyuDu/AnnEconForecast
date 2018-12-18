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


def individual_train(
    model_param: Dict[str, object],
    data_param: Dict[str, object],
    exec_core: "function",
    file_dir: str
) -> None:
    prepared_df = rnn_prepare.prepare_dataset(
        file_dir=file_dir,
        periods=data_param["PERIODS"],
        order=data_param["ORDER"],
        remove=None,
        verbose=False
    )
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = rnn_prepare.generate_splited_dataset(
        raw=prepared_df,
        train_ratio=data_param["TRAIN_RATIO"],
        val_ratio=data_param["VAL_RATIO"],
        lags=data_param["LAGS"]
    )
    data_collection = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    def checkpoints(z):
        return [
            z*x for x in range(1, model_param["epochs"] // z)
        ] + [-1]

    (metrics_dict, predictions) = exec_core(
        param=model_param,
        data=data_collection,
        prediction_checkpoints=checkpoints(
            model_param["epochs"] // 10
        ) + [-1]
    )
    for s in ["train", "val", "test"]:
        plt.close()
        fig = visualize.plot_checkpoints(
            predictions,
            data_collection["y_" + s],
            s)

        if not os.path.exists(model_param["fig_path"]):
            os.makedirs(model_param["fig_path"])
        assert not model_param["fig_path"].endswith("/")
        plt.savefig(model_param["fig_path"] + "/" + f"pred_record_{s}.svg")
        plt.close()
