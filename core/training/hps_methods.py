"""
This script contains helper functions implemented
for hyper-parameter searching / grid searching
"""
import copy
import itertools
import os
from typing import Dict, List, Union

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

import core.tools.metrics as metrics
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
     y_train, y_val, y_test) = rnn_prepare.split_dataset(
        raw=df_ready,
        train_ratio=param["TRAIN_RATIO"],
        val_ratio=param["VAL_RATIO"],
        lags=param["LAGS"]
    )

    # The gross dataset excluding the test set.
    # Excluding the test set for isolation purpose.
    model_data_feed = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }

    # The checkpoint list 
    ckps = checkpoints(param["epochs"] // 10) + [-1]

    predictions = exec_core(
        param=param,
        data=model_data_feed,
        prediction_checkpoints=ckps
    )

    # Summarize VALIDATION set statistics.
    # Fetch the final prediction.
    val_final = list(predictions.values())[-1]["val"]
    print("Final result (validation set):")
    metric_test = metrics.merged_scores(
        actual=pd.DataFrame(y_val),
        pred=pd.DataFrame(val_final),
        verbose=True
    )

    # Visualize prediction during training.
    for set_name in ["train", "val"]:
        pred = dict((e, val[set_name]) for e, val in predictions.items())
        plt.close()
        fig = visualize.plot_checkpoint_individual(
            predictions=pred,
            actual=model_data_feed["y_" + set_name],
            name=set_name)

        if not os.path.exists(param["fig_path"]):
            os.makedirs(param["fig_path"])
        assert not param["fig_path"].endswith("/")
        plt.savefig(param["fig_path"] + "/" + f"pred_record_{s}.svg")
        plt.close()
    
    fig = visualize.plot_checkpoint_combined(
        predictions=predictions,
        actual={"train": y_train, "val": y_val}
    )
    
