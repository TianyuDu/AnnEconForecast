"""
Created: Dec. 29 2018
This script contains methods used in genetic algorithm 
powered hyper parameter searching.
"""
import sys
from typing import Dict, Union
import numpy as np

sys.path.append("../")
import core.tools.rnn_prepare as rnn_prepare

def eval_net(
    model: object,
    param: Dict[str, object],
    file_dir: str,
    metric: str = "mse_val",
    smooth_metric: Union[float, None] = 0.05
) -> float:
    """
    This methods execute a single training session on the neural network
    built based on the spec. provided in param set and evaluate the 
    performance of the network using provided metric.

    It will be passed as an argument to the genetic optimizer
    directly, so it returns a single float number, as required by the
    optimizer.
    
    Args:
        model:
            The model object to be built.

        param:
            A dictionary specifying the following information.
            NOTE: For more detailed info. on the parameter set, refer to
            the template configuration file.
                - Data pre-processing spec.
                - Neural network spec.
                - Model saving spec.

        file_dir:
            The directory of dataset.

        metric:
            The metric used to evaluate the performance.

        smooth_metric:
            None or a float in range (0, 1)
            If None is specified, the metric on the very last epoch will be 
            returned.
            Otherwise, the average of metric evaluated in the last few epochs
            will be returned.
            - Example: 
            if smooth_metric = 0.05, then the average on metric evaluated on the 
            very last 5% epochs will be returned.

    Returns:
        A float specified by 'metric'.
    """
    # ======== Argument Checking ========
    # ======== End ========
    # Prepare the dataset.
    df_ready = rnn_prepare.prepare_dataset(
        file_dir=file_dir,
        periods=int(param["PERIODS"]),
        order=int(param["ORDER"]),
        remove=None,
        verbose=False
    )

    # Split dataset.
    (X_train, X_val, X_test,
     y_train, y_val, y_test)\
    = rnn_prepare.split_dataset(
        raw=df_ready,
        train_ratio=param["TRAIN_RATIO"],
        val_ratio=param["VAL_RATIO"],
        lags=param["LAGS"]
    )

    # The gross dataset excluding the test set.
    # Excluding the test set for isolation purpose.
    data_feed = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }
    ep = param["epochs"]
    ckpts = range(int(ep * 0.95), ep)  # Take the final 5% epochs.
    net = model(
        param=param,
        prediction_checkpoints=ckpts,
        verbose=False
    )
    ret_pack = net.fit(
        data=data_feed,
        ret=["mse_val"]
    )
    return float(np.mean(list(ret_pack["mse_val"].values())))

