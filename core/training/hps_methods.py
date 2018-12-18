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


def gen_hparam_set(
    src_dict: Dict[str, Union[List[object], object]]
) -> List[Dict[str, object]]:
    """
    Generate a collection of hyperparameters for hyparam searching.
    NOTE: in this version, a parameter configuration object is a 
        dictionary with string as keys. When a parameter config
        is fed into a model training session, we use 'globals().update(param_config)'
        to read the config.
    Args:
        src_dict:
            A dictionary with string-keys exactly the same as the sample
            input config below.
            NOTE: for parameters that one wish to search over certain set
                of potential choices, put a LIST of feasible values at the 
                corresponding value in src_dict.
            Example:
                to search over learning_rate parameter,
                set src_dict["learning_rate"] = [0.1, 0.03, 0.01] etc.
    Returns:
        A list (iterable) with all combination of candidates in 
            flexiable (to be searched) parameters.
    """

    # ======== EXAMPLE INPUT ========
    sample_param = {
        "epochs": 1500,
        "num_time_steps": [6, 12, 24],
        "num_inputs": 1,
        "num_outputs": 1,
        "num_neurons": [
            (256, 128),
            (256, 128, 64),
            (512, 256),
            (512, 256, 128),
            (1024, 512),
            (1024, 512, 256)
        ],
        "learning_rate": [0.3, 0.1, 0.03],
        "report_periods": 10,
        "clip_grad": None,
        "tensorboard_dir": "/home/ec2-user/hps_test/tensorboard/",
        "model_path": "/home/ec2-user/hps_test/saved_models/",
        "fig_path": "/home/ec2-user/hps_test/model_figs/"
    }
    # ======== END ========
    # ======== Args Check ========
    assert all(
        k in sample_param.keys()
        for k in src_dict.keys()
    ), "Missing parameter(s) detected."
    # ======== END ========
    gen = list()
    detected_list_keys = list()
    detected_list_vals = list()

    for k, v in src_dict.items():
        if isinstance(v, list):
            detected_list_keys.append(k)
            detected_list_vals.append(v)

    cartesian_prod = list(itertools.product(*detected_list_vals))

    for coor in cartesian_prod:
        new_para = copy.deepcopy(src_dict)
        hparam_str = "-".join(
            f"{k}={v}" for k, v in zip(detected_list_keys, coor))
        for i, key in enumerate(detected_list_keys):
            new_para[key] = coor[i]
        new_para["tensorboard_dir"] += hparam_str
        new_para["model_path"] += hparam_str
        new_para["fig_path"] += hparam_str
        new_para["hparam_str"] = hparam_str
        gen.append(new_para)

    print(f"Total number of parameter sets generated: {len(gen)}")
    return gen


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
