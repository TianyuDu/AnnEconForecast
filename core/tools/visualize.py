"""
Visualization tools.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict


def progbar(curr, total, full_progbar):
    """
    Progress bar used in training process.
    Reference: https://geekyisawesome.blogspot.com/2016/07/python-console-progress-bar-using-b-and.html
    """
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(
        full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')


def plot_checkpoint_individual(
    predictions: Dict[int, np.ndarray],
    actual: np.ndarray,
    name: str
) -> (
    matplotlib.figure.Figure,
    "matplotlib.axes._subplots.AxesSubplot"
):
    """
    # TODO: doc string.
    """
    # width = np.ceil(np.sqrt(len(predictions))).astype(int)
    f, axarr = plt.subplots(
        len(predictions.keys()),
        sharex=False,
        sharey=False,
        figsize=(16, 4 * len(predictions.keys())))

    for i, e in enumerate(predictions.keys()):
        axarr[i].plot(predictions[e].reshape(-1, 1), alpha=0.6)
        axarr[i].plot(actual.reshape(-1, 1), alpha=0.6)

        axarr[i].set_title(f"{name} set prediction at [{e}] epochs.")
        axarr[i].legend([f"{name} prediction", f"{name} actual"])
        axarr[i].grid(True)
    return f, axarr


def plot_checkpoint_combined(
    predictions: Dict[int, Dict[str, np.ndarray]],
    actual: Dict[str, np.ndarray]
) -> (
    matplotlib.figure.Figure,
    "matplotlib.axes._subplots.AxesSubplot"
):
    """
    Combined plot for training set and validation set.
    NOTE: assume no shuffling applied, the validation set is immediately after
    the training set in gross.
    """
    # TODO: add time stampe to x-axis.
    len_train = actual["train"]
    len_val = actual["val"]

    f, axarr = plt.subplots(
        len(predictions.keys()),
        sharex=False,
        sharey=False,
        figsize=(16, 4 * len(predictions.keys()))
    )

    for i, e in enumerate(predictions.keys()):
        current_pred = predictions[e]
        axarr[i].plot(range(len_train), current_pred["train"].reshape(-1, 1), alpha=0.6)
        axarr[i].plot(range(len_train), actual["train"].reshape(-1, 1), alpha=0.6)
        axarr[i].plot(range(len_train, len_train + len_val), current_pred["val"].reshape(-1, 1), alpha=0.6)
        axarr[i].plot(range(len_train, len_train + len_val), actual["val"].reshape(-1, 1), alpha=0.6)

        axarr[i].set_title([f"Model Prediction at [{e}] epochs."])
        axarr[i].legend([
            "Train Prediction",
            "Train Actual",
            "Validation Prediction",
            "Validation Actual"],
        loc="best")
        axarr[i].grid(True)
    
    return f, axarr

