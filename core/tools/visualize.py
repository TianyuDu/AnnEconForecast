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


def plot_checkpoint_sequential(
    predictions: Dict[int, Dict[str, np.ndarray]],
    actual: np.ndarray,
    name: str
) -> (
    matplotlib.figure.Figure,
    "matplotlib.axes._subplots.AxesSubplot"
):
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


# def plot_checkpoint_box(

# ):
