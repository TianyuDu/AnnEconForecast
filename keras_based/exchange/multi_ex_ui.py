"""
Multivariate Version of exchange prediciton.
"""
import os
os.system("clear")
import sys
sys.path.append("./core/containers/")
sys.path.append("./core/models/")
sys.path.append("./core/tools/")
import datetime

import keras
import pandas as pd
import numpy as np
import matplotlib
# TODO: add auto-detect
# for mac OS: os.name == "posix" and sys.platform == "darwin"
# Use this identifier to automatically decide the following.
on_server = bool(int(input("Are you on a server wihtout graphic output? [0/1] >>> ")))
if on_server:
    matplotlib.use(
        "agg",
        warn=False,
        force=True
        )
from matplotlib import pyplot as plt
import sklearn

from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import HoverTool
from bokeh.io import show, output_file

from typing import Union, List

# import config
# import methods
# from methods import *
# from models import *
from multi_config import *

from multivariate_container import MultivariateContainer
from multivariate_lstm import MultivariateLSTM
from bokeh_visualize import advanced_visualize as bvis


def train_new_model():
    """
    Train a new model.
    """
    print(f"Control: Building new container from {file_dir}...")
    print(f"\tTarget is {target}")
    # Build up containers.
    container = MultivariateContainer(
        file_dir,
        target,
        load_multi_ex,
        CON_config)
    print(chr(9608))

    print("Control: Building up models...")
    model = MultivariateLSTM(container, NN_config)
    print(chr(9608))

    model.fit_model(epochs=int(input("Training epochs >>> ")))
    
    save_destination = input("Folder name to save model? [Enter] Using default >>> ")
    print("Control: Saving model training result...")
    if save_destination == "":
        model.save_model()
    else:
        model.save_model(file_dir=save_destination)
    print(chr(9608))


def visualize_training_result():
    print(f"Contro;: Building up container from {file_dir}...")
    container = MultivariateContainer(
        file_dir,
        target,
        load_multi_ex,
        CON_config)
    print(chr(9608))

    print("Control: Building empty model...")
    model = MultivariateLSTM(container, NN_config, create_empty=True)
    print(chr(9608))

    load_target = input("Model folder name >>> ")
    load_target = f"./saved_models/{load_target}/"
    print(f"Control: Loading model from {load_target}...")

    model.load_model(
        folder_dir=load_target
    )
    print(chr(9608))

    # Forecast testing set.
    yhat = model.predict(model.container.test_X)
    yhat = model.container.invert_difference(
        yhat, 
        range(
            model.container.num_obs-len(yhat), 
            model.container.num_obs
        ),
        fillnone=True
    )
    # Forecast trainign set.
    train_yhat = model.predict(model.container.train_X)
    train_yhat = model.container.invert_difference(
        train_yhat, range(len(train_yhat)), fillnone=True
    )
    
    # Visualize
    plt.close()
    plt.plot(yhat, linewidth=0.6, alpha=0.6, label="Test set yhat")
    plt.plot(train_yhat, linewidth=0.6, alpha=0.6, label="Train set yhat")
    plt.plot(model.container.ground_truth_y, linewidth=1.2, alpha=0.3, label="actual")
    plt.legend()
    action = input("Plot result? \n\t[P] plot result. \n\t[S] save result. \n\t>>>")
    assert action.lower() in ["p", "s"], "Invalid command."

    if action.lower() == "p":
        plt.show()
    elif action.lower() == "s":
        fig_name = str(datetime.datetime.now())
        plt.savefig(f"./figure/{fig_name}.svg")
        print(f"Control: figure saved to ./figure/{fig_name}.svg")

if __name__ == "__main__":
    print("""
    =====================================================================
    Hey, you are using the Multivariate Exchange Rate Forecasting Model
        This is a neural network developed to forecast economic indicators
        The model is based on Keras
    @Spikey
        Version. 0.0.1, Sep. 11 2018
    Important files
        Configuration file: ./multi_config.py
        Model definition file: ./models.py
    """)

    task = input("""
    What to do?
        [N] Train new model.
        [R] Restore saved model and continue training.
        [V] Visualize training result using matplotlib.
        [B] Visualize training result using bokeh.
        [Q] Quit.
    >>> """)
    assert task.lower() in ["n", "r", "v", "q", "b"], "Invalid task."
    if task.lower() == "n":
        train_new_model()
    elif task.lower() == "r":
        raise NotImplementedError
    elif task.lower() == "v":
        visualize_training_result()
    elif task.lower() == "b":
        bvis(
            file_dir=file_dir,
            target=target,
            load_multi_ex=load_multi_ex,
            CON_config=CON_config,
            NN_config=NN_config
        )
    elif task.lower() == "q":
        quit()
