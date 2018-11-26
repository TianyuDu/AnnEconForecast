import sys
import pandas as pd
import numpy as np

import bokeh
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import HoverTool
from bokeh.io import show, output_file
from bokeh.models.widgets import Panel, Tabs

sys.path.append("./core/containers/")
sys.path.append("./core/models/")
sys.path.append("./core/tools/")

from multivariate_container import MultivariateContainer
from multivariate_lstm import MultivariateLSTM


def advanced_visualize(
    file_dir: str,  # Dataset directory.
    target: str,
    load_multi_ex: callable,
    CON_config: dict,
    NN_config: dict,
    show_plot: bool=False
) -> None:
    """
    # TODO: write the doc.
    Predict and Visualize the result.
    """
    print("[IPR]Visualize model result using bokeh...")
    print(f"Building up from container with data at {file_dir}...")

    container = MultivariateContainer(
        file_dir,
        target,
        load_multi_ex,
        CON_config)

    print("Building empty model placeholder...")
    model = MultivariateLSTM(
        container,
        NN_config,
        create_empty=True)

    # The folder where model is stored.
    load_target = input("Model directory >>> ")
    load_target = f"./saved_models/{load_target}/"
    print(f"Loading model from directory: {load_target}...")
    model.load_model(folder_dir=load_target)

    # timeline = pd.DatetimeIndex(container.dataset.index)
    # Time line for x-axis

    # true_y = np.diff(model.container.get_true_y())
    output_file(f"{load_target}visualized.html")
    print(f"Saving plotting html file to {load_target}visualized.html...")

    # ======== Differenced Value ========
    print("Building up forecasting sequence for differenced value...")
    train_yhat = model.predict(model.container.train_X)
    train_yhat = np.squeeze(train_yhat)

    test_yhat = model.predict(model.container.test_X)
    test_yhat = np.squeeze(test_yhat)

    timeline = pd.DatetimeIndex(model.container.dataset.index)
    true_y = np.diff(model.container.get_true_y())

    # ================ Training Set ================
    plot_diff_train = bokeh.plotting.figure(
        title="Differencing value, training set",
        x_axis_label="Date",
        y_axis_label="Differenced Value",
        x_axis_type="datetime",
        plot_width=1400,
        plot_height=400
    )

    plot_diff_train.line(
        timeline[1: len(train_yhat)+1],
        np.squeeze(model.container.train_y),
        color="navy",
        alpha=0.4,
        legend="Training Set Actual Values"
    )

    plot_diff_train.line(
        timeline[1: len(train_yhat)+1],
        train_yhat,
        color="red",
        alpha=0.7,
        legend="Training Set Predicted Values"
    )

    # ================ Testing Set ================
    plot_diff_test = bokeh.plotting.figure(
        title="Differencing value, testing set",
        x_axis_label="Date",
        y_axis_label="Differenced Value",
        x_axis_type="datetime",
        plot_width=1400,
        plot_height=400
    )

    plot_diff_test.line(
        timeline[-len(test_yhat):],
        np.squeeze(model.container.test_y),
        color="navy",
        alpha=0.4,
        legend="Test Set Actual Values"
    )

    plot_diff_test.line(
        timeline[-len(test_yhat):],
        test_yhat,
        color="red",
        alpha=0.7,
        legend="Test Set Predicted Values"
    )

    tab_diff = bokeh.models.widgets.Panel(
        child=column(children=[plot_diff_test, plot_diff_train]),
        title="Differenced values"
    )

    # ======== Raw scale values ========

    train_yhat = model.predict(model.container.train_X)
    train_yhat = model.container.invert_difference(
        train_yhat,
        range(len(train_yhat)), fillnone=True
    )
    train_yhat = np.squeeze(train_yhat).astype(np.float32)

    test_yhat = model.predict(model.container.test_X)
    test_yhat = model.container.invert_difference(
        test_yhat,
        range(
            model.container.num_obs - len(test_yhat),  # Last n observation.
            model.container.num_obs),
        fillnone=True
    )
    test_yhat = np.squeeze(test_yhat).astype(np.float32)

    timeline = pd.DatetimeIndex(model.container.dataset.index)

    raw_plot = figure(
        title="Aggregate Graph: all",
        x_axis_label="Date",
        y_axis_label="Actual Value",
        x_axis_type="datetime",
        plot_width=1400,
        plot_height=400
    )

    raw_plot.line(
        timeline,
        model.container.get_true_y(),
        color="navy",
        alpha=0.3,
        legend="Actual values"
    )

    raw_plot.line(
        timeline,
        train_yhat,
        color="red",
        alpha=0.7,
        legend="Training set predictions"
    )

    raw_plot.line(
        timeline,
        test_yhat,
        color="green",
        alpha=0.7,
        legend="Testing set predictions"
    )

    tab_raw = bokeh.models.widgets.Panel(
        child=raw_plot,
        title="Raw scale"
    )

    # ================ Training information ================
    hist = pd.read_csv(f"{load_target}hist.csv")
    loss = np.squeeze(hist["loss"])
    val_loss = np.squeeze(hist["val_loss"])

    num_epochs = len(loss)

    log_loss = np.log(loss)
    log_val_loss = np.log(val_loss)

    # ======== RAW ========

    loss_plot = figure(
        title="Loss Record(Raw)",
        x_axis_label="Epoch",
        y_axis_label="Loss",
        plot_width=1400,
        plot_height=400
    )

    loss_plot.line(
        range(num_epochs),
        loss,
        color="red",
        alpha=0.7,
        legend="loss on training set"
    )

    loss_plot.line(
        range(num_epochs),
        val_loss,
        color="green",
        alpha=0.7,
        legend="loss on validation set"
    )

    # ======== LOG ========
    log_loss_plot = figure(
        title="Loss Record(Log)",
        x_axis_label="Epoch",
        y_axis_label="Log Loss",
        plot_width=1400,
        plot_height=400
    )

    log_loss_plot.line(
        range(num_epochs),
        log_loss,
        color="red",
        alpha=0.7,
        legend="log loss on training set"
    )

    log_loss_plot.line(
        range(num_epochs),
        log_val_loss,
        color="green",
        alpha=0.7,
        legend="log loss on validation set"
    )

    tab_hist = bokeh.models.widgets.Panel(
        child=column(children=[loss_plot, log_loss_plot]),
        title="Training History"
    )

    # ================ Finalizing ================
    tabs = bokeh.models.widgets.Tabs(tabs=[tab_raw, tab_diff, tab_hist])

    if show_plot:
        bokeh.io.show(tabs)
    else:
        bokeh.io.save(tabs)
