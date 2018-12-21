# ANNEF

> Artificial Neural Networks in Economic Forecasting

## About this Project

ANNEF is a project focusing on the interdisciplinary areas of economics, computer science, and statistics.

With recent advances in artificial neural networks, ANNs are shown to be excellent in image recognition and translation tasks. However, we found relatively rare work done on examining the power of artificial neural networks on time series tasks.

In this project, we aim to implement a wide range of models, both from structural and non-structural, to forecast various economic indicators, including unemployment and foreign exchange rates.

#### Documentation Sitemap

* [The Project Methodology and Workflow](method.md)

* [The First Demonstration](demo1.md)

* [The Second Demonstration](demo2.md)

* [The Project Road Map](roadmap.md)

* [How to Train a Model](train.md)

* [How to Restore a Saved Model](restore.md)

* [Hyper-Parameter Tuning and Searching](hps.md)

* [A List of References](ref.md)



## Project Directory Layout

#### Main Model

- `/core`  core files containing codes
- `/data` dataset directory
- `/notebooks`  Jupyter notebooks
- `/saved_models`  this is the default directory for TensorFlow to store models after training.
- `/tensorboard`  this is the default directory for TensorFlow to store tensor board visualization files.

#### Archived Models

- `/keras_based`  models built on `keras`
- `/matlab_based`  models built on `MatLab` 

## Packages

In this project, we chose `pandas` ,  `numpy` and `sklearn` packages to handle data pre-processing tasks. 

For neural networks in this project, we implemented them using `tensorflow` package. 

As well,  `tensorboard` library helps network graph and training visualization.

Statistical models from time series analysis are used a benchmarks in this project. 

Comparisons among neural networks and statistical models allow we to evaluate the forecasting performance, in terms accuracy, of those networks we built. 

Those models are implemented using `statsmodels` package.
