# Project Layout
Here is the layout for files in this projects, it helps one to locate the wanted scripts and debug with ease.

## The model backend by Tensorflow library

> This category contains models that we are actively developing.

### The <u>Core</u>

> `AnnEconForecast/core`

This directory contains all scripts handling tasks including data-preprocessing, model building and visualization.

**Note**: generally one does not need to modify files under this directory the only wanted is replicating the results in the demonstration or run existing models on a different dataset (but in the same format).

### Constants

> `AnnEconForecast/constants.py`

This file holds all user-specified constants in a python dictionary fashion.

### Where our <u>datasets</u> go

>  `AnnEconForecast/annefdata`

By default, we placed all dataset files under this directory. Placing dataset there is not a strict requirement since our model, by default settings, would load the specified dataset via an absolute path instead of a relative one.

### Demonstration

> `AnnEconForecast/demo`

This folder holds source codes, visualizations and saved model files for demo purposes.

### Configurations for HPT

>  `AnnEconForecast/hpt_configs`

Configurations for hyper-parameter tunning, please refer to the in-text comments in `AnnEconForecast/hpt_configs/template.py` for more information.

### Documentations

> `AnnEconForecast/docs`

Here is the documentation.

### Training? Restoring? HPT? Run the <u>Notebook</u>

> `AnnEconForecast/notebooks`

We used Jupyter notebooks to create an interactive environment for training, restoring and visualizing models in this project, each notebook contains detailed comments on how to use them.

---- 
## Earlier models

> ANNEF is a long-term project, but this repository is a new one. We have merged codes for earlier models into this repository and put them under this category.
>
> Currently, they are not under active-development, but we might restart some of them in the future.

- `/keras_based`  models built on `keras`
- `/matlab_based`  models built on `MatLab` 

