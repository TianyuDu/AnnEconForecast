### How to Run Models in this Project

All backbones of models are stored in `.py `  files under `/core`  directory.

To execute certain model, one can run Jupyter notebooks in `/notebook`  directory and specify parameters within the notebook.

Detailed explanation, instruction and result visualization for a certain neural net can all be found in the corresponding Jupyter notebook.

### Project Layout

#### Main Model

- `/core`  core files containing codes
- `/data` dataset directory
- `/notebooks`  Jupyter notebooks
- `/saved_models`  this is the default directory for TensorFlow to store models after training.
- `/tensorboard`  this is the default directory for TensorFlow to store tensor board visualization files.

#### Archived Models

- `/keras_based`  models built on `keras`
- `/matlab_based`  models built on `MatLab` 