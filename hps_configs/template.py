"""
Sample hyper-parameter searching configuration file
"""
# Name
EXPERIMENT_NAME = "template"

# MAIN_DIRECTORY = "/Volumes/Intel/annef_model_data/2018DEC17_MAC_01"
MAIN_DIRECTORY = "~"
main = {
    # ======== Data Pre-processing Parameter ========
    "PERIODS": 1,
    "ORDER": 1,
    "LAGS": [6, 12],
    "TRAIN_RATIO": 0.8,
    "VAL_RATIO": 0.1,
    # ======== Model Training Parameter ========
    "epochs": 150,
    "num_inputs": 1,
    "num_outputs": 1,
    "num_time_steps": None,  # num_time_steps is identical to LAGS
    "num_neurons": [
        (16, 32),
        (32, 64),
        (64, 64, 128)
    ],
    "learning_rate": [
        0.1
    ],
    "clip_grad": None,
    "report_periods": 10,
    "tensorboard_path": MAIN_DIRECTORY + "/tensorboard/",
    "model_path": MAIN_DIRECTORY + "/saved_models/",
    "fig_path": MAIN_DIRECTORY + "/model_figs/"
}
