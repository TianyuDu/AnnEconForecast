"""
Sample hyper-parameter searching configuration file
"""
# Name
EXPERIMENT_NAME = "mac_lite"

# MAIN_DIRECTORY = "/Volumes/Intel/annef_model_data/2018DEC17_MAC_01"
MAIN_DIRECTORY = "?"
main = {
    # ======== Data Pre-processing Parameter ========
    "PERIODS": 1,
    "ORDER": 1,
    "LAGS": [3, 6],
    "TRAIN_RATIO": 0.8,
    "VAL_RATIO": 0.1,
    # ======== Model Training Parameter ========
    "epochs": [3, 4, 5],
    "num_inputs": 1,
    "num_outputs": 1,
    "num_time_steps": None,  # num_time_steps is identical to LAGS
    "num_neurons": [
        (16, 32),
        (32, 64),
        (16, 16, 32),
        (16, 16, 32, 32)
    ],
    "learning_rate": [
        0.1,
        0.03,
        0.01
    ],
    "clip_grad": [None, 10.0, 20.0, 30.0, 50.0],
    "report_periods": 10,
    "tensorboard_path": MAIN_DIRECTORY + "/tensorboard/",
    "model_path": MAIN_DIRECTORY + "/saved_models/",
    "fig_path": MAIN_DIRECTORY + "/model_figs/"
}
