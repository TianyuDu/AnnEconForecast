"""
Sample hyper-parameter searching configuration file
"""
# Name
EXPERIMENT_NAME = "mac_train"

# MAIN_DIRECTORY = "/Volumes/Intel/annef_model_data/2018DEC17_MAC_01"
MAIN_DIRECTORY = "/Users/tianyudu/Desktop/sample_model"
main = {
    # ======== Data Pre-processing Parameter ========
    "PERIODS": 1,
    "ORDER": 1,
    "LAGS": 12,
    "TRAIN_RATIO": 0.8,
    "VAL_RATIO": 0.1,
    # ======== Model Training Parameter ========
    "epochs": 150,
    "num_time_steps": 12,
    "num_inputs": 1,
    "num_outputs": 1,
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
    "tensorboard_dir": MAIN_DIRECTORY + "/tensorboard/",
    "model_path": MAIN_DIRECTORY + "/saved_models/",
    "fig_path": MAIN_DIRECTORY + "/model_figs/"
}

assert main["LAGS"] == main["num_time_steps"], "Lags and num_time_steps parameters should have equal length."
