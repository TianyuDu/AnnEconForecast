"""
Sample hyper-parameter searching configuration file
"""
# Name
EXPERIMENT_NAME = "mac_train"

# Data Pre-processing configuration
dp_config = {
    "PERIODS": 1,
    "ORDER": 1,
    "LAGS": 12,
    "TRAIN_RATIO": 0.8,
    "VAL_RATIO": 0.1
}

# File configuration
file_config = {
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "TENSORBOARD_DIR": f"../tensorboard/{EXPERIMENT_NAME}",
    "MODEL_PATH": f"/home/ec2-user/saved_models/{EXPERIMENT_NAME}"
}

# Model training parameters
MAIN_DIRECTORY = "/home/ec2-user/test_hps_Nov29_1/"
train_param = {
    "epochs": 50,
    "num_time_steps": [12, 24],
    "num_inputs": 1,
    "num_outputs": 1,
    "num_neurons": [
        (512, 256),
        (1024, 512)
    ],
    "learning_rate": [
        0.1,
        0.03
    ],
    "clip_grad": None,
    "report_periods": 10,
    "tensorboard_dir": MAIN_DIRECTORY + "tensorboard/",
    "model_path": MAIN_DIRECTORY + "saved_models/",
    "fig_path": MAIN_DIRECTORY + "model_figs/"
}
