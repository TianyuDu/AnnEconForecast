"""
Sample hyper-parameter searching configuration file
"""
# Name
EXPERIMENT_NAME = "ec2-stacked-DEXCAUS"

# Data Pre-processing configuration
dp_config = {
    "PERIODS": 1,
    "ORDER": 1,
    "LAGS": 12,
    "TRAIN_RATIO": 0.8
}

# File configuration
file_config = {
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "TENSORBOARD_DIR": f"../tensorboard/{EXPERIMENT_NAME}",
    "MODEL_PATH": f"/home/ec2-user/saved_models/{EXPERIMENT_NAME}"
}

# Model training parameters
train_param = {
    "epochs": 300,
    "num_time_steps": [12, 24],
    "num_inputs": 1,
    "num_outputs": 1,
    "num_neurons": [
        (256, 128, 64),
        (512, 256, 128),
        (1024, 512, 256)
    ],
    "learning_rate": [
        0.3,
        0.1,
        0.03
    ],
    "report_periods": 10,
    "tensorboard_dir": "/home/ec2-user/AnnEconForecast/test_dir/tb/",
    "model_path": "/home/ec2-user/AnnEconForecast/test_dir/saved_models/",
    "fig_path": "/home/ec2-user/AnnEconForecast/test_dir/model_figs/"
}
