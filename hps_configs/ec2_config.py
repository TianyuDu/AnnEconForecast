"""
Default hyper parameter searching configuration working on EC2.
"""
# Name of configuration
EXPERIMENT_NAME = "ec2_default"

# Model training parameters
MAIN_DIRECTORY = "/home/ec2-user/ec2_hps/2018DEC17_EC2_01"
main = {
    # ======== Data Pre-processing Parameter ========
    "PERIODS": 1,
    "ORDER": 1,
    "LAGS": [6, 12, 18],
    "TRAIN_RATIO": 0.8,
    "VAL_RATIO": 0.1,
    # ======== Model Training Parameter ========
    "epochs": [150, 300, 500],
    "num_inputs": 1,
    "num_outputs": 1,
    "num_time_steps": None,  # num_time_steps is identical to LAGS
    "num_neurons": [
        (128, 512),
        (64, 128, 256),
        (128, 256, 215)
    ],
    "learning_rate": [
        0.1,
        0.03
    ],
    "clip_grad": None,
    "report_periods": 10,
    "tensorboard_dir": MAIN_DIRECTORY + "/tensorboard/",
    "model_path": MAIN_DIRECTORY + "/saved_models/",
    "fig_path": MAIN_DIRECTORY + "/model_figs/"
}