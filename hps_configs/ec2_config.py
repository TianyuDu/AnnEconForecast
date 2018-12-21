"""
Default hyper parameter searching configuration working on EC2.
"""
# Name of configuration
EXPERIMENT_NAME = "ec2_large"

# Model training parameters
MAIN_DIRECTORY = "/home/ec2-user/ec2_hps/2018DEC21_01"
main = {
    # ======== Data Pre-processing Parameter ========
    "PERIODS": 1,
    "ORDER": 1,
    "LAGS": [3, 6, 9, 12],
    "TRAIN_RATIO": 0.8,
    "VAL_RATIO": 0.1,
    # ======== Model Training Parameter ========
    "epochs": [150, 300, 500, 1000],
    "num_inputs": 1,
    "num_outputs": 1,
    "num_time_steps": None,  # num_time_steps is identical to LAGS
    "num_neurons": [
        (128, 256),
        (256, 512),
        (512, 1024),
        (128, 256, 512),
        (256, 512, 1024),
        (512, 1024, 2048)
    ],
    "learning_rate": [
        0.003,
        0.01,
        0.03
    ],
    "clip_grad": [None, 10, 50],
    "report_periods": 10,
    "tensorboard_path": MAIN_DIRECTORY + "/tensorboard/",
    "model_path": MAIN_DIRECTORY + "/saved_models/",
    "fig_path": MAIN_DIRECTORY + "/model_figs/"
}