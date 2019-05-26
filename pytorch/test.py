CPIAUCSUL_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/CPIAUCSL.csv"
SUNSPOT_DATA_E = "/home/ec2-user/environment/AnnEconForecast/data/sunspots.csv"
SUNSPOT_DATA = "/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast/data/sunspots.csv"

PROFILE = {
    "TRAIN_SIZE": 231,  # Include both training and validation sets.
    "TEST_SIZE": 58,
    "LAGS": 6,
    "VAL_RATIO": 0.2,  # Validation ratio.
    "LEARNING_RATE": 0.01,
    "NEURONS": (32, 128),
    "EPOCHS": 20,
    "LOG_NAME": "untitled",
    "TASK_NAME": "LastOut LSTM on sunspot",
    "DATA_DIR": SUNSPOT_DATA
}


import main_lstm
if __name__ == "__main__":
    # globals().update(PROFILE)
    main_lstm.core(**PROFILE, profile_record=PROFILE)

