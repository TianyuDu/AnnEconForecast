import datetime
import os

import keras
import numpy as np
import pandas as pd

from base_model import BaseModel
from multivariate_container import MultivariateContainer
from typing import Union


class MultivariateLSTM(BaseModel):
    def __init__(
            self,
            container: MultivariateContainer,
            config: bool=None,
            create_empty: bool=False) -> None:
        """
        Initialization method.
        """
        _, self.time_steps, self.num_fea = container.train_X.shape
        print(f"MultivariateLSTM Initialized: \
        \n\tTime Step: {self.time_steps}\
        \n\tFeature: {self.num_fea}")

        self.config = config
        self.container = container
        self.hist = None

        if create_empty:
            self.core = None
        else:
            self.core = self._construct_lstm_model(self.config)

        self._gen_file_name()
        print(
            f"\tMultivariateLSTM: Current model will be save to ./saved_models/f{self.file_name}/")

    def _construct_lstm_model(
            self,
            config: dict,
            verbose: bool=True
    ) -> keras.Model:
        """
        Construct the Stacked lstm model, 
        Note: Modify this method to change model configurations.
        # TODO: Add arbitray layer support. 
        """
        print("MultivariateLSTM: Generating LSTM model using Model API.")

        input_sequence = keras.layers.Input(
            shape=(self.time_steps, self.num_fea),
            dtype="float32",
            name="input_sequence")

        normalization = keras.layers.BatchNormalization()(input_sequence)

        lstm = keras.layers.LSTM(
            units=config["nn.lstm1"],
            return_sequences=False
        )(normalization)

        dense1 = keras.layers.Dense(
            units=config["nn.dense1"],
            name="Dense1"
        )(lstm)

        predictions = keras.layers.Dense(
            1,
            name="Prediction"
        )(dense1)

        model = keras.Model(inputs=input_sequence, outputs=predictions)

        model.compile(loss="mse", optimizer="adam")

        if verbose:
            print("\tMultivariateLSTM: LSTM model constructed with configuration: ")
            keras.utils.print_summary(model)
        return model

    def _construct_lstm_sequential(
            self,
            config: dict,
            verbose: bool=True
    ) -> keras.Sequential:
        """
        Construct the Stacked lstm model, 
        Note: Modify this method to change model configurations.
        # TODO: Add arbitray layer support. 
        """
        print("MultivariateLSTM: Generating LSTM model with Keras Sequential API")
        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=config["nn.lstm1"],
            input_shape=(self.time_steps, self.num_fea),
            return_sequences=True,
            name="LSTM1"
        ))
        model.add(
            keras.layers.LSTM(
                units=config["nn.lstm2"],
                name="LSTM2"
            ))
        model.add(
            keras.layers.Dense(
                units=config["nn.dense1"],
                name="Dense1"
            ))
        model.add(
            keras.layers.Dense(
                units=1,
                name="Dense_output"
            ))
        model.compile(loss="mse", optimizer="adam")

        if verbose:
            print("\tMultivariateLSTM: LSTM model constructed with configuration: ")
            keras.utils.print_summary(model)
        return model

    def update_config(
            self,
            new_config: dict
    ) -> None:
        """
        Update the neural network configuration, and re-construct, re-compile the core.
        """
        # TODO: add check configuration method here.
        print("MultivariateLSTM: Updating neural network configuration...")
        self.prev_config = self.config
        self.config = new_config
        self.core = self._construct_lstm_model(self.config, verbose=False)
        print("\tDone.")

    def fit_model(
            self,
            epochs: int=10
    ) -> None:
        start_time = datetime.datetime.now()
        print("MultivariateLSTM: Start fitting.")
        self.hist = self.core.fit(
            self.container.train_X,
            self.container.train_y,
            epochs=epochs,
            batch_size=32 if self.config is None else self.config["batch_size"],
            validation_split=0.1 if self.config is None else self.config["validation_split"]
        )
        finish_time = datetime.datetime.now()
        time_taken = finish_time - start_time
        print(f"\tFitting finished, {epochs} epochs for {str(time_taken)}")

    def predict(
            self,
            X_feed: np.ndarray
    ) -> np.ndarray:
        y_hat = self.core.predict(X_feed, verbose=1)
        # y_hat = self.container.scaler_y.inverse_transform(y_hat)
        # y_hat returned used to compare with self.container.*_X directly.
        return y_hat

    def save_model(
            self, 
            file_dir: str=None
    ) -> None:
        if file_dir is None:
            # If no file directory specified, use the default one.
            file_dir = self.file_name

        # Try to create record folder.
        try:
            folder = f"./saved_models/{file_dir}/"
            os.system(f"mkdir {folder}")
            print(f"Experiment record directory created: {folder}")
        except:
            print("Current directory: ")
            _ = os.system("pwd")
            raise FileNotFoundError(
                "Failed to create directory, please create directory ./saved_models/")

        # Save model structure to JSON
        print("Saving model structure...")
        model_json = self.core.to_json()
        with open(f"{folder}model_structure.json", "w") as json_file:
            json_file.write(model_json)
        print("Done.")

        # Save model weight to h5
        print("Saving model weights...")
        self.core.save_weights(f"{folder}model_weights.h5")
        print("Done")

        # Save model illustration to png file.
        print("Saving model visualization...")
        try:
            keras.utils.plot_model(
                self.core,
                to_file=f"{folder}model.png",
                show_shapes=True,
                show_layer_names=True)
        except:
            print("Model illustration cannot be saved.")
        
        # Save training history (if any)
        if self.hist is not None:
            hist_loss = np.squeeze(np.array(self.hist.history["loss"]))
            hist_val_loss = np.squeeze(np.array(self.hist.history["val_loss"]))
            combined = np.stack([hist_loss, hist_val_loss])
            combined = np.transpose(combined)
            df = pd.DataFrame(combined, dtype=np.float32)
            df.columns = ["loss", "val_loss"]
            df.to_csv(f"{folder}hist.csv", sep=",")
            print(f"Training history is saved to {folder}hist.csv...")
            
        else:
            print("No training history found.")

        print("Done.")

    def load_model(
            self, 
            folder_dir: str
    ) -> None:
        """
        #TODO: doc
        """
        if not folder_dir.endswith("/"):
            # Assert the correct format, folder_dir should be
            folder_dir += "/"

        print(f"Load model from folder {folder_dir}")

        # construct model from json
        print("Reconstruct model from Json file...")
        try:
            json_file = open(f"{folder_dir}model_structure.json", "r")
        except FileNotFoundError:
            raise Warning(
                f"Json file not found. Expected: {folder_dir}model_structure.json"
            )

        model_file = json_file.read()
        json_file.close()
        self.core = keras.models.model_from_json(model_file)
        print("Done.")

        # load weights from h5
        print("Loading model weights...")
        try:
            self.core.load_weights(
                f"{folder_dir}model_weights.h5", by_name=True)
        except FileNotFoundError:
            raise Warning(
                f"h5 file not found. Expected: {folder_dir}model_weights.h5"
            )
        print("Done.")
        self.core.compile(loss="mse", optimizer="adam")

    def summarize_training(self):
        """
        Summarize training result to string file.
        - Loss
        - Epochs
        - Time taken
        """
        raise NotImplementedError

    def visualize_training(self):
        """
        Visualize the training result:
        - Plot training set loss and validation set loss.
        """
        # TODO: move visualize training to general methods.
        raise NotImplementedError
