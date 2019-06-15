"""
Created: May 19 2019
Different types of LSTM models
"""

from typing import Tuple

import numpy as np
import torch


class StackedLSTM(torch.nn.Module):
    """
    The baseline LSTM module. The possible variations are made in the forward phase.
    """
    def __init__(
        self,
        neurons: Tuple[int]=(32, 64),  # 2-layer stacked deep LSTM
        num_inputs: int=1,  # Dimension of feature series
        num_outputs: int=1  # Dimension of target series
    ) -> None:
        # LSTM Cells.
        # A stacked/multi-layer LSTM model.
        # Number of features in input seq, 1 if univariate time series.
        # Number of features in output seq, 1 if univariate time series.
        super().__init__()
        self.lstm_neurons = neurons
        self.lstm1 = torch.nn.LSTMCell(
            input_size=num_inputs,
            hidden_size=self.lstm_neurons[0]
        )
        self.lstm2 = torch.nn.LSTMCell(
            input_size=self.lstm_neurons[0],
            hidden_size=self.lstm_neurons[1]
        )

        # Linear layer: output layer
        self.linear = torch.nn.Linear(
            in_features=self.lstm_neurons[1],
            out_features=num_outputs
        )

class PoolingLSTM(StackedLSTM, torch.nn.Module):
    """
    The LSTM with pooling layers as the final output layer.
    ================================================
    INPUT: (x[t-L], ..., x[t-1]),
    TARGET: x[t],
    RNN OUTPUT: (x-hat[t-L+1], ..., x-hat[t]),
    PREDICTED: W * (x-hat[t-L+1], ..., x-hat[t]) + b -> 1dim
    ================================================
    """
    def __init__(
        self,
        lags: int,
        neurons: Tuple[int],
        num_inputs: int=1,  # Dimension of feature series
        num_outputs: int=1,  # Dimension of target series
        ) -> None:
        super().__init__(neurons=neurons, num_inputs=num_inputs, num_outputs=num_outputs)
        # Output Pooling Overtime Layer, for PoolingLSTM only.
        self.pooling = torch.nn.Linear(
            in_features=lags,
            out_features=num_outputs
        )

    def forward(self, inputs):
        """
        inputs: tensor@(NSamples, SequenceLength) univariate time series.
        """
        # Initialize hidden and new cell states.
        # LSTM Layer 1
        h_t = torch.randn(inputs.size(0), self.lstm_neurons[0])
        c_t = torch.randn(inputs.size(0), self.lstm_neurons[0])
        # LSTM Layer 2
        h_t2 = torch.randn(inputs.size(0), self.lstm_neurons[1])
        c_t2 = torch.randn(inputs.size(0), self.lstm_neurons[1])
        
        out_seq = list()
        for i, input_t in enumerate(inputs.chunk(inputs.size(1), dim=1)):
            # ==========================================================
            # Expand input tensor by time step.
            # To a SequenceLength tuple with each element@(batchsize, 1)
            # Corresponding to the i-th observation in all samples.
            # (inputs, (h, c)) -> (h, c) updated.
            # ==========================================================
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            out = self.linear(h_t2)  # (batchsize, 1) single time step output
            print(f"Time step {i}: {out.shape}")
            out_seq.append(out)
        # out_seq @ (batchsize, lags)
        # assert out_seq.shape == (inputs.size(1), self.lags)
        # Single step forecasting, using the pooling layer.
        out_seq = torch.stack(out_seq, dim=1).squeeze()
        assert out_seq.shape == inputs.shape
        # For each training instance:
        # INPUT: (x[t-L], ..., x[t-1])
        # OUT_SEQ: (x-hat[t-L+1], x-hat[t-L+2], ..., x-hat[t])
        # Then pool over time to construct the final prediction.
        predict = self.pooling(out_seq)
        return predict

class LastOutLSTM(StackedLSTM, torch.nn.Module):
    """
    The last output LSTM with single step forecast.
    ================================================
    INPUT_FEATURE: (x[t-L], ..., x[t-1]),
    TARGET: x[t]
    RNN OUTPUT: (x-hat[t-L+1], ..., x-hat[t]),
    PREDICTED: x-hat[t]
    ================================================
    In contrast to the PoolingLSTM, the last-out-LSTM does not require a pre-defined "numlag" parameter.
    """
    def __init__(
        self,
        neurons: Tuple[int],
        num_inputs: int=1,
        num_outputs: int=1
        ) -> None:
        super().__init__(neurons=neurons, num_inputs=num_inputs, num_outputs=num_outputs)

    def forward(self, inputs):
        """
        inputs: tensor with shale (batchsize, num_lags) univariate time series.
        """
        # Initialize hidden and new cell states.
        # LSTM Layer 1
        h_t = torch.randn(inputs.size(0), self.lstm_neurons[0])
        c_t = torch.randn(inputs.size(0), self.lstm_neurons[0])
        # LSTM Layer 2
        h_t2 = torch.randn(inputs.size(0), self.lstm_neurons[1])
        c_t2 = torch.randn(inputs.size(0), self.lstm_neurons[1])

        out_seq = list()
        for (i, input_t) in enumerate(inputs.chunk(inputs.size(1), dim=1)):
            # Expand input tensor by time step.
            # To a SequenceLength tuple with each element@(batchsize, 1)
            # Corresponding to the i-th observation in all samples.
            # (inputs, (h, c)) -> (h, c) updated.
            
            # ==== To correct input type ==== 
            # TODO: remove this after debugging on server.
            # if input_t.dtype == torch.float64:
            #     print("Input Type detected: ", input_t.dtype)
            #     input_t = input_t.float()
            #     print(f"order: {i}/{len(inputs.chunk(inputs.size(1), dim=1))}")
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            out = self.linear(h_t2)  # (batchsize, 1) single time step output
            out_seq.append(out)
        # Take the last element as the forecasting
        predict = out_seq[-1]
        return predict
