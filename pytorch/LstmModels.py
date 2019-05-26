"""
Created: May 19 2019
"""

from typing import Tuple

import numpy as np
import torch

# Test the multiple inheritance with super() call.
# class First():
#     def __init__(self):
#         print("first")

# class Second():
#     def __init__(self):
#         print("second")

# class Third(First, Second):
#     def __init__(self):
#         super().__init__()

# c = Third()

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
    The pooling LSTM.
    """
    def __init__(
        self,
        lags: int,
        neurons: Tuple[int]=(32, 64),
        num_inputs: int = 1,  # Dimension of feature series
        num_outputs: int = 1  # Dimension of target series
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
        # LSTM Layer 1
        h_t = torch.randn(
            inputs.size(0),
            self.lstm_neurons[0],
            dtype=torch.double
        )
        
        c_t = torch.randn(
            inputs.size(0),
            self.lstm_neurons[0],
            dtype=torch.double
        )
        # LSTM Layer 2
        h_t2 = torch.randn(
            inputs.size(0),
            self.lstm_neurons[1],
            dtype=torch.double
        )
        c_t2 = torch.randn(
            inputs.size(0),
            self.lstm_neurons[1],
            dtype=torch.double
        )
        
        # print(f"Shapes:\n\th:{h_t.shape}, h2:{h_t2.shape}\n\tc:{c_t.shape}, c2:{c_t2.shape}")

        out_seq = list()
        for input_t in inputs.chunk(inputs.size(1), dim=1):
            # Expand input tensor by time step.
            # To a SequenceLength tuple with each element@(batchsize, 1)
            # Corresponding to the i-th observation in all samples.
            # (inputs, (h, c)) -> (h, c) updated.
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            out = self.linear(h_t2)  # (batchsize, 1) single time step output
            out_seq.append(out)
        # out_seq @ (batchsize, lags)
        # Single step forecasting, using the pooling layer
        # to map ALL y-hat in the sequence to a single output.
        out_seq = torch.stack(out_seq, dim=1).squeeze(dim=2)
        # Then pool over time to construct the final prediction.
        predict = self.pooling(out_seq)
        
        # print(f"predict shape: {predict.shape}")

        # TODO: clean up code here.
        # For regression problem, the output 
        # outputs.append(out)
        # Current: outputs @ (N, 1) * SeqLen
        # for _ in range(future):
        #     h_t, c_t = self.lstm1(output, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     out_seq.append(output)
        # Current: outputs @ (N, 1) * SeqLen+F
        # Add SeqLen+F to dim=1 -> (N, SeqLen+F, 1)
        # Drop dim=2 -> (N, SeqLen+F)
        # out_seq = torch.stack(out_seq, dim=1).squeeze(dim=2)
        return predict

class LastOutLSTM(StackedLSTM, torch.nn.Module):
    """
    The last output LSTM with single step forecast.
    Given a sequence taking N lagged values (x[t-L], ..., x[t-1]), 
    the last y-hat produced by RNN is taken to be the predicted value of x[t].
    Therefore, in contrast to the PoolingLSTM,
    the last out LSTM does not require a pre-defined "numlag" parameter.
    """
    def __init__(self):
        #TODO: STOPPED HERE