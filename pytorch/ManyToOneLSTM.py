"""
Created: May 19 2019
"""

from typing import Tuple

import numpy as np
import torch


class ManyToOneLSTM(torch.nn.Module):
    def __init__(
        self,
        neurons: Tuple[int]=(32, 64),
        num_inputs: int = 1,  # Dimension of feature series
        num_outputs: int = 1  # Dimension of target series
        ) -> None:
        super().__init__()
        self.lstm_neurons = neurons
        # LSTM Cells.
        # A stacked/multi-layer LSTM model.
        # Number of features in input seq, 1 if univariate time series.
        # num_inputs = 1
        # Number of features in output seq, 1 if univariate time series.
        # num_outputs = 1
        self.lstm1 = torch.nn.LSTMCell(
            input_size=num_inputs,
            hidden_size=self.lstm_neurons[0]
        )
        self.lstm2 = torch.nn.LSTMCell(
            input_size=self.lstm_neurons[0],
            hidden_size=self.lstm_neurons[1]
        )
        # Output Linear Model.
        self.linear = torch.nn.Linear(
            in_features=self.lstm_neurons[1],
            out_features=num_outputs
        )

    def forward(self, inputs, future: int=0):
        """
        inputs: tensor@(NSamples, SequenceLength) univariate time series.
        """
        outputs = list() 
        # Output sequence, with the same order as input sequence.
        # LSTM Hidden and Cell States.
        # Boardcasting fashion @(N, hidden)
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

        for input_t in inputs.chunk(inputs.size(1), dim=1):
            # Expand input tensor by time step.
            # To a SequenceLength tuple with each element@(NSamples, 1)
            # Corresponding to the i-th observation in all samples.
            # (inputs, (h, c)) -> (h, c) updated.
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            out = self.linear(h_t2)
            # return h_t2
            return out
            outputs.append(output)
        # h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        # h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        # return h_t2
        # out = self.linear(h_t2[-1])
        # For regression problem, the output 
        outputs.append(out)
        # Current: outputs @ (N, 1) * SeqLen
        # if predict the future.
        for _ in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # Current: outputs @ (N, 1) * SeqLen+F
        # Add SeqLen+F to dim=1 -> (N, SeqLen+F, 1)
        # Drop dim=2 -> (N, SeqLen+F)
        outputs = torch.stack(outputs, dim=1).squeeze(dim=2)
        return outputs
