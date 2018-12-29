"""
The baseline object for the general recurrent neural network
object in the OOP version of this project.
"""
from typing import Dict, Iterable, List, Union

import numpy as np


class GenericRNN:
    """
    The generic recurrent network.
    This is the abstract class used as the base of any recurrent 
    neural network.
    """
    def __init__(
        self,
        param: Dict[str, object],
        prediction_checkpoints: Iterable[int] = [-1],
        verbose: bool = True
    ) -> None:
        assert isinstance(param, dict),\
        "Parameter set should be a dict."

        assert all(isinstance(key, str) for key in param.keys()),\
        "All keys in parameter set should be string."

        assert all(isinstance(x, int) for x in prediction_checkpoints),\
        "Invalid recording checkpoint, all elements should be integers."

        assert all(-1 <= x <= param["epochs"] for x in prediction_checkpoints),\
        "Some element(s) in checkpoint are out of range."

        # Admit arguments.
        self.param = param
        self.param["num_time_steps"] = param["LAGS"]
        self.ckpts = prediction_checkpoints
    
    def build(self) -> None:
        """
        Build up the neural network.
        """
        raise NotImplementedError()

    def train(
        self,
        data: Dict[str, np.ndarray],
        evaluate: Union[None, str, List[str]] = None
    ) -> Union[None, Dict[str, float]]:
        """
        Train the neural network specified in this object.
        """
        raise NotImplementedError()
    
    def exec_core(
        self,
        param: Dict[str, object],
        data: Dict[str, np.ndarray],
        prediction_checkpoints: Iterable[int] = [-1],
        verbose: bool = True
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        The non-OOP fashion exec_core method inherited from the 
        eariler version.
        """
        raise NotImplementedError()
