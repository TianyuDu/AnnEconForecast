"""
May 19 2019
"""
import numpy as np
import pandas as pd


class Logger():
    """
    The logger object is used to store metrics on training and 
    validation set during the training process.
    """
    def __init__(self):
        self.t, self.v = [], []

    def __repr__(self):
        return f"Log with:\n\ttime={self.t}\n\tvalue={self.v}"

    def add(self, time, value):
        self.t.append(time)
        self.v.append(value)

    def clear(self):
        self.__init__()

    def get_array(self, ln: bool=False):
        if ln:
            f = lambda x: np.log(np.array(x))
            return self.t, f(self.v)
        else:
            return np.array(self.t), np.array(self.v)

    def get_df(self, ln: bool=False):
        if ln:
            ln_v = [np.log(x) for x in self.v]
            return pd.DataFrame(data=ln_v, index=self.t)
        else:
            return pd.DataFrame(data=self.v, index=self.t)

    def max(self) -> float:
        return np.max(self.v)
    
    def min(self) -> float:
        return np.min(self.v)

    def argmax(self) -> set:
        t_ary, v_ary = self.get_array()
        return set(t_ary[v_ary == self.max()])

    def argmin(self) -> set:
        t_ary, v_ary = self.get_array()
        return set(t_ary[v_ary == self.min()])


if __name__ == "__main__":
    l = TrainLogger()
    for t, v in enumerate(range(0, 1000, 100)):
        l.add(t, v)
    l.add(10, 0)
    l.add(11, 900)
