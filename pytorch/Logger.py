"""
May 19 2019
"""

class logger:
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
