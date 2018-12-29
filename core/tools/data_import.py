"""
This package contains methods handling
reading raw data into data frames.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
from datetime import datetime

from constants import *


def load_dataset(
    dir: str,
    remove: object = None,
    verbose: bool = False
) -> pd.DataFrame:
    def parser(x):
	    return datetime.strptime(x, "%Y-%m-%d")

    df = pd.read_csv(
        dir,
        sep=",",
        header="infer",
        parse_dates=[0],
        date_parser=parser,
        index_col=0,
        engine="python")
    
    if verbose:
        print(f"Dataset loaded.\
        \n\tIndex type: {str(df.index.dtype)}\
        \n\tData type: {str(df.values.dtype)}")
    col_name = df.columns[0]
    if remove is not None:
        df = df[df[col_name] != remove]
    df = df.astype(np.float32)
    return df
