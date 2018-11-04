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

from constants import UNRATE_DIR


def load_dataset(
    dir: str,
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
    print(f"Dataset loaded.\
    \n\tIndex type: {str(df.index.dtype)}\
    \n\tData type: {str(df.values.dtype)}")
    return df
