"""
This package contains methods handling
reading raw data into data frames.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime

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
        parse_date=[0],
        date_parser=parser,
        index_col=0)
    print(f"Dataset loaded, with type {df.values.dtype}.")
    return df
