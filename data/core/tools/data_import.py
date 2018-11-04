"""
This package contains methods handling
reading raw data into data frames.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from constants import UNRATE_DIR

def load_dataset(
    dir: str,
) -> pd.DataFrame:
    df = pd.read_csv(dir, sep=",", header="infer", index_col=0)
    print(f"Dataset loaded, with type {df.values.dtype}.")
    return df
