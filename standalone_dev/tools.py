import numpy as np
import pandas as pd


def progbar(curr, total, full_progbar):
    """
    Progress bar used in training process.
    Reference: https://geekyisawesome.blogspot.com/2016/07/python-console-progress-bar-using-b-and.html
    """
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
#     print('\r', '#'*filled_progbar + '-'*(
#         full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    print('\r', '#'*filled_progbar + '-'*(
        full_progbar-filled_progbar), f"[{curr}/{total}, {frac:>7.2%}]", end='')



def inv_diff(
    diff: pd.DataFrame,
    src: pd.DataFrame,
    order: int=1
) -> pd.DataFrame:
    col_name = src.columns[0]
    inv = pd.DataFrame([np.nan] * len(src))  # Inverted series
    inv.columns = src.columns
    inv.index = src.index

    for (idx, date) in enumerate(diff.index):
        # For each date in the differenced series.
        if idx - order < 0:
            inv[col_name][date] = np.nan
            continue
        
        base_date = diff.index[idx - order]
        base_val = src[col_name][base_date]
        inv[col_name][date] = base_val + diff[col_name][date]

    return inv


def gen_slp_seq(
    src: pd.DataFrame
):
    raise NotImplementedError()