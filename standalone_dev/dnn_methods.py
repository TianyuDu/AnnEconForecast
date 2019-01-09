import numpy as np
import 


# Instance data type, a single training example.
Instance = Tuple[np.ndarray, np.ndarray, pd.Timestamp]

def gen_seq_slp(
    df: pd.DataFrame,
    num_time_steps: int,
    label_col: str = None
) -> List[Instance]:
    """
    Generate the supervised learning problem with
    sequence-valued label in each instance.
    Sliding Window Method
    data.shape = (num_obs, 1)
    """
    X_set = df.copy()
    if label_col is None:
        # The next-observed values of ALL features are interpreted as label.
        y_set = df.copy()
    else:
        y_set = df[label_col].copy()
        
    instances = list()
    for t in range(len(X_set)):
        try:
            feature = X_set.iloc[t: t+num_time_steps, :]
            label = y_set.iloc[t+1: t+num_time_steps+1, :]
            assert len(feature) == len(label)
            instances.append((
                feature.values,
                label.values,
                label.index[-1]
            ))
        except AssertionError:
            print(f"Failed time step ignored: {t}")

    return instances


def gen_pt_slp(diff, num_time_steps):
    """
    Generate the supervised learning problem with
    point-valued label in each instance.
    """
    slp_sequential = gen_slp_sequential(
        diff,
        num_time_steps=num_time_steps
    )
    instances = [
        (x, y[-1], t)
        for x, y, t in slp_sequential
    ]
    return instances













