"""
Convert numpy data type to basic python data type.
"""
import numpy as np

def clean(d) -> dict:
    new = dict()

    def clean_item(item) -> object:
        # Use this to detect something.
        # if isinstance(x, tuple):
        #     raise Exception
        if not isinstance(item, list):
            try:
                return np.asscalar(item)
            except AttributeError:
                return item
        return [
            clean_item(x)
            for x in item
        ]

    for (key, val) in d.items():
        if isinstance(key, str):
            new[key] = val
            continue
        new[key] = clean_item(val)

    return new
        
