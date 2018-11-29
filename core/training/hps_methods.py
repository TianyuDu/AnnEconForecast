"""
This script contains helper functions implemented
for hyper-parameter searching / grid searching
"""
import itertools
import copy
from typing import List, Dict, Union


def gen_hparam_set(
    src_dic: Dict[str, Union[List[object], object]]
) -> List[str, object]:
    """
    TODO: Stopped here.
    """
    gen = list()
    detected_list_keys = list()
    detected_list_vals = list()

    for k, v in src_dic.items():
        if isinstance(v, list):
            detected_list_keys.append(k)
            detected_list_vals.append(v)

    cartesian_prod = list(itertools.product(*detected_list_vals))

    for coor in cartesian_prod:
        new_para = copy.deepcopy(src_dic)
        hparam_str = "-".join(
            f"{k}={v}" for k, v in zip(detected_list_keys, coor))
        for i, key in enumerate(detected_list_keys):
            new_para[key] = coor[i]
        new_para["tensorboard_dir"] += hparam_str
        new_para["model_path"] += hparam_str
        gen.append(new_para)

    print(f"Total number of parameter sets generated: {len(gen)}")
    return gen
