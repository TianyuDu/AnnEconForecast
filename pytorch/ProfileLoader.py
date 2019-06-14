import json
import os
from typing import List, Union
import warnings


class ProfileLoader():
    def __init__(self, path: str, verbose: bool=False) -> None:
        assert os.path.exists(path), "Path does not exist."
        if not path.endswith("/"):
            path += "/"
        self.path = path
        self.verbose = verbose
        self.all_profiles = [
            f for f in os.listdir(self.path)
            if f.endswith(".json")
        ]
        assert len(self.all_profiles), "No profile found."
        self.c = 0
    
    def get_next(self) -> Union[dict, None]:
        try:
            with open(self.path + self.all_profiles[self.c], "r") as f:
                data = json.load(f)
            self.c += 1
            return data
        except IndexError:
            warnings.warn("All profile has been returned, call ProfileLoader.reset()")
            return None
        
    def get_all(self) -> List[dict]:
        total = list()
        for name in self.all_profiles:
            with open(self.path + name, "r") as f:
                d = json.load(f)
            total.append(d)
        return total
    
    def reset(self) -> None:
        self.c = 0
