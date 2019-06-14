import json
import os
from typing import List


class ProfileLoader():
    def __init__(self, path: str) -> None:
        self.path = path
    
    def get_next_profile(self) -> dict:
        pass
        
    def get_all(self) -> List[dict]:
        pass
