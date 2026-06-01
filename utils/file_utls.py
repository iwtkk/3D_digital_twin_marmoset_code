import json
import pickle
from typing import TypedDict

class SampleData(TypedDict):
    user_id: int
    score: float
    name: str

def open_pickle(file):
    with open(file, 'rb') as f:
        data: SampleData = pickle.load(f)
    return data

def open_json(file):
    with open(file, 'rb') as f:
        data: SampleData = json.load(f)
    return data