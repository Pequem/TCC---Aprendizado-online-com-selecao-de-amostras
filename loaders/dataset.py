import pandas as pd

class Dataset:
    def __init__(self, path) -> None:
        self.path = path

    def read(self):
        data = pd.read_csv(self.path)
        return data