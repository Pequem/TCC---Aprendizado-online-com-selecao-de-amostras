import pandas as pd
from sklearn import preprocessing

from helpers.log import Log
class Dataset:
    def __init__(self, path) -> None:
        self.log = Log()
        self.path = path

    def __processData(self, df):
        self.log.degub('Processing dataset ' + self.path)
        scaler = preprocessing.MaxAbsScaler().fit(df)
        result = scaler.transform(df)
        self.log.degub('Finish processing ' + self.path)
        return result

    def read(self):
        self.log.degub('Loading dataset ' + self.path)
        data = pd.read_csv(self.path)
        self.log.degub('Finish loading ' + self.path)
        return self.__processData(data)