from numpy import float32
import pandas as pd
from sklearn import preprocessing
from concurrent import futures
from multiprocessing import cpu_count
import os
import numpy as np

from helpers.log import Log


class Dataset:
    def __init__(self, path) -> None:
        self.log = Log()
        self.path = path

    def needExclude(self, column):
        for value in column:
            if (not type(value) == int) and (not type(value) == float):
                return True
            if not value == 0:
                return False
        return True

    def __cleanData(self, df: pd.DataFrame):
        futureObjs = []
        columnsToExclude = ['TimeStamp']
        with futures.ProcessPoolExecutor(cpu_count()) as executor:
            for column in df.columns:
                future = executor.submit(self.needExclude, df[column])
                futureObjs.append({'column': column, 'future': future})
            for future in futureObjs:
                if future['future'].result():
                    columnsToExclude.append(future['column'])
        self.log.debug('Removed ' + str(len(columnsToExclude)) +
                       ' of ' + str(len(df.columns)) + ' columns from  ' + self.path)
        df = df.drop(columnsToExclude, axis=1)
        return df

    def __removeNaNandINF(self, df: pd.DataFrame):
        df.replace([np.inf], np.finfo(float32).max, inplace=True)
        df.replace([-np.inf], -np.finfo(float32).max, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def __normalize(self, df):
        scaler = preprocessing.MaxAbsScaler().fit(df)
        result = pd.DataFrame(scaler.transform(
            df), columns=df.columns, dtype=float32)
        return result

    def __processData(self, df):
        self.log.debug('Processing dataset ' + self.path)
        df = self.__cleanData(df)
        df = df.astype(float32)
        self.log.debug('Finish processing ' + self.path)
        return df

    def getCachePath(self):
        return './cache/' + self.path

    def checkIfHasCache(self):
        return os.path.exists(self.getCachePath())

    def writeCache(self, df: pd.DataFrame):
        cachePath = self.getCachePath()
        dirPath = cachePath.split('/')
        dirPath.pop()
        dirPath = '/'.join(dirPath)
        os.makedirs(dirPath, exist_ok=True)
        df.to_csv(cachePath)

    def read(self, normalize=True):
        if self.checkIfHasCache():
            self.log.debug('Loading dataset from cache ' + self.path)
            data = pd.read_csv(self.getCachePath())
            data = data.iloc[:, 1:]
            data = data.astype(float32)
            data = self.__removeNaNandINF(data)
            self.log.debug('Finish dataset from cache ' + self.path)
            return data

        self.log.debug('Loading dataset ' + self.path)
        data = pd.read_csv(self.path)
        data = self.__processData(data)
        if normalize:
            data = self.__normalize(data)

        self.writeCache(data)
        data = self.__removeNaNandINF(data)
        self.log.debug('Finish loading ' + self.path)
        return data
