from math import sqrt
from multiprocessing import cpu_count
import pandas
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np


def processing_features(x: pandas.DataFrame, y: pandas.DataFrame, k):
    selector = SelectKBest(f_regression, k=k)
    selector.fit(x, y)
    columns = selector.get_support(indices=True)
    return x.iloc[:, columns]


def remove_outliers(x: pandas.DataFrame, y: pandas.DataFrame):
    lines_to_remove = []
    for column in x.columns:
        std = np.std(x[column])
        mean = np.mean(x[column])
        for line in range(0, len(x[column])):
            distance_from_mean = sqrt((x[column][line] - mean) ** 2)
            if distance_from_mean > std * 10:
                lines_to_remove.append(line)

    lines_to_remove = np.unique(lines_to_remove)

    print(len(lines_to_remove))
    print(len(x))

    x.drop(lines_to_remove, axis=0, inplace=True)
    y.drop(lines_to_remove, axis=0, inplace=True)

    print(len(x))

    print("Droped " + str(len(lines_to_remove)) + " outliers")

    return x, y
