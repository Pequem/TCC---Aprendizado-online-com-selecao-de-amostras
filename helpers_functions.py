from concurrent import futures
import matplotlib.pyplot as plt
import pandas as pd
from loaders.dataset import Dataset as DatasetLoader
from sklearn.model_selection import train_test_split


def split_data(data, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(
        data['x'], data['y'], test_size=test_size)
    train_data = {
        'x': x_train,
        'y': y_train
    }
    test_data = {
        'x': x_test,
        'y': y_test
    }
    return train_data, test_data


def load_data_set(dataset_config):
    x_promisses = []
    y_promisses = []

    x_dataset = []
    y_dataset = None

    target = ''

    with futures.ProcessPoolExecutor() as executor:
        # load X datas
        for xs in dataset_config['x']:
            d_loader = DatasetLoader(xs)
            task = executor.submit(d_loader.read)
            x_promisses.append(task)

        # load Y data
        d_loader = DatasetLoader(dataset_config['y'])
        task = executor.submit(d_loader.read, False)
        y_promisses.append(task)
        target = dataset_config['target']

        for promise in x_promisses:
            x_dataset.append(promise.result())

        for promise in y_promisses:
            y_dataset = promise.result()[target]

    return {
        'x': x_dataset,
        'y': y_dataset
    }


def plot_predicted_data(y_data, predicted_data):
    y_data_with_index = []
    predicted_data_with_index = []

    # Set index in value
    for index in range(0, len(y_data)):
        y_data_with_index.append({'value': y_data[index], 'index': index})
        predicted_data_with_index.append(
            {'value': predicted_data[index], 'index': index})

    # Sort array by value
    y_data_with_index.sort(key=lambda data: data['value'])
    #predicted_data_with_index.sort(key=lambda data: data['value'])

    # Put predict array in same order that y array
    predicted_data_ordered = []
    for index in y_data_with_index:
        predicted_data_ordered.append(
            predicted_data_with_index[index['index']])

    # Reset index number to make then sequential
    for index in range(0, len(y_data_with_index)):
        predicted_data_ordered[index] = {
            'value': predicted_data_ordered[index]['value'], 'index': index}
        y_data_with_index[index] = {
            'value': y_data_with_index[index]['value'], 'index': index}

    # Plot
    pd.DataFrame(y_data_with_index).plot(
        kind='scatter', x='index', y='value', s=0.2)
    pd.DataFrame(predicted_data_ordered).plot(
        kind='scatter', x='index', y='value',  s=0.2)
    plt.show()
