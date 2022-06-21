import code
from concurrent import futures
import sys
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from numpy import sort
import pandas as pd
from preprocessing import remove_outliers, processing_features
from loaders.config import Loader as ConfigLoader
from loaders.dataset import Dataset as DatasetLoader
from helpers.log import Log
from model import Model
from helpers.data import split_data
from online_emulator import OnlineEmulator
from selection_algorithm.reservoir import Reservoir


def load_data_sets(configs):
    executor = futures.ProcessPoolExecutor(12)
    x_promisses = []
    y_promisses = []

    x_datasets = []
    y_dataset = None

    target = ''

    for dataset in configs['datasets']:
        for xs in dataset['x']:
            d_loader = DatasetLoader(xs)
            task = executor.submit(d_loader.read)
            x_promisses.append(task)

        d_loader = DatasetLoader(dataset['y'])
        task = executor.submit(d_loader.read)
        y_promisses.append(task)
        target = dataset['target']

    for promise in x_promisses:
        x_datasets.append(promise.result())

    for promise in y_promisses:
        y_dataset = promise.result()[target]

    return {
        'x': x_datasets,
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
        kind='scatter', x='index', y='value', s=1)
    pd.DataFrame(predicted_data_ordered).plot(
        kind='scatter', x='index', y='value',  s=1)
    plt.show()


if __name__ == '__main__':

    freeze_support()

    log = Log()

    log.debug('Script start')

    config_file = 'config.json'

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    loader = ConfigLoader(config_file)

    configs = loader.read()

    log.debug('Configs loaded')

    datasets = load_data_sets(configs)

    x = pd.concat(datasets['x'], axis=1)
    y = datasets['y']

    x = processing_features(x, y, 16)

    #x, y = remove_outliers(x, y)

    data = {
        'x': x,
        'y': y
    }

    train_data, test_data = split_data(data)

    log.debug('Start Train')
    #model = Model()
    # model.train(train_data)

    log.debug('Start Predict')
    #predicted = model.predict(test_data['x'])

    log.debug('End Predict')

    # plot_predicted_data(
    # test_data['y'].values,
    #    predicted
    # )

    log.debug('Done')

    model = Model()
    reservoir = Reservoir()
    emulator = OnlineEmulator(model, reservoir, data)
    emulator.start()

    accuracies = emulator.get_accuracies()
    accuracies = pd.DataFrame(accuracies)
    accuracies.plot(kind='line', x='train_number', y='r2')
    accuracies.plot(kind='line', x='train_number', y='mae')
    accuracies.plot(kind='line', x='train_number', y='nmae')

    y_data = emulator.test_data['y'].values
    predicted_data = emulator.model.predict(emulator.test_data['x'])

    plot_predicted_data(y_data, predicted_data)

    # code.interact(local=locals())
