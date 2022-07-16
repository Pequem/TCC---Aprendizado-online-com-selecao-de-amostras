import code
from distutils.log import debug
import os
import sys
from multiprocessing import freeze_support
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import processing_features
from loaders.config import Loader as ConfigLoader
from helpers.log import Log
from model import Model
from helpers_functions import split_data, load_data_set, plot_predicted_data
from online_emulator import OnlineEmulator
from selection_algorithm.boxes import Boxes
from selection_algorithm.reservoir import Reservoir


def plot_metrics(emulator, test_len, dataset_len, save_path):
    accuracies = emulator.get_accuracies()
    accuracies = pd.DataFrame(accuracies)

    for metric in ['r2', 'mae', 'nmae']:
        fig, ax = plt.subplots()
        title = metric.upper() + ' - ' + 'Dataset: ' + str(dataset_len) + \
            ' | Test: ' + str(test_len) + '| Mean: ' + \
            str(round(accuracies[metric].mean(), 4))
        accuracies.plot(kind='line', x='train_number',
                        y=metric, title=title, ax=ax)
        plt.savefig(save_path + metric + '_d_' +
                    str(dataset_len) + '_t_' + str(test_len) + '.png')
        plt.close(fig)


def plot_metrics2(emulator, test_len, dataset_len, dimension, division, distance, save_path):
    accuracies = emulator.get_accuracies()
    accuracies = pd.DataFrame(accuracies)

    for metric in ['r2', 'mae', 'nmae']:
        fig, ax = plt.subplots()
        title = metric.upper() + ' - ' + 'Dataset: ' + str(dataset_len) + \
            ' | Test: ' + str(test_len) + '| Mean: ' + \
            str(round(accuracies[metric].mean(), 4)) + ' | ' + \
            str(dimension) + '-' + str(division) + '-' + str(distance)
        accuracies.plot(kind='line', x='train_number',
                        y=metric, title=title, ax=ax)
        plt.savefig(save_path + metric + '_d_' +
                    str(dataset_len) + '_t_' + str(test_len) + '_' + str(dimension) + '_' + str(division) + '_' + str(distance) + '.png')
        plt.close(fig)


def test_reservoir(dataset_len, test_len, dataset_name, y_name):
    save_path = 'results/reservoir/' + y_name + '/' + dataset_name + '/'
    os.makedirs(save_path, exist_ok=True)

    log.debug(dataset_name + ' - Test = Dataset=' + str(dataset_len) +
              '|Test=' + str(test_len))

    model = Model()
    reservoir = Reservoir()
    emulator = OnlineEmulator(
        model, reservoir, data, dataset_len, test_len)
    emulator.start()

    plot_metrics(emulator, test_len, dataset_len, save_path)


def test_boxes(dataset_len, test_len, divisions, dimension, distance, dataset_name, y_name):
    save_path = 'results/boxes/' + y_name + '/' + dataset_name + '/'
    os.makedirs(save_path, exist_ok=True)

    log.debug(dataset_name + ' - Test = Dataset=' + str(dataset_len) +
              '|Test=' + str(test_len) + ' | ' +
              str(dimension) + '-' + str(divisions) + '-' + str(distance))

    model = Model()
    boxes = Boxes(dimension, divisions, dataset_len, distance)
    emulator = OnlineEmulator(
        model, boxes, data, dataset_len, test_len)
    emulator.start()

    plot_metrics2(emulator, test_len, dataset_len,
                  dimension, divisions, distance, save_path)


if __name__ == '__main__':

    values_for_test = [10, 100, 1000, 10000]
    freeze_support()

    log = Log()

    log.debug('Script start')

    config_file = 'config.json'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    loader = ConfigLoader(config_file)
    configs = loader.read()
    log.debug('Configs loaded')

    for dataset_config in configs['datasets']:
        dataset = load_data_set(dataset_config)

        x = pd.concat(dataset['x'], axis=1)
        y = dataset['y']

        x = processing_features(x, y, 16)

        data = {
            'x': x,
            'y': y
        }

        # test_reservoir(1000, 1000, 'teste')

        # test_boxes(10, 10, 3, 5, 0.03, 'teste', 'y')

        # log.debug('done')

        # exit()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dataset_length in values_for_test:
                for test_length in values_for_test:
                    for dimension in [3, 4]:
                        for divisions in [3, 4, 5]:
                            for distance in [0.01, 0.05, 0.1, 0.2, 0.4]:
                                executor.submit(test_boxes, dataset_length,
                                                test_length, divisions, dimension, distance, dataset_config['name'], dataset_config['target'])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dataset_length in values_for_test:
                for test_length in values_for_test:
                    executor.submit(test_reservoir, dataset_length,
                                    test_length, dataset_config['name'], dataset_config['target'])
