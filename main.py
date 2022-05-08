from loaders.config import Loader as ConfigLoader
from loaders.dataset import Dataset as DatasetLoader
from helpers.log import Log
from concurrent import futures
from multiprocessing import freeze_support
import pandas as pd
import sys
import code
from model import Model

def LoadDataSets(configs):
    executor = futures.ProcessPoolExecutor(12)
    x_promisses = []
    y_promisses = []
    
    x_datasets = []
    y_dataset = None

    for dataset in configs['datasets']:
        for xs in dataset['x']:
            d_loader = DatasetLoader(xs)
            t = executor.submit(d_loader.read)
            x_promisses.append(t)

        d_loader = DatasetLoader(dataset['y'])
        t = executor.submit(d_loader.read)
        y_promisses.append(t)

    for promise in x_promisses:
        x_datasets.append(promise.result())

    for promise in y_promisses:
        y_dataset = promise.result()

    return {
        'x': x_datasets,
        'y': y_dataset
    }

if __name__ == '__main__': 

    freeze_support()

    log = Log()

    log.debug('Script start')

    configFile = 'config.json'

    if (len(sys.argv) > 1):
        configFile = sys.argv[1]

    loader = ConfigLoader(configFile)

    configs = loader.read()

    log.debug('Configs loaded')

    datasets = LoadDataSets(configs)

    model = Model({
        'x': pd.concat(datasets['x'], axis=1),
        'y': datasets['y']['AvgInterDispDelay']
    })

    model.train()
    
    print(model.test_accuracy())

    #code.interact(local=locals())

