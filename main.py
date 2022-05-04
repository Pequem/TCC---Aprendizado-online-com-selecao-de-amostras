import pandas as pd
from loaders.config import Loader as ConfigLoader
from loaders.dataset import Dataset as DatasetLoader
from helpers.log import Log
import sys
import code

log = Log()

log.degub('Script start')

loader = ConfigLoader(sys.argv[1])

configs = loader.read()

log.degub('Configs loaded')

x_datasets = []
y_dataset = None

for dataset in configs['datasets']:
    for xs in dataset['x']:
        log.degub('Loading dataset ' + xs)
        d_loader = DatasetLoader(xs)
        x_datasets.append(d_loader.read())

    log.degub('Loading dataset ' + dataset['y'])
    d_loader = DatasetLoader(dataset['y'])
    y_dataset = d_loader.read()

code.interact(local=locals())