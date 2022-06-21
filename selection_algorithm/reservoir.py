import random

import pandas


class Reservoir:

    dataset = []
    dataset_len = 0
    dataset_change = False

    def set_init_dataset(self, dataset):
        self.dataset = dataset
        self.dataset_len = len(self.dataset['x'])

    def get_dataset(self):
        return self.dataset

    def should_retrain(self):
        response = self.dataset_change
        self.dataset_change = False
        return response

    def handle(self, sample, index):
        prob = random.randint(1, index)
        if prob < self.dataset_len:
            x_values = self.dataset['x'].values
            y_values = self.dataset['y'].values
            x_values[prob] = sample['x']
            y_values[prob] = sample['y']

            self.dataset['x'] = pandas.DataFrame(
                data=x_values, columns=self.dataset['x'].columns)
            self.dataset['y'] = pandas.DataFrame(
                data=y_values
            )
            self.dataset_change = True
