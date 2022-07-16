import random

import pandas


class Reservoir:

    def set_init_dataset(self, dataset):
        self.dataset = dataset
        self.dataset_len = len(self.dataset['x'])
        self.dataset_change_count = 0

    def get_dataset(self):
        return self.dataset

    def should_retrain(self):
        prob = random.randint(0, self.dataset_len)
        if prob < self.dataset_change_count:
            self.dataset_change_count = 0
            return True
        return False

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
            self.dataset_change_count += 1
