from sklearn.metrics import accuracy_score
from helpers.data import split_data


class OnlineEmulator:

    test_data = None
    train_data = None
    train_count = 0
    accuracies = []

    def __init__(self, model, sample_select, data, dataset_length=100):
        self.model = model
        self.dataset_length = dataset_length
        self.sample_select = sample_select
        self.prepare_data(data)
        self.do_first_train()

    def prepare_data(self, data):
        train_data, test_data = split_data(data)
        self.train_data = train_data
        self.test_data = test_data

    def init_sample_select_alg_data_set(self):
        init_x = self.train_data['x'][:self.dataset_length]
        init_y = self.train_data['y'][:self.dataset_length]
        self.sample_select.set_init_dataset({
            'x': init_x,
            'y': init_y
        })

    def do_first_train(self):
        self.init_sample_select_alg_data_set()
        self.train()

    def train(self):
        self.train_count += 1
        self.model.train(self.sample_select.get_dataset())
        accurary = self.model.test_accuracy(self.test_data)
        accurary['train_number'] = self.train_count
        self.accuracies.append(accurary)

    def start(self):
        for i in range(self.dataset_length, len(self.train_data['x'])):
            x = self.train_data['x'].iloc[i].values
            y = self.train_data['y'].iloc[i]
            self.sample_select.handle({'x': x, 'y': y}, i)
            if self.sample_select.should_retrain():
                self.train()

    def get_accuracies(self):
        return self.accuracies
