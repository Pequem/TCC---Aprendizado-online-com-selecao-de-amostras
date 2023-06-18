class OnlineEmulator:

    def __init__(self, model, sample_select, data, dataset_length=100, test_data_length=100):
        self.model = model
        self.dataset_length = dataset_length
        self.sample_select = sample_select
        self.data = data
        self.train_count = 0
        self.current_position = 0
        self.accuracies = []
        self.test_data_length = test_data_length
        self.do_first_train()

    def init_sample_select_alg_data_set(self):
        init_x = self.data['x'][:self.dataset_length]
        init_y = self.data['y'][:self.dataset_length]
        self.sample_select.set_init_dataset({
            'x': init_x,
            'y': init_y
        })

    def do_first_train(self):
        self.init_sample_select_alg_data_set()
        self.train()

    def test(self):
        data_x = self.data['x'][self.current_position:
                                self.current_position + self.test_data_length + 1]
        data_y = self.data['y'][self.current_position:
                                self.current_position + self.test_data_length + 1]

        accurary = self.model.test_accuracy(
            {
                'x': data_x,
                'y': data_y
            }
        )
        accurary['train_number'] = self.train_count
        self.accuracies.append(accurary)

    def train(self):
        self.train_count += 1
        self.model.train(self.sample_select.get_dataset())

    def start(self):
        for i in range(self.dataset_length, len(self.data['x'])):
            self.current_position = i
            x = self.data['x'].iloc[i].values
            y = self.data['y'].iloc[i]
            self.sample_select.handle({'x': x, 'y': y}, i)
            if self.current_position % 100 == 0:
                self.test()
            if self.sample_select.should_retrain():
                self.train()

    def get_accuracies(self):
        return self.accuracies

    def get_model(self):
        return self.model

    def get_data(self):
        return self.data
