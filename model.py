from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

class Model:

    test_size = 0.3
    n_estimators = 10
    model = None
    data = None

    def __split_data(self, data):
        x_train, x_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=self.test_size)
        return {
            'train': {
                'x': x_train,
                'y': y_train
            },
            'test': {
                'x': x_test,
                'y': y_test
            }
        }

    def __init__(self, data, test_size = 0.3, n_estimators = 10) -> None:
        self.rawData = data
        self.test_size = test_size
        self.n_estimators = n_estimators

    def train(self):
        if (self.data == None):
            self.data = self.__split_data(self.rawData)
    
        self.model = RandomForestRegressor(n_estimators=self.n_estimators)
        self.model.fit(self.data['train']['x'], self.data['train']['y'])

    def test_accuracy(self):
        if (self.model):
            return self.model.score(self.data['test']['x'], self.data['test']['y'])
        else:
            raise 'Model not trained'

    def predict(self,x_predict):
        if (self.model == None):
            raise 'Model not trained'
        return self.model.predict(x_predict)