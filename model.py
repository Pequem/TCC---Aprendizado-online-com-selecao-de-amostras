from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from numpy import mean


class Model:

    random_state = 100

    def __init__(self, n_estimators=10, online_mode=False) -> None:
        self.n_estimators = n_estimators
        self.online_mode = online_mode
        self.__init_model()

    def __init_model(self):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            warm_start=self.online_mode,
            random_state=self.random_state,
            n_jobs=-1
        )

    def train(self, data):
        self.model.fit(data['x'], data['y'].values.ravel())

    def test_accuracy(self, test_data):
        if self.model:
            mae = mean_absolute_error(
                test_data['y'], self.predict(test_data['x']))
            return {
                'r2': self.model.score(test_data['x'], test_data['y']),
                'nmae': mae / mean(test_data['y']),
                'mae': mae
            }
        else:
            raise Exception('Model not trained')

    def predict(self, x_predict):
        if self.model is None:
            raise Exception("Model not trained")
        return self.model.predict(x_predict)
