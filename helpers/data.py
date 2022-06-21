from sklearn.model_selection import train_test_split


def split_data(data, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(
        data['x'], data['y'], test_size=test_size)
    train_data = {
        'x': x_train,
        'y': y_train
    }
    test_data = {
        'x': x_test,
        'y': y_test
    }
    return train_data, test_data
