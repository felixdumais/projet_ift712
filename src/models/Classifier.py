from abc import abstractmethod


class Classifier:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def error(self):
        pass

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data