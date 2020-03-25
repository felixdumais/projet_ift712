from abc import abstractmethod


class Classifier:
    def __init__(self):
        pass


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def error(self):
        pass

    @abstractmethod
    def _research_hyperparameter(self):
        pass

    def get_x_train(self):
        return self.X_train

    def get_x_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

