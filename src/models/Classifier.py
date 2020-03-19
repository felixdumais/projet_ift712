from abc import abstractmethod


class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test, loss):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.loss = loss


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, image_to_predict):
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

