from abc import abstractmethod


class Classifier:
    def __init__(self):
        self.classifier = None
        pass

    def get_model(self):
        return self.classifier

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, image_to_predict):
        pass

    @abstractmethod
    def error(self):
        pass

    @abstractmethod
    def _research_hyperparameter(self, X_train, y_train):
        pass


