from abc import abstractmethod

class Classifier:
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def error(self):
        pass