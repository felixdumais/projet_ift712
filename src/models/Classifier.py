from abc import abstractmethod
import pickle
import os


class Classifier:
    def __init__(self):
        self.classifier = None

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

    def save_model(self, filename='model'):
        """
        Function that saves the classifier

        :arg
            self (SVMClassifier): instance of the class

        :return
            None

        """
        check_dir = 'trained_models'
        if not os.path.isdir(check_dir):
            os.mkdir(check_dir)

        pickle.dump(self.classifier, open(os.path.join(check_dir, filename), 'wb'))

    def load_model(self, filename):
        """
        Function that loads the classifier

        :arg
            self (SVMClassifier): instance of the class

        :return
            None

        """
        if not os.path.isfile(filename):
            raise OSError('Cannot load the model: {}'.format(filename))

        self.classifier = pickle.load(open(filename, 'rb'))


