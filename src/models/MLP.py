from src.models.Classifier import Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class MLP(Classifier):
    def __init__(self, X_train, X_test, y_train, y_test, loss):
        super().__init__(X_train, X_test, y_train, y_test, loss)
        self.classifier = MLPClassifier(hidden_layer_sizes=(1000, 100),
                                        activation='relu',
                                        solver='adam',
                                        alpha=0.0001,
                                        batch_size=10,
                                        learning_rate='adaptive',
                                        learning_rate_init=0.001,
                                        verbose=True,
                                        max_iter=3)
        self.classifier = MultiOutputClassifier(self.classifier)
        self.kfolded = False

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.classifier.predict(self.X_test)
        return y_pred

    def error(self):
        pass

    def get_model(self):
        return self.classifier

    def _research_hyperparameter(self):
        pass



