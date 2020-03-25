from models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class SVMClassifier(Classifier):
    def __init__(self, X_train, X_test, y_train, y_test, loss):
        super().__init__(X_train, X_test, y_train, y_test, loss)
        self.classifier = OneVsRestClassifier(LinearSVC(loss=self.loss, verbose=True, tol=0.001))
        self.kfolded = False

    def train(self):
        if self.kfolded is False:
            self._research_hyperparameter()
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.classifier.predict(self.X_test)
        return y_pred

    def error(self):
        pass

    def get_model(self):
        return self.classifier

    def _research_hyperparameter(self):
        self.kfolded = True
        current_score = None
        for i in range(1, 3):
            C = 0.1 * i
            self.classifier.set_params(estimator__C=C)
            scores = cross_val_score(self.classifier, self.X_train, self.y_train, cv=2, verbose=1, n_jobs=4)

            mean_score = scores.mean()
            if current_score is None:
                best_C = C
                current_score = mean_score
            elif mean_score > current_score:
                best_C = C
                current_score = mean_score

        self.classifier.set_params(estimator__C=best_C)



