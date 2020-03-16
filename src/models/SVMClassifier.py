from src.models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class SVMClassifier(Classifier):
    def __init__(self, X_train, X_test, y_train, y_test, loss):
        super().__init__(X_train, X_test, y_train, y_test, loss)
        self.classifier = OneVsRestClassifier(SVC(kernel='linear', verbose=False, tol=0.001))
        self.kfolded = False

    def train(self):
        # if self.kfolded is False:
        #     self._research_hyperparameter()
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
        for i in range(2):
            C = 0.0001*10**i
            for j in range(2):
                gamma = 0.0001*10**j
                self.classifier.set_params(estimator__estimator__C=C, estimator__estimator__gamma=gamma)
                scores = cross_val_score(self.classifier, self.X_train, self.y_train, cv=5, verbose=False, n_jobs=-1)

                mean_score = scores.mean()
                print('C = {}'.format(C))
                print('gamma = {}'.format(gamma))
                print('scores = {}'.format(scores))
                print('mean_scores = {}'.format(mean_score))
                if current_score is None:
                    best_C = C
                    best_gamma = gamma
                    current_score = mean_score
                elif mean_score > current_score:
                    best_C = C
                    best_gamma = gamma
                    current_score = mean_score

        self.classifier.set_params(estimator__estimator__C=best_C, estimator__estimator__gamma=best_gamma)




