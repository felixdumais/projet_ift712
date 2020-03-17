from src.models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class SVMClassifier(Classifier):
    def __init__(self, X_train, X_test, y_train, y_test, loss):
        super().__init__(X_train, X_test, y_train, y_test, loss)
        svm = SVC(kernel='linear', verbose=True, tol=0.001)
        self.classifier = BinaryRelevance(classifier=svm)
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
        # current_score = None
        # for i in range(6):
        #     C = 0.000001*10**i
        #     for j in range(6):
        #         gamma = 0.000001*10**j
        #         self.classifier.set_params(classifier__C=C, classifier__gamma=gamma)
        #         scores = cross_val_score(self.classifier, self.X_train, self.y_train, cv=5, verbose=False, n_jobs=-1)
        #
        #         mean_score = scores.mean()
        #         print('C = {}'.format(C))
        #         print('gamma = {}'.format(gamma))
        #         print('scores = {}'.format(scores))
        #         print('mean_scores = {}'.format(mean_score))
        #         if current_score is None:
        #             best_C = C
        #             best_gamma = gamma
        #             current_score = mean_score
        #         elif mean_score > current_score:
        #             best_C = C
        #             best_gamma = gamma
        #             current_score = mean_score
        #
        # self.classifier.set_params(classifier__C=best_C, classifier__gamma=best_gamma)

        C = [0.00001*10**x for x in list(range(7))]
        parameters = {'classifier__C': C}
        self.classifier = GridSearchCV(self.classifier, parameters, n_jobs=-1, verbose=1)
        self.classifier.fit(self.X_train, self.y_train)
        print(self.classifier.cv_results_)




