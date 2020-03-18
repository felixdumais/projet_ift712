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
    def __init__(self, X_train, X_test, y_train, y_test, loss, cv=True):
        super().__init__(X_train, X_test, y_train, y_test, loss)
        svm = SVC(C=1,
                  degree=3,
                  kernel='linear',
                  verbose=False,
                  tol=0.001,
                  max_iter=-1)
        self.classifier = OneVsRestClassifier(estimator=svm, n_jobs=-1)
        self.cv = cv

    def train(self):
        if self.cv is True:
            self._research_hyperparameter()
        else:
            self.classifier.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.classifier.predict(self.X_test)

        boolean_vector = y_pred[:, 5] == 1
        y_pred[boolean_vector, :] = 0
        y_pred[:, 5] = boolean_vector

        return y_pred

    def error(self):
        pass

    def get_model(self):
        return self.classifier

    def _research_hyperparameter(self):
        # current_score = None
        # for i in range(2):
        #     C = 0.000001*10**i
        #     for j in range(2):
        #         gamma = 0.000001*10**j
        #         self.classifier.set_params(estimator__C=C, estimator__gamma=gamma)
        #         scores = cross_val_score(self.classifier, self.X_train, self.y_train, cv=3, verbose=False, n_jobs=-1)
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
        # self.classifier.set_params(estimator__C=best_C, estimator__gamma=best_gamma)
        # self.classifier.fit(self.X_train, self.y_train)

        C = [1.*10**x for x in list(range(3))]
        gamma = [0.01*10**x for x in list(range(3))]
        degree = [x for x in list(range(3, 6))]
        kernel = ['linear', 'rbf', 'poly']
        parameters = [{'estimator__C': C, 'estimator__kernel': [kernel[0]]},
                      {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__kernel': [kernel[1]]},
                      {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__degree': degree, 'estimator__kernel': [kernel[2]]}]

        self.classifier = GridSearchCV(self.classifier, parameters,
                                       n_jobs=-1,
                                       verbose=2,
                                       cv=3,
                                       return_train_score=True,
                                       scoring='precision_macro')
        self.classifier.fit(self.X_train, self.y_train)
        print('Cross validation result')
        print(self.classifier.cv_results_)
        print('Best estimator: {}'.format(self.classifier.best_estimator_))
        print('Best score: {}'.format(self.classifier.best_score_))
        print('Best hyperparameters: {}'.format(self.classifier.best_params_))
        print('Refit time: {}'.format(self.classifier.refit_time_))




