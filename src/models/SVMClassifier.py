from src.models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class SVMClassifier(Classifier):
    def __init__(self, X_train, X_test, y_train, y_test, loss, cv=True):
        super().__init__(X_train, X_test, y_train, y_test, loss)
        svm = SVC(C=10,
                  degree=3,
                  kernel='rbf',
                  verbose=False,
                  gamma=0.01,
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

    def plot_metric(self):
        pass

    def _research_hyperparameter(self):

        C = [1.*10**x for x in list(range(3))]
        gamma = [0.000001*10**x for x in list(range(5))]
        degree = [x for x in list(range(3, 6))]
        kernel = ['linear', 'rbf', 'poly']
        # parameters = [{'estimator__C': C, 'estimator__kernel': [kernel[0]]},
        #               {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__kernel': [kernel[1]]},
        #               {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__degree': degree, 'estimator__kernel': [kernel[2]]}]

        parameters = [{'estimator__C': [10], 'estimator__gamma': gamma, 'estimator__kernel': [kernel[1]]}]

        kappa_scorer = make_scorer(cohen_kappa_score)
        self.classifier = GridSearchCV(self.classifier, parameters,
                                       n_jobs=-1,
                                       verbose=2,
                                       cv=3,
                                       return_train_score=True,
                                       scoring=kappa_scorer)

        self.classifier.fit(self.X_train, self.y_train)
        print('Cross validation result')
        print(self.classifier.cv_results_)
        print('Best estimator: {}'.format(self.classifier.best_estimator_))
        print('Best score: {}'.format(self.classifier.best_score_))
        print('Best hyperparameters: {}'.format(self.classifier.best_params_))
        print('Refit time: {}'.format(self.classifier.refit_time_))




