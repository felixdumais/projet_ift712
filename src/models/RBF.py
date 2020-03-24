from src.models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class RBFClassifier(Classifier):
    def __init__(self, X_train, X_test, y_train, y_test, loss):

        super().__init__(X_train, X_test, y_train, y_test, loss)
        kernel = RBF(length_scale=1.0)
        self.classifier = OneVsRestClassifier(GaussianProcessClassifier(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10, max_iter_predict=10, warm_start=False, copy_X_train=True, random_state=None, n_jobs=2))
        # multi_class='one_vs_rest', 
        #OneVsRestClassifier(
        self.kfolded = True

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
        n_restarts_optimizer = [10**x for x in list(range(1,3))]
        max_iter_predict = [10**x for x in list(range(1,3))]
        parameters = [{'estimator__n_restarts_optimizer': n_restarts_optimizer,'estimator__max_iter_predict': max_iter_predict}]

        self.classifier = GridSearchCV(self.classifier, parameters,
                                       n_jobs=1,
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


