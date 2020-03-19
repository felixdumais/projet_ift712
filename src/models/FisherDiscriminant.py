from src.models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class FisherDiscriminant(Classifier):
    def __init__(self, X_train, X_test, y_train, y_test, loss):
        super().__init__(X_train, X_test, y_train, y_test, loss)
        self.classifier = OneVsRestClassifier(LinearDiscriminantAnalysis(solver='svd', tol=0.1, n_components=12), n_jobs = 2)
        #svd : this solver is recommended for data with a large number of features. 
        #Any other solver causes a MemoryError because of the size of the samples. 
        #Shrinkage is impossible with this solver (set to None by default).
        #tol set to 0.1 and n_components set to 12 following an optimisation of the hyperparameters with _research_hyperparameters       self.kfolded = True

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
        n_components = [12 + x for x in list(range(3))]
        tol = [10^(-1*x) for x in list(range(1,3))]
        parameters = [{'estimator__n_components': n_components},
                      {'estimator__n_components': n_components, 'estimator__tol': tol}]

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

