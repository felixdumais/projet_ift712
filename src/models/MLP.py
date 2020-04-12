from models.Classifier import Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import time
import numpy as np



class MLP(Classifier):
    def __init__(self, cv=False):
        super().__init__()

        mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                            activation='relu',
                            solver='adam',
                            alpha=0.01,
                            batch_size=100,
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            tol=1e-4,
                            verbose=False,
                            max_iter=75,
                            shuffle=True,
                            warm_start=False,
                            early_stopping=True,
                            validation_fraction=0.1,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-8,
                            n_iter_no_change=20)

        self.classifier = OneVsRestClassifier(estimator=mlp, n_jobs=-1)
        self.cv = cv
        self.trained = False

    def train(self, X_train, y_train):
        """
        Function that train the classifier

        :arg
            self (MLP): instance of the class
            X_train (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                   normalized pixel value
            y_train (numpy array): 1D or 2D numpy array corresponding to the targets

        :return
            None

        """
        if self.cv is True:
            self._research_hyperparameter(X_train, y_train)
            print('Done')
        else:
            print('Fitting on MLP Classifier...')
            self.classifier.fit(X_train, y_train)

    def predict(self, image_to_predict):
        """
        Function that do a prediction on a set of data

        :arg
            self (MLP): instance of the class
            X_test (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                  normalized pixel value

        :return
            y_pred (numpy array): 1D or 2D numpy array corresponding to the targets

        """

        y_pred = self.classifier.predict(image_to_predict)

        return y_pred

    def predict_proba(self, X_test):
        """
        Function that do a prediction on a set of data

        :arg
            self (MLP): instance of the class
            X_test (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                  normalized pixel value

        :return
            self.classifier.predict_proba(X_test) (numpy array): probability

        """
        return self.classifier.predict_proba(X_test)

    def error(self):
        pass

    def _research_hyperparameter(self, X_train, y_train):
        """
        Function that optimize some desired hyperparameter with cross-validation

        :arg
            self (MLP): instance of the class
            X_train (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                   normalized pixel value
            y_train (numpy array): 1D or 2D numpy array corresponding to the targets

        :return
            None

        """
        start = time.time()
        print("Start time computation")

        alpha = [0.0001*10**x for x in list(range(5))]
        combination = (10, 100, 1000)
        comb1 = list(combinations_with_replacement(combination, 1))
        comb2 = list(combinations_with_replacement(combination, 2))
        comb3 = list(combinations_with_replacement(combination, 3))
        total_com = comb1 + comb2 + comb3

        parameters = [{'estimator__alpha': alpha, 'estimator__hidden_layer_sizes': total_com}]

        # kappa_scorer = make_scorer(cohen_kappa_score)
        self.classifier = GridSearchCV(self.classifier, parameters,
                                       n_jobs=-1,
                                       verbose=2,
                                       cv=3,
                                       return_train_score=True,
                                       scoring='f1_macro')

        self.classifier.fit(X_train, y_train)
        print('Cross validation result')
        print(self.classifier.cv_results_)
        print('Best estimator: {}'.format(self.classifier.best_estimator_))
        print('Best score: {}'.format(self.classifier.best_score_))
        print('Best hyperparameters: {}'.format(self.classifier.best_params_))
        print('Refit time: {}'.format(self.classifier.refit_time_))

        end = time.time()
        print(end - start)
