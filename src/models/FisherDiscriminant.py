from models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
import pickle


class FisherDiscriminant(Classifier):
    def __init__(self, cv = False):
        super().__init__()
        self.cv = cv
        self.trained = False
        self.classifier = OneVsRestClassifier(LinearDiscriminantAnalysis(solver='svd', tol=0.01, n_components=12), n_jobs = 2)
 

    def train(self, X_train, y_train):
        """
        Function that train the classifier

        :arg
            self (FisherDiscriminant): instance of the class
            X_train (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                   normalized pixel value
            y_train (numpy array): 1D or 2D numpy array corresponding to the targets

        :return
            None

        """
        if self.cv is True:
            self._research_hyperparameter(X_train, y_train)
        else:
            self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Function that do a prediction on a set of data

        :arg
            self (FisherDiscriminant): instance of the class
            X_test (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                  normalized pixel value

        :return
            y_pred (numpy array): 1D or 2D numpy array corresponding to the targets

        """
        y_pred = self.classifier.predict(X_test)
        return y_pred

    def error(self):
        pass
    
    def predict_proba(self, X_test):
        """
        Function that do a prediction on a set of data

        :arg
            self (FisherDiscriminant): instance of the class
            X_test (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                  normalized pixel value

        :return
            self.classifier.predict_proba(X_test) (numpy array): probability

        """
        return self.classifier.predict_proba(X_test)

    def _research_hyperparameter(self, X_train, y_train):
        """
        Function that optimize some desired hyperparameters with cross-validation

        :arg
            self (FisherDiscriminant): instance of the class
            X_train (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                   normalized pixel value
            y_train (numpy array): 1D or 2D numpy array corresponding to the targets

        :return
            None

        """
        n_components = [12 + x for x in list(range(3))]
        tol = [10**(-1*x) for x in list(range(1,3))]
        parameters = [{'estimator__n_components': n_components},
                      {'estimator__n_components': n_components, 'estimator__tol': tol}]

        self.classifier = GridSearchCV(self.classifier, parameters,
                                       n_jobs=1,
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

