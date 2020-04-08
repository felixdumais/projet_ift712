from models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV
import pickle


class RBFClassifier(Classifier):
    def __init__(self, cv = False):#X_train, X_test, y_train, y_test, loss):

        super().__init__()#(X_train, X_test, y_train, y_test, loss)
        kernel = RBF(length_scale=1.0)
        self.classifier = OneVsRestClassifier(GaussianProcessClassifier(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=2, max_iter_predict=10, warm_start=False, copy_X_train=True, random_state=None, n_jobs=2))
        #self.kfolded = True
        self.cv = cv
        self.trained = False

    def train(self, X_train, y_train):
        if self.cv is True:
            self._research_hyperparameter(X_train, y_train)
        self.classifier.fit(X_train, y_train)
        
    def predict(self, image_to_predict):
        """
        Function that do a prediction on a set of data

        :arg
            self (RBF): instance of the class
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
            self (RBF): instance of the class
            X_test (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                  normalized pixel value

        :return
            self.classifier.predict_proba(X_test) (numpy array): probability

        """
        return self.classifier.predict_proba(X_test)

    def error(self):
        pass


    def _research_hyperparameter(self, X_train, y_train):
        n_restarts_optimizer = [10**x for x in list(range(1,3))]
        max_iter_predict = [10**x for x in list(range(1,3))]
        parameters = [{'estimator__n_restarts_optimizer': n_restarts_optimizer,'estimator__max_iter_predict': max_iter_predict}]

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


