from models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer
import pickle



class SVMClassifier(Classifier):
    def __init__(self, cv=False):
        super().__init__()
        self.cv = cv
        self.trained = False
        svm = SVC(C=10,
                  degree=3,
                  kernel='rbf',
                  verbose=True,
                  gamma=0.01,
                  tol=0.001,
                  probability=True,
                  max_iter=-1)
        self.classifier = OneVsRestClassifier(estimator=svm, n_jobs=-1)

    def train(self, X_train, y_train):
        """
        Function that train the classifier

        :arg
            self (SVMClassifier): instance of the class
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
            self (SVMClassifier): instance of the class
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
            self (SVMClassifier): instance of the class
            X_test (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                  normalized pixel value

        :return
            self.classifier.predict_proba(X_test) (numpy array): probability

        """
        return self.classifier.predict_proba(X_test)

    def save_model(self):
        """
        Function that saves the classifier

        :arg
            self (SVMClassifier): instance of the class

        :return
            None

        """
        filename = '../trained_models/SVM_model.mdl'
        pickle.dump(self.classifier, open(filename, 'wb'))

    def load_model(self):
        """
        Function that loads the classifier

        :arg
            self (SVMClassifier): instance of the class

        :return
            None

        """
        filename = '../trained_models/SVM_model.mdl'
        self.classifier = pickle.load(open(filename, 'rb'))

    def _research_hyperparameter(self, X_train, y_train):
        """
        Function that optimize some desired hyperparameters with cross-validation

        :arg
            self (SVMClassifier): instance of the class
            X_train (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                   normalized pixel value
            y_train (numpy array): 1D or 2D numpy array corresponding to the targets

        :return
            None

        """

        C = [1.*10**x for x in list(range(3))]
        gamma = [0.000001*10**x for x in list(range(5))]
        degree = [x for x in list(range(3, 6))]
        kernel = ['linear', 'rbf', 'poly']
        parameters = [{'estimator__C': C, 'estimator__kernel': [kernel[0]]},
                      {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__kernel': [kernel[1]]},
                      {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__degree': degree, 'estimator__kernel': [kernel[2]]}]

        kappa_scorer = make_scorer(cohen_kappa_score)
        self.classifier = GridSearchCV(self.classifier, parameters,
                                       n_jobs=-1,
                                       verbose=2,
                                       cv=3,
                                       return_train_score=True,
                                       scoring=kappa_scorer)

        self.classifier.fit(X_train, y_train)
        print('Cross validation result')
        print(self.classifier.cv_results_)
        print('Best estimator: {}'.format(self.classifier.best_estimator_))
        print('Best score: {}'.format(self.classifier.best_score_))
        print('Best hyperparameters: {}'.format(self.classifier.best_params_))
        print('Refit time: {}'.format(self.classifier.refit_time_))




