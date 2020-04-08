from models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class LogisticRegressor(Classifier):
    def __init__(self, cv=False):
        super().__init__()
        self.cv = cv
        self.trained = False
        logit = LogisticRegression(C=1.0, class_weight='balanced',
                                   dual=False,
                                   fit_intercept=False,
                                   intercept_scaling=1,
                                   l1_ratio=None, max_iter=200,
                                   multi_class='ovr', n_jobs=-1,
                                   penalty='l2',
                                   random_state=None,
                                   solver='liblinear', tol=0.01,
                                   verbose=False,
                                   warm_start=False)
        self.classifier = OneVsRestClassifier(estimator=logit, n_jobs=-1)

    def train(self, X_train, y_train):
        """
        Function that train the classifier

        :arg
            self (LogisticRegressor): instance of the class
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
            print('Fitting on Logistic Regressor Classifier...')
            self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Function that do a prediction on a set of data

        :arg
            self (LogisticRegressor): instance of the class
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
            self (LogisticRegressor): instance of the class
            X_test (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                  normalized pixel value

        :return
            self.classifier.predict_proba(X_test) (numpy array): probability

        """
        return self.classifier.predict_proba(X_test)

    def _research_hyperparameter(self, X_train, y_train):
        """
        Function that optimize some desired hyperparameter with cross-validation

        :arg
            self (LogisticRegressor): instance of the class
            X_train (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                   normalized pixel value
            y_train (numpy array): 1D or 2D numpy array corresponding to the targets

        :return
            None

        """
        C = [1.*10**x for x in list(range(3))]
        tol = [0.0001*10**x for x in list(range(3))]
        fit_intercept = [False, True]
        solver = ['liblinear','sag','newton-cg','lbfgs']
        class_weight=[None, 'balanced']
        multi_class = ['ovr','multinomial']
        parameters = [{'estimator__C': C, 'estimator__tol': tol, 'estimator__fit_intercept':fit_intercept,'estimator__solver':[solver[0]],
                       'estimator__class_weight':class_weight, 'estimator__multi_class':[multi_class[0]]},
                      {'estimator__C': C, 'estimator__tol': tol, 'estimator__fit_intercept':fit_intercept,
                       'estimator__solver':solver[1:], 'estimator__class_weight':class_weight, 'estimator__multi_class':multi_class}]

        
        self.classifier = GridSearchCV(self.classifier, parameters,
                                       n_jobs=-1,
                                       verbose=2,
                                       cv=3,
                                       return_train_score=True,
                                       scoring="f1_macro")

        self.classifier.fit(X_train, y_train)
        print('Cross validation result')
        print(self.classifier.cv_results_)
        print('Best estimator: {}'.format(self.classifier.best_estimator_))
        print('Best score: {}'.format(self.classifier.best_score_))
        print('Best hyperparameters: {}'.format(self.classifier.best_params_))
        print('Refit time: {}'.format(self.classifier.refit_time_))
