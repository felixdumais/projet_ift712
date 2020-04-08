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
        if self.cv is True:
            self._research_hyperparameter(X_train, y_train)
        else:
            self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.classifier.predict(X_test)
        return y_pred

    def error(self):
        pass

    def predict_proba(self, X_test):
        return self.classifier.predict_proba(X_test)

    def _research_hyperparameter(self, X_train, y_train):


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
