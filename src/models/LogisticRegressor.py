from models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer



class LogisticRegressor(Classifier):
    def __init__(self, cv=False):
        super().__init__()
        self.cv = cv
        self.trained = False
        logit = LogisticRegression(penalty='l2',
                                   dual=False,
                                   tol=0.0001,
                                   C=1.0,
                                   fit_intercept=True,
                                   intercept_scaling=1,
                                   class_weight="balanced",
                                   solver='liblinear',
                                   max_iter=100,
                                   multi_class='ovr',
                                   verbose=True,
                                   warm_start=False,
                                   n_jobs=None)
                                 #  l1_ratios=None)
        self.classifier = OneVsRestClassifier(estimator=logit, n_jobs=-1)
        #self.classifier = logit

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


#        C = [1.*10**x for x in list(range(3))]
#        gamma = [0.000001*10**x for x in list(range(5))]
#        degree = [x for x in list(range(3, 6))]
#        kernel = ['linear', 'rbf', 'poly']
#        parameters = [{'estimator__C': C, 'estimator__kernel': [kernel[0]]},
#                      {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__kernel': [kernel[1]]},
#                      {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__degree': degree, 'estimator__kernel': [kernel[2]]}]

        
        self.classifier = GridSearchCV(self.classifier, parameters,
                                       n_jobs=-1,
                                       verbose=2,
                                       cv=3,
                                       return_train_score=True,
                                       scoring="f1_weighted")

        self.classifier.fit(X_train, y_train)
        print('Cross validation result')
        print(self.classifier.cv_results_)
        print('Best estimator: {}'.format(self.classifier.best_estimator_))
        print('Best score: {}'.format(self.classifier.best_score_))
        print('Best hyperparameters: {}'.format(self.classifier.best_params_))
        print('Refit time: {}'.format(self.classifier.refit_time_))
