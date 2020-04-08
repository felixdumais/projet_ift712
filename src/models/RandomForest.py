from models.Classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForest(Classifier):
    def __init__(self, cv=False):
        super().__init__()
        self.cv = cv
        self.trained = False
        forest = RandomForestClassifier(bootstrap=True,
                                         ccp_alpha=0.0,
                                         class_weight='balanced',
                                         criterion='gini',
                                         max_depth=1.0,
                                         max_features='log2',
                                         max_leaf_nodes=None,
                                         max_samples=None,
                                         min_impurity_decrease=0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0,
                                         n_estimators=50, n_jobs=-1,
                                         oob_score=True,
                                         random_state=None,
                                         verbose=False,
                                         warm_start=False)
        self.classifier = OneVsRestClassifier(estimator=forest)

    def train(self, X_train, y_train):
        """
        Function that train the classifier

        :arg
            self (RandomForest): instance of the class
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
            self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Function that do a prediction on a set of data

        :arg
            self (RandomForest): instance of the class
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
            self (RandomForest): instance of the class
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
            self (RandomForest): instance of the class
            X_train (numpy array): 2D numpy array where each rows represent a flatten image and each column is a
                                   normalized pixel value
            y_train (numpy array): 1D or 2D numpy array corresponding to the targets

        :return
            None

        """
        n_estimators = [50+50*x for x in list(range(3))]
        criterion = ['gini', 'entropy']
        max_depth = [1.*10**x for x in list(range(3))]+[None]
        max_features = ['sqrt', 'log2', None]
        bootstrap = [True]
        oob_score = [True]
        class_weight = ['balanced', 'balanced_subsample', None]
        parameters = [{'estimator__n_estimators': n_estimators, 'estimator__criterion': criterion, 'estimator__max_depth':max_depth,
                        'estimator__max_features':max_features, 'estimator__class_weight':class_weight},
                       {'estimator__n_estimators': n_estimators, 'estimator__criterion': criterion, 'estimator__max_depth':max_depth,
                        'estimator__max_features':max_features, 'estimator__bootstrap':bootstrap, 'estimator__oob_score':oob_score,
                        'estimator__class_weight':class_weight}]

        
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
