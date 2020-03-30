from src.models.Classifier import Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer
from itertools import combinations_with_replacement


class MLP(Classifier):
    def __init__(self, cv=False):
        super().__init__()

        mlp = MLPClassifier(hidden_layer_sizes=(10, 10),
                            activation='relu',
                            solver='adam',
                            alpha=0.001,
                            batch_size=1000,
                            learning_rate='adaptive',
                            learning_rate_init=0.001,
                            verbose=True,
                            max_iter=25,
                            warm_start=True)

        self.classifier = OneVsRestClassifier(estimator=mlp, n_jobs=-1)
        self.cv = cv
        self.trained = False


    def train(self, X_train, y_train):
        if self.cv is True:
            self._research_hyperparameter(X_train, y_train)
            print('Done')
        else:
            self.classifier.fit(X_train, y_train)

    def predict(self, image_to_predict):
        y_pred = self.classifier.predict(image_to_predict)

        return y_pred

    def predict_proba(self, X_test):
        return self.classifier.predict_proba(X_test)

    def error(self):
        pass

    def get_model(self):
        return self.classifier



    def _research_hyperparameter(self, X_train, y_train):

        alpha = [0.0001*10**x for x in list(range(3))]
        combination = (10, 100)
        comb1 = list(combinations_with_replacement(combination, 1))
        comb2 = list(combinations_with_replacement(combination, 2))
        total_com = comb1 + comb2

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
