from sklearn.model_selection import train_test_split
import numpy as np
from models.LogisticRegressor import LogisticRegressor
from models.RandomForest import RandomForest
from models.SVMClassifier import SVMClassifier
from models.FisherDiscriminant import FisherDiscriminant
from models.RBF import RBFClassifier
from models.MLP import MLP
import copy
from Metrics import Metrics
import matplotlib.pyplot as plt
import os



class Trainer():
    def __init__(self, cross_validation, data, classifier_type, train_size, classifier):
        self.cross_validation = cross_validation
        self.data = data
        self.classifier_type = classifier_type
        self.train_size = train_size
        self.classifier = classifier

    def _split_data(self):
        """
        Function that split the dataset into training and testing set

        :arg
            self (Trainer): instance of the class

        :return
            train_image_all (numpy array): Numpy array containing train images
            test_image_all (numpy array): Numpy array containing test images
            train_labels_all (numpy array): Numpy array containing train target values including No Finding as a pathology
            test_labels_all (numpy array): Numpy array containing test target values including No Finding as a pathology
            train_labels_bool (numpy array): Numpy array containing train targets of sick people
            test_labels_bool (numpy array): Numpy array containing test targets of sick people
            train_image_sick (numpy array): Numpy array containing train targets excluding No Finding patients
            train_labels_sick (numpy array): Numpy array containing test targets excluding No Finding patients
        """
        image_all, labels_all = self.data.get_all_data()
        _, labels_bool = self.data.get_sick_bool_data()

        random_seed = 10

        train_image_all, test_image_all, train_labels_all, test_labels_all = train_test_split(image_all, labels_all,
                                                                                              train_size=self.train_size,
                                                                                              random_state=random_seed)
        if self.classifier_type == 2:
            _, _, train_labels_bool, test_labels_bool = train_test_split(image_all, labels_bool,
                                                                         train_size=self.train_size,
                                                                         random_state=random_seed)

            train_image_sick = train_image_all[train_labels_bool == 1]
            train_labels_sick = train_labels_all[train_labels_bool == 1]
            train_labels_sick = np.delete(train_labels_sick, 5, axis=1)

            return train_image_all, test_image_all, train_labels_all, test_labels_all, \
                   train_labels_bool, test_labels_bool, train_image_sick, train_labels_sick

        else:
            return train_image_all, test_image_all, train_labels_all, test_labels_all

    def classifier_selection(self):
        """
        Function that instanciates classifiers

        :arg
            self (Trainer): instance of the class

        :return
            model (Classifier): Selected model when self.classfier is 1
            model1 (Classifier): First selected model when self.classfier is 2
            model2 (Classifier): Second selected model when self.classfier is 2
            classifier_list (list): List with selected classifier names
        """
        if self.classifier == 'SVM':
            classifier_list = ['SVM']
            if self.classifier_type == 1:
                model = SVMClassifier(cv=self.cross_validation)
            elif self.classifier_type == 2:
                model1 = SVMClassifier(cv=self.cross_validation)
                model2 = SVMClassifier(cv=self.cross_validation)

        elif self.classifier == 'LogisticRegressor':
            classifier_list = ['LogisticRegressor']
            model = LogisticRegressor()
            if self.classifier_type == 1:
                model = LogisticRegressor(cv=self.cross_validation)
            elif self.classifier_type == 2:
                model1 = LogisticRegressor(cv=self.cross_validation)
                model2 = LogisticRegressor(cv=self.cross_validation)

        elif self.classifier == 'MLP':
            classifier_list = ['MLP']
            if self.classifier_type == 1:
                model = MLP(cv=self.cross_validation)
            elif self.classifier_type == 2:
                model1 = MLP(cv=self.cross_validation)
                model2 = MLP(cv=self.cross_validation)

        elif self.classifier == 'RandomForest':
            classifier_list = ['RandomForest']
            model = RandomForest()
            if self.classifier_type == 1:
                model = RandomForest(cv=self.cross_validation)
            elif self.classifier_type == 2:
                model1 = RandomForest(cv=self.cross_validation)
                model2 = RandomForest(cv=self.cross_validation)

        elif self.classifier == 'RBF':
            classifier_list = ['RBF']
            if self.classifier_type == 1:
                model = RBFClassifier(cv=self.cross_validation)
            elif self.classifier_type == 2:
                model1 = RBFClassifier(cv=self.cross_validation)
                model2 = RBFClassifier(cv=self.cross_validation)

        elif self.classifier == 'Fisher':
            classifier_list = ['Fisher']
            if self.classifier_type == 1:
                model = FisherDiscriminant(cv=self.cross_validation)
            elif self.classifier_type == 2:
                model1 = FisherDiscriminant(cv=self.cross_validation)
                model2 = FisherDiscriminant(cv=self.cross_validation)

        elif self.classifier == 'all':
            classifier_list = ['SVM', 'MLP', 'LogisticRegressor', 'RandomForest', 'RBF', 'Fischer']
            if self.classifier_type == 1:
                model_SVM = SVMClassifier(cv=self.cross_validation)
                model_MLP = MLP(cv=self.cross_validation)
                model_Logit = LogisticRegressor(cv=self.cross_validation)
                model_Forest = RandomForest(cv=self.cross_validation)
                model_RBF = RBFClassifier(cv=self.cross_validation)
                model_Fischer = FisherDiscriminant(cv=self.cross_validation)
                model = [model_SVM, model_MLP, model_Logit, model_Forest, model_RBF, model_Fischer]
            elif self.classifier_type == 2:
                model_SVM = SVMClassifier(cv=self.cross_validation)
                model_MLP = MLP(cv=self.cross_validation)
                model_Logit = LogisticRegressor(cv=self.cross_validation)
                model_Forest = RandomForest(cv=self.cross_validation)
                model_RBF = RBFClassifier(cv=self.cross_validation)
                model_Fischer = FisherDiscriminant(cv=self.cross_validation)
                model1 = [model_SVM, model_MLP, model_Logit, model_Forest, model_RBF, model_Fischer]
                model2 = copy.deepcopy(model1)
        else:
            raise SyntaxError('Invalid model name')

        if self.classifier_type == 1:
            return model, classifier_list
        elif self.classifier_type == 2:
            return model1, model2, classifier_list

    def training(self):
        """
        Function that trains classifiers

        :arg
            self (Trainer): instance of the class

        :return
            None
        """
        if self.classifier_type == 1:
            model, classifier_list = self.classifier_selection()
            train_image_all, test_image_all, train_labels_all, test_labels_all = self._split_data()
            if isinstance(model, list):
                pred = []
                proba = []
                for i, clf in enumerate(model):
                    try:
                        clf.train(train_image_all, train_labels_all)
                        pred_clf = clf.predict(test_image_all)
                        proba_clf = clf.predict_proba(test_image_all)
                        pred.append(pred_clf)
                        proba.append(proba_clf)
                        clf.save_model(filename=clf.__class__.__name__ + '_all')
                    except Exception as e:
                        print(e)
                        print('Could not train {}'.format(clf.__class__.__name__))
            else:
                model.train(train_image_all, train_labels_all)
                pred = [model.predict(test_image_all)]
                proba = [model.predict_proba(test_image_all)]
                model.save_model(filename=model.__class__.__name__ + '_all')

            label_list = self.data.label_.columns.values.tolist()
            self.display_metrics(classifier_list, test_labels_all, pred, proba, label_list)

        elif self.classifier_type == 2:
            model1, model2, classifier_list = self.classifier_selection()

            train_image_all, test_image_all, train_labels_all, test_labels_all,  \
                train_labels_bool, test_labels_bool, train_image_sick, train_labels_sick = self._split_data()
            if isinstance(model1, list) and isinstance(model2, list):
                pred = []
                proba = []
                for i, clf1 in enumerate(model1):
                    try:
                        clf1.train(train_image_all, train_labels_bool)
                        clf2 = model2[i]
                        clf2.train(train_image_sick, train_labels_sick)
                        prediction_matrix = np.zeros(test_labels_all.shape)
                        proba_matrix = np.zeros(test_labels_all.shape)

                        sick_bool_pred = clf1.predict(test_image_all)
                        idx_of_sick = np.nonzero(sick_bool_pred == 1)
                        test_image_sick = test_image_all[idx_of_sick]
                        sick_type_pred = clf2.predict(test_image_sick)
                        sick_type_pred = np.insert(sick_type_pred, 5, 0, axis=1)
                        prediction_matrix[idx_of_sick] = sick_type_pred
                        prediction_matrix[:, 5] = 1 - sick_bool_pred

                        sick_bool_proba = clf1.predict_proba(test_image_all)
                        sick_type_proba = clf2.predict_proba(test_image_sick)
                        sick_type_proba = np.insert(sick_type_proba, 5, 0, axis=1)
                        proba_matrix[idx_of_sick] = sick_type_proba
                        proba_matrix[:, 5] = sick_bool_proba[:, 0]

                        pred.append(prediction_matrix)
                        proba.append(proba_matrix)

                        clf1.save_model(filename=clf1.__class__.__name__ + '_bool')
                        clf2.save_model(filename=clf2.__class__.__name__ + '_sick')
                    except Exception as e:
                        print(e)
                        print('Could not train {}'.format(clf1.__class__.__name__))

            else:
                model1.train(train_image_all, train_labels_bool)
                model2.train(train_image_sick, train_labels_sick)
                prediction_matrix = np.zeros(test_labels_all.shape)
                proba_matrix = np.zeros(test_labels_all.shape)

                sick_bool_pred = model1.predict(test_image_all)
                idx_of_sick = np.nonzero(sick_bool_pred == 1)
                test_image_sick = test_image_all[idx_of_sick]
                sick_type_pred = model2.predict(test_image_sick)
                sick_type_pred = np.insert(sick_type_pred, 5, 0, axis=1)
                prediction_matrix[idx_of_sick] = sick_type_pred
                prediction_matrix[:, 5] = 1 - sick_bool_pred

                sick_bool_proba = model1.predict_proba(test_image_all)
                sick_type_proba = model2.predict_proba(test_image_sick)
                sick_type_proba = np.insert(sick_type_proba, 5, 0, axis=1)
                proba_matrix[idx_of_sick] = sick_type_proba
                proba_matrix[:, 5] = sick_bool_proba[:, 0]
                pred = [prediction_matrix]
                proba = [proba_matrix]

                model1.save_model(filename=model1.__class__.__name__ + '_bool')
                model2.save_model(filename=model2.__class__.__name__ + '_sick')

            label_list = self.data.label_.columns.values.tolist()
            self.display_metrics(classifier_list, test_labels_all, pred, proba, label_list)

    def predict_with_saved_model(self, *args):
        """
        Function that do a prediction with saved classifier

        :arg
            args (tuple): Tuple containing path of saved classifier

        :return
            None
        """
        if self.classifier == 'all':
            print('Cannot predict all on saved classifier')
            return None

        if self.classifier_type == 1:
            model, classifier_list = self.classifier_selection()
            train_image_all, test_image_all, train_labels_all, test_labels_all = self._split_data()
            model.load_model(args[0])
            pred = [model.predict(test_image_all)]
            proba = [model.predict_proba(test_image_all)]
            label_list = self.data.label_.columns.values.tolist()
            self.display_metrics(classifier_list, test_labels_all, pred, proba, label_list)
        elif self.classifier_type == 2:
            model1, model2, classifier_list = self.classifier_selection()

            train_image_all, test_image_all, train_labels_all, test_labels_all,  \
                train_labels_bool, test_labels_bool, train_image_sick, train_labels_sick = self._split_data()

            model1.load_model(args[0])
            model2.load_model(args[1])

            prediction_matrix = np.zeros(test_labels_all.shape)
            proba_matrix = np.zeros(test_labels_all.shape)

            sick_bool_pred = model1.predict(test_image_all)
            idx_of_sick = np.nonzero(sick_bool_pred == 1)
            test_image_sick = test_image_all[idx_of_sick]
            sick_type_pred = model2.predict(test_image_sick)
            sick_type_pred = np.insert(sick_type_pred, 5, 0, axis=1)
            prediction_matrix[idx_of_sick] = sick_type_pred
            prediction_matrix[:, 5] = 1 - sick_bool_pred

            sick_bool_proba = model1.predict_proba(test_image_all)
            sick_type_proba = model2.predict_proba(test_image_sick)
            sick_type_proba = np.insert(sick_type_proba, 5, 0, axis=1)
            proba_matrix[idx_of_sick] = sick_type_proba
            proba_matrix[:, 5] = sick_bool_proba[:, 0]
            pred = [prediction_matrix]
            proba = [proba_matrix]

            label_list = self.data.label_.columns.values.tolist()
            self.display_metrics(classifier_list, test_labels_all, pred, proba, label_list)

    @staticmethod
    def display_metrics(self, classifier_list: list, test_labels_all, pred: list, proba: list, label_list: list):
        """
        Function that display some metrics in regard of the training results

        :arg
            self (Trainer): instance of the class
            classifier_list (list): List with selected classifier names
            test_labels_all (numpy array): Numpy array containing target values of the testing set
            pred (list): list of the prediction. Each index of the list contains a numpy array
            proba (list): list of the probability. Each index of the list contains a numpy array
            label_list (list): list of the name of pathologies

        :return
            None
        """
        metrics = Metrics()

        plt.figure()
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        for i, value in enumerate(proba):
            fpr, tpr = metrics.roc_metrics(test_labels_all, value)
            plt.plot(fpr, tpr, label=classifier_list[i])

        if len(classifier_list) > 1:
            mean_proba = np.dstack(proba)
            mean_proba = np.mean(mean_proba, axis=2)
            fpr, tpr = metrics.roc_metrics(test_labels_all, mean_proba)
            plt.plot(fpr, tpr, label='Voting classifiers')

        plt.legend(loc='lower right')
        plt.show(block=False)

        plt.figure()
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        for i, value in enumerate(proba):
            precision, recall = metrics.precision_recall(test_labels_all, value)
            plt.plot(precision, recall, label=classifier_list[i])

        if len(classifier_list) > 1:
            mean_proba = np.dstack(proba)
            mean_proba = np.mean(mean_proba, axis=2)
            fpr, tpr = metrics.precision_recall(test_labels_all, mean_proba)
            plt.plot(fpr, tpr, label='Voting classifiers')

        plt.legend(loc='lower left')
        plt.show(block=False)

        if len(classifier_list) > 1:
            mean_pred = np.dstack(pred)
            mean_pred = np.mean(mean_pred, axis=2)
            mean_pred[mean_pred >= 0.5] = 1
            mean_pred[mean_pred < 0.5] = 0
            pred = mean_pred
        else:
            pred = pred[0]

        cohen_kappa_score, kappa_class = metrics.cohen_kappa_score(test_labels_all, pred)
        f1_score, f1_class = metrics.f1_score(test_labels_all, pred)
        accuracy, accuracy_class = metrics.accuracy(test_labels_all, pred)
        precision, precision_class = metrics.precision(test_labels_all, pred)
        recall, recall_class = metrics.recall(test_labels_all, pred)
        #
        print('Cohen: {}'.format(cohen_kappa_score))
        print('F1: {}'.format(f1_score))
        print('Accuracy: {}'.format(accuracy))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))

        titles = ['names', 'Cohen', 'F1_score', 'Accuracy', 'Precision', 'Recall']
        kappa_class_disp = ['%.4f' % elem for elem in kappa_class]
        f1_class_disp = ['%.4f' % elem for elem in f1_class]
        accuracy_class_disp = ['%.4f' % elem for elem in accuracy_class]
        precision_class_disp = ['%.4f' % elem for elem in precision_class]
        recall_class_disp = ['%.4f' % elem for elem in recall_class]

        element = [titles] + list(
           zip(label_list, kappa_class_disp, f1_class_disp, accuracy_class_disp, precision_class_disp,
               recall_class_disp))
        for i, d in enumerate(element):
           line = '|'.join(str(x).ljust(19) for x in d)
           print(line)
           if i == 0:
               print('-' * (len(line)-10))
