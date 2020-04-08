import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Metrics:
    def __init__(self):
        pass

    def accuracy(self, y_true, y_pred):
        """
        Function that compute the accuracy on a multiclass-multilabel problem

        :arg
            self (Metrics): instance of the class
            y_true (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the true pathology
            y_pred (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the predicted pathology

        :return
            mean_accuracy (float): mean accuracy of all the prediction done
            accuracy_class (list): accuracy of each class

        """
        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]

        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        accuracy_class = []
        for i in range(y_true.shape[1]):
            accuracy_class.append(accuracy_score(y_true[:, i], y_pred[:, i]))

        y_true = y_true.flatten(order='C')
        y_pred = y_pred.flatten(order='C')
        mean_accuracy = accuracy_score(y_true, y_pred)

        return mean_accuracy, accuracy_class

    def roc_metrics(self, y_true, y_proba):
        """
        Function that compute the false and true positive rates

        :arg
            self (Metrics): instance of the class
            y_true (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the true pathology
            y_proba (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the predicted probability pathology

        :return
            fpr (numpy array): false positive rate
            tpr (numpy array): true positive rate

        """
        y_true = y_true.flatten(order='C')
        y_proba = y_proba.flatten(order='C')

        fpr, tpr, _ = roc_curve(y_true, y_proba)

        return fpr, tpr

    def precision_recall(self, y_true, y_proba):
        """
        Function that compute the precision and recall

        :arg
            self (Metrics): instance of the class
            y_true (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the true pathology
            y_proba (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the predicted probability pathology

        :return
            precision (numpy array): false positive rate
            recall (numpy array): true positive rate

        """
        y_true = y_true.flatten(order='C')
        y_proba = y_proba.flatten(order='C')

        precision, recall, _ = precision_recall_curve(y_true, y_proba)

        return precision, recall

    def cohen_kappa_score(self, y_true, y_pred):
        """
        Function that compute the kappa cohen score on a multiclass-multilabel problem

        :arg
            self (Metrics): instance of the class
            y_true (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the true pathology
            y_pred (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the predicted pathology

        :return
            mean_kappa_score (float): mean kappa cohen score of all the prediction done
            kappa_class (list): kappa cohen score of each class

        """
        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]

        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        kappa_class = []
        for i in range(y_true.shape[1]):
            kappa_class.append(cohen_kappa_score(y_true[:, i], y_pred[:, i]))

        y_true = y_true.flatten(order='C')
        y_pred = y_pred.flatten(order='C')

        mean_kappa_score = cohen_kappa_score(y_true, y_pred)

        return mean_kappa_score, kappa_class

    def f1_score(self, y_true, y_pred):
        """
        Function that compute the f1 score on a multiclass-multilabel problem

        :arg
            self (Metrics): instance of the class
            y_true (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the true pathology
            y_pred (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the predicted pathology

        :return
            mean_f1 (float): mean f1 score of all the prediction done
            f1_class (list): f1 score score of each class

        """
        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]

        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        f1_class = []
        for i in range(y_true.shape[1]):
            f1_class.append(f1_score(y_true[:, i], y_pred[:, i]))

        y_true = y_true.flatten(order='C')
        y_pred = y_pred.flatten(order='C')

        mean_f1 = f1_score(y_true, y_pred)

        return mean_f1, f1_class

    def precision(self, y_true, y_pred):
        """
        Function that compute the precision score on a multiclass-multilabel problem

        :arg
            self (Metrics): instance of the class
            y_true (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the true pathology
            y_pred (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the predicted pathology

        :return
            mean_precision (float): mean precision score of all the prediction done
            precision_class (list): precision score score of each class

        """
        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]

        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        precision_class = []
        for i in range(y_true.shape[1]):
            precision_class.append(precision_score(y_true[:, i], y_pred[:, i]))

        y_true = y_true.flatten(order='C')
        y_pred = y_pred.flatten(order='C')

        mean_precision = precision_score(y_true, y_pred)

        return mean_precision, precision_class

    def recall(self, y_true, y_pred):
        """
        Function that compute the recall score on a multiclass-multilabel problem

        :arg
            self (Metrics): instance of the class
            y_true (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the true pathology
            y_pred (numpy array): 1D or 2D numpy array where each rows correspond to an image and each column correspond
                                  to the predicted pathology

        :return
            mean_recall (float): mean recall score of all the prediction done
            recall_class (list): recall score score of each class

        """
        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]

        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        recall_class = []
        for i in range(y_true.shape[1]):
            recall_class.append(recall_score(y_true[:, i], y_pred[:, i]))

        y_true = y_true.flatten(order='C')
        y_pred = y_pred.flatten(order='C')

        mean_recall = recall_score(y_true, y_pred)

        return mean_recall, recall_class
