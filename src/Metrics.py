import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve

class Metrics:
    def __init__(self):
        pass

    def accuracy(self, y_true, y_pred):
        TP, TN, FP, FN = self.confusion_matrix(y_true, y_pred)

        accuracy = (TP + TN) / (TP + TN + FP + FN)

        return accuracy

    def false_positive_rate(self, y_true, y_pred):
        TP, TN, FP, FN = self.confusion_matrix(y_true, y_pred)

        FPR = FP / (TN + FP)

        return FPR

    def false_negative_rate(self, y_true, y_pred):
        TP, TN, FP, FN = self.confusion_matrix(y_true, y_pred)

        FNR = FN / (FN + TP)

        return FNR

    def recall(self, y_true, y_pred):
        TP, TN, FP, FN = self.confusion_matrix(y_true, y_pred)

        recall = TP / (FN + TP)

        return recall

    def precision(self, y_true, y_pred):
        TP, TN, FP, FN = self.confusion_matrix(y_true, y_pred)

        precision = TP / (FP + TP)

        return precision

    def specificity(self, y_true, y_pred):
        TP, TN, FP, FN = self.confusion_matrix(y_true, y_pred)

        specificity = TN / (TN + FP)

        return specificity

    def f_measure(self, y_true, y_pred):
        recall = self.recall(y_true, y_pred)
        precision = self.precision(y_true, y_pred)

        f_measure = (2*recall*precision) / (precision + recall)

        return f_measure

    def plot_confusion_matrix(self, classifier, X_test, y_test, class_names):
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, X_test, y_test,
                                         display_labels=class_names,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

    def roc_metrics(self, y_true, y_proba):
        y_true = y_true.flatten(order='C')
        y_proba = y_proba.flatten(order='C')

        fpr, tpr, _ = roc_curve(y_true, y_proba)

        return fpr, tpr

    def plot_prec_rec(self, precision: list, recall: list):
        plt.figure()
        plt.title('Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        plt.plot(recall, precision, 'bo--')
        plt.show(block=False)
