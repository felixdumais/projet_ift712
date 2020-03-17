import numpy as np
import matplotlib.pyplot as plt

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

    def confusion_matrix(self, y_true, y_pred):
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

        return TP, TN, FP, FN

    def plot_ROC(self, recall: list, FPR: list):
        plt.figure()
        plt.title('ROC curve')
        plt.xlabel('False positive rate')
        plt.ylabel('Recall')

        plt.plot(FPR, recall, 'ro--')
        plt.show(block=False)

    def plot_prec_rec(self, precision: list, recall: list):
        plt.figure()
        plt.title('Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        plt.plot(recall, precision, 'bo--')
        plt.show(block=False)
