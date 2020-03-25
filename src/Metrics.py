import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
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
        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]

        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        accuracy_class = []
        for i in range(y_true.shape[1]):
            accuracy_class.append(accuracy_score(y_true[:, i], y_pred[:, i]))

        mean_accuracy = accuracy_score(y_true, y_pred)

        return mean_accuracy, accuracy_class

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]

        for i, label in enumerate(class_names):
            fig, ax = plt.subplots(1, 2)
            j = 0
            for title, normalize in titles_options:
                confusion_matrix_res = confusion_matrix(y_true[:, i], y_pred[:, i], normalize=normalize)
                disp = ConfusionMatrixDisplay(confusion_matrix_res,
                                             display_labels=['Negative', 'Positive'])
                disp.plot(cmap=plt.cm.Blues, ax=ax[j])
                fig.suptitle(label)
                ax[j].set_title(title)
                j += 1

    def roc_metrics(self, y_true, y_proba):
        y_true = y_true.flatten(order='C')
        y_proba = y_proba.flatten(order='C')

        fpr, tpr, _ = roc_curve(y_true, y_proba)

        return fpr, tpr

    def precision_recall(self, y_true, y_proba):
        y_true = y_true.flatten(order='C')
        y_proba = y_proba.flatten(order='C')

        precision, recall, _ = precision_recall_curve(y_true, y_proba)

        return precision, recall

    def cohen_kappa_score(self, y_true, y_pred):
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
