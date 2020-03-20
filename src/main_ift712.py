# -*- coding:utf-8 -*-

import argparse
from src.models.SVMClassifier import SVMClassifier
from src.models.MLP import MLP

from src.DataHandler import DataHandler
from sklearn.model_selection import train_test_split
from src.Metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
import os

'''
Cours IFT712, projet de session
Auteurs:
    Félix Dumais (14053686)
    Joëlle Fréchette-Viens
    Nicolas Fontaine
'''

def argument_parser():
    parser = argparse.ArgumentParser(usage='\n python3 main_ift712.py [model]',
                                     description="")
    parser.add_argument('--model', type=str, default="SVM",
                        choices=["SVM", "MLP", "RandomForest", "LogisticRegressor", "all"])
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--predict', default=None,
                        help="Selected a saved trained model")
    parser.add_argument('--metrics', action='store_true',
                        help='Display different metrics to the user')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()

def main():
    #args = argument_parser()

    #classifier = args.model
    #validation = args.validation
    #learning_rate = args.lr
    #predict = args.predict
    #verbose = args.verbose

    classifier = 'SVM'
    validation = 0.1
    learning_rate = 0.001
    predict = True
    verbose = True
    metrics = True
    image_path = '../data/sample/images'
    label_full_path = '../data/sample/sample_labels.csv'


    if verbose:
        print('Formatting dataset...')
    data = DataHandler(image_path=image_path, label_full_path=label_full_path)
    image_all, labels_all = data.get_all_data()
    _, labels_bool = data.get_sick_bool_data()
    image_sick, labels_sick = data.get_only_sick_data()
    data.plot_data()
    data.show_samples()


    if verbose:
        print('Training of the model...')

    train_image, test_image, train_labels, test_labels = train_test_split(image_all, labels_all, train_size=0.8, random_state=10)

    if classifier == 'SVM':
        model = SVMClassifier(train_image, test_image, train_labels, test_labels, loss='hinge')

    elif classifier == 'MLP':
        model = MLP(train_image, test_image, train_labels, test_labels, loss=None)
    # Do this with every models
    else:
        raise SyntaxError('Invalid model name')

    model.train()
    clf = model.get_model()
    clf_proba = clf.predict_proba(test_image)
    clf_pred = clf.predict(test_image)
    #predict_label = model.predict()
    metrics = Metrics()

    fpr, tpr = metrics.roc_metrics(test_labels, clf_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show(block=False)

    precision, recall = metrics.precision_recall(test_labels, clf_proba)
    plt.figure()
    plt.plot(precision, recall)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show(block=False)

    cohen_kappa_score, kappa_class = metrics.cohen_kappa_score(test_labels, clf_pred)
    f1_score, f1_class = metrics.f1_score(test_labels, clf_pred)
    accuracy, accuracy_class = metrics.accuracy(test_labels, clf_pred)
    precision, precision_class = metrics.precision(test_labels, clf_pred)
    recall, recall_class = metrics.recall(test_labels, clf_pred)

    print('Cohen: {}'.format(cohen_kappa_score))
    print('F1: {}'.format(f1_score))
    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))

    titles = ['names', 'Cohen', 'F1_score', 'Accuracy', 'Precision', 'Recall']
    kappa_class = ['%.4f' % elem for elem in kappa_class]
    f1_class = ['%.4f' % elem for elem in f1_class]
    accuracy_class = ['%.4f' % elem for elem in accuracy_class]
    precision_class = ['%.4f' % elem for elem in precision_class]
    recall_class = ['%.4f' % elem for elem in recall_class]
    element = [titles] + list(zip(data.label_.columns.values.tolist(), kappa_class, f1_class, accuracy_class, precision_class, recall_class))
    for i, d in enumerate(element):
        line = '        |'.join(str(x).ljust(12) for x in d)
        print(line)
        if i == 0:
            print('-' * len(line))

    # class_names = data.label_.columns.values.tolist()
    # metrics.plot_confusion_matrix(test_labels, svm_pred, class_names)
    plt.show()


if __name__ == '__main__':
    main()