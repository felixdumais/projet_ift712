# -*- coding:utf-8 -*-

import argparse
from models.SVMClassifier import SVMClassifier
from models.LogisticRegressor import LogisticRegressor
from DataHandler import DataHandler
from sklearn.model_selection import train_test_split
from Metrics import Metrics
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

    classifier = 'LogisticRegressor'
    validation = 0.1
    learning_rate = 0.001
    predict = True
    verbose = True
    metrics = True

    image_path = '../data/sample/images_reduced_folder'
    label_full_path = '../data/sample/sample_labels.csv'
    classifier_type = 2
    random_seed = 10



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

    train_image_all, test_image_all, train_labels_all, test_labels_all = train_test_split(image_all, labels_all,
                                                                                          train_size=0.85,
                                                                                          random_state=random_seed)
    if classifier_type == 2:
        _, _, train_labels_bool, test_labels_bool = train_test_split(image_all, labels_bool,
                                                                     train_size=0.85,
                                                                     random_state=random_seed)

        train_image_sick = train_image_all[train_labels_bool == 1]
        train_labels_sick = train_labels_all[train_labels_bool == 1]
        train_labels_sick = np.delete(train_labels_sick, 5, axis=1)
        test_image_sick = test_image_all[test_labels_bool == 1]
        test_labels_sick = test_labels_all[test_labels_bool == 1]
        test_labels_sick = np.delete(test_labels_sick, 5, axis=1)

    if classifier == 'SVM':
        if classifier_type == 1:
            model = SVMClassifier()
        elif classifier_type == 2:
            model1 = SVMClassifier()
            model2 = SVMClassifier()
    elif classifier == "LogisticRegressor":
        model = LogisticRegressor()
        model.train(train_image, train_labels)

    elif classifier == 'MLP':
        if classifier_type == 1:
            model = MLP()
        elif classifier_type == 2:
            model1 = MLP()
            model2 = MLP()
    # Do this with every models
    else:
        raise SyntaxError('Invalid model name')

    #model.train()
    predict_label = model.predict(test_image)
    #metrics = Metrics()
    #FPR = metrics.false_positive_rate(test_labels, predict_label)
    #FNR = metrics.false_negative_rate(test_labels, predict_label)
    #recall = metrics.recall(test_labels, predict_label)
    #precision = metrics.precision(test_labels, predict_label)
    #specificity = metrics.specificity(test_labels, predict_label)
    #accuracy = metrics.accuracy(test_labels, predict_label)
    #f_measure = metrics.f_measure(test_labels, predict_label)
    if classifier_type == 1:
        model.train(train_image_all, train_labels_all)
        pred = model.predict(test_image_all)
        proba = model.predict_proba(test_image_all)

        metrics = Metrics()

        fpr, tpr = metrics.roc_metrics(test_labels_all, proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show(block=False)

        precision, recall = metrics.precision_recall(test_labels_all, proba)
        plt.figure()
        plt.plot(precision, recall)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show(block=False)

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

        label_list = data.label_.columns.values.tolist()

        element = [titles] + list(
            zip(label_list, kappa_class_disp, f1_class_disp, accuracy_class_disp, precision_class_disp,
                recall_class_disp))
        for i, d in enumerate(element):
            line = '        |'.join(str(x).ljust(12) for x in d)
            print(line)
            if i == 0:
                print('-' * len(line))

    elif classifier_type == 2:
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

        metrics = Metrics()

        # fpr, tpr = metrics.roc_metrics(test_labels_bool, sick_bool_proba)
        # plt.figure()
        # plt.plot(fpr, tpr)
        # plt.title('ROC Curve (Sick vs Not sick)')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.show(block=False)

        precision, recall = metrics.precision_recall(test_labels_all, proba_matrix)
        plt.figure()
        plt.plot(precision, recall)
        plt.title('Precision-Recall Curve (2 classifiers)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show(block=False)

        fpr, tpr = metrics.roc_metrics(test_labels_all, proba_matrix)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC Curve (2 classifiers)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show(block=False)

        # precision, recall = metrics.precision_recall(test_labels_sick, sick_type_proba)
        # plt.figure()
        # plt.plot(precision, recall)
        # plt.title('Precision-Recall Curve (pathology classifier)')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.show(block=False)

        # cohen_kappa_score_bool, _ = metrics.cohen_kappa_score(test_labels_bool, sick_bool_pred)
        # f1_score_bool, _ = metrics.f1_score(test_labels_bool, sick_bool_pred)
        # accuracy_bool, _ = metrics.accuracy(test_labels_bool, sick_bool_pred)
        # precision_bool, _ = metrics.precision(test_labels_bool, sick_bool_pred)
        # recall_bool, _ = metrics.recall(test_labels_bool, sick_bool_pred)
        # #
        # print('Cohen (sick vs not): {}'.format(cohen_kappa_score_bool))
        # print('F1 (sick vs not): {}'.format(f1_score_bool))
        # print('Accuracy (sick vs not): {}'.format(accuracy_bool))
        # print('Precision (sick vs not): {}'.format(precision_bool))
        # print('Recall (sick vs not): {}'.format(recall_bool))

        cohen_kappa_score, kappa_class = metrics.cohen_kappa_score(test_labels_all, prediction_matrix)
        f1_score, f1_class = metrics.f1_score(test_labels_all, prediction_matrix)
        accuracy, accuracy_class = metrics.accuracy(test_labels_all, prediction_matrix)
        precision, precision_class = metrics.precision(test_labels_all, prediction_matrix)
        recall, recall_class = metrics.recall(test_labels_all, prediction_matrix)

        print('Cohen (2 classifiers): {}'.format(cohen_kappa_score))
        print('F1 (2 classifiers): {}'.format(f1_score))
        print('Accuracy (2 classifiers): {}'.format(accuracy))
        print('Precision (2 classifiers): {}'.format(precision))
        print('Recall (2 classifiers): {}'.format(recall))

        titles = ['names', 'Cohen', 'F1_score', 'Accuracy', 'Precision', 'Recall']
        kappa_class_disp = ['%.4f' % elem for elem in kappa_class]
        f1_class_disp = ['%.4f' % elem for elem in f1_class]
        accuracy_class_disp = ['%.4f' % elem for elem in accuracy_class]
        precision_class_disp = ['%.4f' % elem for elem in precision_class]
        recall_class_disp = ['%.4f' % elem for elem in recall_class]

        label_list = data.label_.columns.values.tolist()

        element = [titles] + list(
            zip(label_list, kappa_class_disp, f1_class_disp, accuracy_class_disp, precision_class_disp,
                recall_class_disp))
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