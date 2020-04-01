# -*- coding:utf-8 -*-

import argparse
from src.models.SVMClassifier import SVMClassifier
from src.models.MLP import MLP
from src.models.RBF import RBFClassifier
from src.DataHandler import DataHandler
from sklearn.model_selection import train_test_split
from src.Metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import copy

'''
Cours IFT712, projet de session
Auteurs:
    Félix Dumais (14053686)
    Joëlle Fréchette-Viens (15057894)
    Nicolas Fontaine
'''


def display_metrics(classifier_list: list, test_labels_all, pred: list, proba: list, label_list: list):
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
        line = '        |'.join(str(x).ljust(12) for x in d)
        print(line)
        if i == 0:
            print('-' * len(line))

def argument_parser():
    parser = argparse.ArgumentParser(usage='\n python3 main_ift712.py [model]',
                                     description="")
    parser.add_argument('--model', type=str, default="SVM",
                        choices=["SVM", "MLP", "RBF", "RandomForest", "LogisticRegressor", "all"])
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

    classifier = 'RBF'#'SVM'
    verbose = True
    image_path = '../data/sample/images'
    label_full_path = '../data/sample/sample_labels.csv'
    classifier_type = 2
    random_seed = 10

    if verbose:
        print('Formatting dataset...')
    data = DataHandler(image_path=image_path, label_full_path=label_full_path, resampled_width=32, resampled_height=32)
    image_all, labels_all = data.get_all_data()
    _, labels_bool = data.get_sick_bool_data()
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

    if classifier == 'SVM':
        classifier_list = ['SVM']
        if classifier_type == 1:
            model = SVMClassifier(cv=False)
        elif classifier_type == 2:
            model1 = SVMClassifier(cv=False)
            model2 = SVMClassifier(cv=False)

    elif classifier == 'MLP':
        classifier_list = ['MLP']
        if classifier_type == 1:
            model = MLP(cv=False)
        elif classifier_type == 2:
            model1 = MLP(cv=False)
            model2 = MLP(cv=False)

    elif classifier == 'RBF':
        classifier_list = ['RBF']
        if classifier_type == 1:
            model = RBFClassifier(cv=False)
        elif classifier_type == 2:
            model1 = RBFClassifier(cv=False)
            model2 = RBFClassifier(cv=False)

    elif classifier == 'all':
        classifier_list = ['SVM', 'MLP']
        if classifier_type == 1:
            model_SVM = SVMClassifier(cv=False)
            model_MLP = MLP(cv=False)
            model = [model_SVM, model_MLP]
        elif classifier_type == 2:
            model_SVM = SVMClassifier(cv=False)
            model_MLP = MLP(cv=False)
            model1 = [model_SVM, model_MLP]
            model2 = copy.deepcopy(model1)
    # Do this with every models
    else:
        raise SyntaxError('Invalid model name')

    if classifier_type == 1:
        if isinstance(model, list):
            pred = []
            proba = []
            for _, clf in enumerate(model):
                clf.train(train_image_all, train_labels_all)
                pred_clf = clf.predict(test_image_all)
                proba_clf = clf.predict_proba(test_image_all)
                pred.append(pred_clf)
                proba.append(proba_clf)
        else:
            model.train(train_image_all, train_labels_all)
            pred = [model.predict(test_image_all)]
            proba = [model.predict_proba(test_image_all)]

        label_list = data.label_.columns.values.tolist()
        display_metrics(classifier_list, test_labels_all, pred, proba, label_list)

    elif classifier_type == 2:
        if isinstance(model1, list) and isinstance(model2, list):
            pred = []
            proba = []
            for i, clf1 in enumerate(model1):
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

        label_list = data.label_.columns.values.tolist()
        display_metrics(classifier_list, test_labels_all, pred, proba, label_list)

    plt.show()


if __name__ == '__main__':
    main()