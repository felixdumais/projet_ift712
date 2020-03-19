# -*- coding:utf-8 -*-

import argparse
from src.models.SVMClassifier import SVMClassifier
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
    image_path = '../data/sample/images_reduced_folder'
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
    # Do this with every models
    else:
        raise SyntaxError('Invalid model name')

    model.train()
    svm = model.get_model()
    svm_proba = svm.predict_proba(test_image)
    #predict_label = model.predict()
    metrics = Metrics()

    fpr, tpr = metrics.roc_metrics(test_labels, svm_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    class_names = data.labels_
    metrics.plot_confusion_matrix(model, test_image, test_labels, class_names)


if __name__ == '__main__':
    main()