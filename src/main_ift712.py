# -*- coding:utf-8 -*-

import argparse
from src.models.SVM_classifier import SVMClassifier
from src.DataHandler import DataHandler
from sklearn.model_selection import train_test_split

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
                        choices=["SVM", "MLP", "RandomForest", "LogisticRegressor"])
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
    predict = False
    verbose = True
    image_path = '../data/sample/images_reduced_folder'
    label_full_path = '../data/sample_labels.csv'

    if verbose:
        print('Formatting dataset...')
    data = DataHandler(image_path=image_path, label_full_path=label_full_path)
    image, labels = data.get_data()

    if verbose:
        print('Training of the model...')

    train_image, test_image, train_labels, test_labels = train_test_split(image, labels, train_size=0.8, random_state=10)
    train_data = zip(train_image, train_labels)
    test_data = zip(test_image, test_labels)

    if classifier == 'SVM':
        model = SVMClassifier(train_data, test_data)
    # Do this with every models
    else:
        raise SyntaxError('Invalid model name')

    model.train()

    if predict:
        model.predict()

    # Add part where we display some metrics


if __name__ == '__main__':
    main()