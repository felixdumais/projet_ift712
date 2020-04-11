# -*- coding:utf-8 -*-

import argparse
from DataHandler import DataHandler
import matplotlib.pyplot as plt
from Trainer import Trainer
import time

'''
Cours IFT712, projet de session
Auteurs:
    Félix Dumais (14053686)
    Joëlle Fréchette-Viens (15057894)
    Nicolas Fontaine
'''

def argument_parser():
    """
    Function instanciate the class ArgumentParser so we parse argument through the terminal

    :arg
        None

    :return
        parser.parse_args() (argparse.Namespace): Return namespaces pass with argumentparser
    """
    parser = argparse.ArgumentParser(usage='\n python main_ift712.py'
                                           '\n python main_ift712.py [model]'
                                           '\n python main_ift712.py [model] [train_size]'
                                           '\n python main_ift712.py [model] [train_size] [cv]'
                                           '\n python main_ift712.py [model] [train_size] [cv] [classifier_type]'
                                           '\n python main_ift712.py [model] [train_size] [cv] [classifier_type] [verbose]',
                                     description="To use the program, no argument are mandatory.")
    parser.add_argument('--model', type=str, default="SVM",
                        help='DEFAULT: SVM --> Model to train. If all is selected all the models are trained.',
                        choices=["SVM", "Fisher", "MLP", "RBF", "RandomForest", "LogisticRegressor", "all"])
    parser.add_argument('--train_size', type=float, default=0.85,
                        help='DEFAULT: 0.85 --> Percentage of the dataset used for training')
    parser.add_argument('--cv', type=bool, default=False,
                        help='DEFAULT: False --> Use CV to do k-fold cross validation')
    parser.add_argument('--classifier_type', type=int, default=1,
                        help='DEFAULT: 1 --> Use CV to do k-fold cross validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='To have some feed back from the program.')
    parser.add_argument('--clf1', type=str, default=None,
                        help='DEFAULT: None --> Specify saved classifier.')
    parser.add_argument('--clf2', type=str, default=None,
                        help='DEFAULT: None --> Specify saved classifier.')

    return parser.parse_args()


def main():
    args = argument_parser()

    # Get variables pass with argument parser
    # classifier = args.model
    # verbose = args.verbose
    # cross_validation = args.cv
    # classifier_type = args.classifier_type
    # train_size = args.train_size
    # clf1 = args.clf1
    # clf2 = args.clf2

    # Define the path of the images and the targets CSV folder
    image_path = '../data/sample/images'
    label_full_path = '../data/sample/sample_labels.csv'

    classifier = 'MLP'
    verbose = False
    classifier_type = 2
    cross_validation = False
    train_size = 0.85
    clf1 = None
    clf2 = None

    if classifier_type != 1 and classifier_type != 2:
        raise OSError('Wrong classifier type. Classifier type must be either 1 or 2')

    if train_size <= 0 or train_size >= 1:
        raise OSError('Wrong train size. Train size must be exclusively between 0 and 1')

    if verbose:
        print('Starting training ...')
        print('Classifier: {}'.format(classifier))
        print('Type of classifier: {}'.format(classifier_type))
        print('Cross validation: {}'.format(cross_validation))
        print('Train size: {}'.format(train_size))

    if verbose:
        print('Formatting dataset...')

    # Instanciate DataHandler
    data = DataHandler(image_path=image_path, label_full_path=label_full_path, resampled_width=128, resampled_height=128)

    # Plot data and samples from the dataset
    data.plot_data()
    data.show_samples()

    if verbose:
        print('Training of the model...')

    # Instanciate Trainer variable
    trainer = Trainer(cross_validation, data, classifier_type, train_size, classifier)
    if clf1 is not None or clf2 is not None:
        # Prediction with already saved models
        trainer.predict_with_saved_model(clf1, clf2)


    else:
        # Training on images
        start = time.time()
        trainer.training()
        end = time.time()
        print(end - start)

    plt.show()

if __name__ == '__main__':
    main()
