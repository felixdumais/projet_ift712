# -*- coding:utf-8 -*-

import argparse
from DataHandler import DataHandler
import matplotlib.pyplot as plt
from Trainer import Trainer

'''
Cours IFT712, projet de session
Auteurs:
    Félix Dumais (14053686)
    Joëlle Fréchette-Viens (15057894)
    Nicolas Fontaine
'''

def argument_parser():
    parser = argparse.ArgumentParser(usage='\n python3 main_ift712.py [model]',
                                     description="")
    parser.add_argument('--model', type=str, default="SVM",
                        choices=["SVM", "Fisher", "MLP", "RBF", "RandomForest", "LogisticRegressor", "all"])
    parser.add_argument('--train_size', type=float, default=0.85,
                        help='Percentage of the dataset used for training')
    parser.add_argument('--cv', type=bool, default=False,
                        help='Use CV to do k-fold cross validation')
    parser.add_argument('--classifier_type', type=int, default=1,
                        help='Use CV to do k-fold cross validation')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def main():
    # args = argument_parser()
    #
    # classifier = args.model
    # verbose = args.verbose
    # cross_validation = args.cv
    # classifier_type = args.classifier_type
    # train_size = args.train_size


    image_path = '../data/sample/images_reduced_folder'
    label_full_path = '../data/sample/sample_labels.csv'
    random_seed = 10

    classifier = 'all'
    verbose = True
    classifier_type = 1
    cross_validation = False
    train_size = 0.85

    if classifier_type != 1 and classifier_type != 2:
        raise OSError('Wrong classifier type. Classifier type must be either 1 or 2')

    if train_size <= 0 or train_size >= 1:
        raise OSError('Wrong train size. Train size must be exclusively between 0 and 1')

    if verbose:
        print('Starting training ...')
        print('Classifier: {}'.format(classifier))
        print('Type of classifier: {}'.format(classifier_type))
        print('Cross validation: {}'.format(cross_validation))

    if verbose:
        print('Formatting dataset...')
    data = DataHandler(image_path=image_path, label_full_path=label_full_path, resampled_width=32, resampled_height=32)
    data.plot_data()
    data.show_samples()

    if verbose:
        print('Training of the model...')

    trainer = Trainer(cross_validation, data, classifier_type, train_size, classifier)
    trainer.training()

    plt.show()


if __name__ == '__main__':
    main()
