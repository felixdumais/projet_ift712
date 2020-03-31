import os
import matplotlib.image as mpimg
import pandas as pd
from src.utils import unique
from tqdm import tqdm
import numpy as np
import cv2
import zipfile
import shutil
import copy
import matplotlib.pyplot as plt

class DataHandler:
    def __init__(self, label_full_path, image_path, resampled_width=64, resampled_height=64):
        self._download()
        self.label_ = None
        self.image_list_ = None
        self.label_full_path = label_full_path
        self.image_path = image_path
        self.resampled_width = resampled_width
        self.resampled_height = resampled_height
        self.all_data, self.sick_bool_data, self.only_sick_data = self._get_formated_data()

    def _download(self):
        """
        Function that install the dataset in the data directory. It downloads the dataset with the kaggle API and unzip
        the zip file downloaded (sample.zip.)

        :arg
            self (DataHandler): instance of the class

        :return
            None
        """
        check_dir = '../data/sample'
        if os.path.isdir(check_dir):
            print('Files already downloaded')
            return None

        os.system('kaggle datasets download -d nih-chest-xrays/sample')

        save_directory = '../data'
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        zip_file = 'sample.zip'

        print('Extracting zip file ...')
        with zipfile.ZipFile(zip_file, 'r') as zip_obj:
            zip_obj.extractall(save_directory)

        os.remove(zip_file)
        shutil.rmtree('../data/sample/sample')

        print('File extracted')

    def _import_png_folder(self):
        """
        Function that downloaded images in the images folder and places them in a list.

        :arg
            self (DataHandler): instance of the class

        :return
            image_list (list): List or images. The format of images are array
            id_list (list): List of string corresponding to the name of each image
        """
        image_list = []
        id_list = []
        with tqdm(total=len(os.listdir(self.image_path))) as pbar:
            for file in os.listdir(self.image_path):
                if file.endswith(".png"):
                    full_file_path = os.path.join(self.image_path, file)
                    image = mpimg.imread(full_file_path)
                    if image.ndim > 2:
                        image = image[:, :, 0]

                    image_list.append(image)
                    id_list.append(file)

                pbar.update(1)

        self.image_list_ = image_list
        return image_list, id_list

    def _import_csv(self):
        """
        Function that read the csv in which the target of each image is written.

        :arg
            self (DataHandler): instance of the class

        :return
            df (pandas dataframe): pandas dataframe corresponding to the targets. Each columns correspond to a pathology
                                   each rows correspond to an image.
        """
        try:
            df = pd.read_csv(self.label_full_path)
        except IOError as e:
            raise e

        df = df.iloc[:, [0, 1]]

        split_data = df['Finding Labels'].str.split('|')
        list1 = split_data.to_list()
        flat_list = [item for sublist in list1 for item in sublist]
        unique_list = unique(flat_list)

        df = pd.concat([df, pd.DataFrame(columns=unique_list)], sort=False)

        for value in unique_list:
            bool_value = df['Finding Labels'].str.contains(value)
            df[value] = bool_value.astype(int)

        df = df.drop(labels=['Finding Labels'], axis=1)

        return df

    def _get_formated_data(self):
        """
        Function that format the data in numpy array

        :arg
            self (DataHandler): instance of the class

        :return
            (image_flatten_array, all_labels) (tuple of numpy array):
                image_flatten_array (numpy array): 2D numpy array, each column is a normalized pixel value (0, 1) and
                                                   row correspond to a new image
                all_labels (numpy array): 2D binary numpy array, each column correspond to a pathology including the
                                          No Finding as a pathology and each row correspond to a new image
            (image_flatten_array, bool_sick_labels) (tuple of numpy array):
                image_flatten_array (numpy array): 2D numpy array, each column is a normalized pixel value (0, 1) and
                                                   row correspond to a new image
                bool_sick_labels (numpy array): 1D binary numpy array, each value correspond to either 0 or 1. If the
                                                value is 1, it means that the image corresponding to the index is the
                                                image of a sick patient. 0 means a healthy patient
            (only_sick_image, only_sick_labels) (tuple of numpy array):
                only_sick_image (numpy array): 2D numpy array, each column is a normalized pixel value (0, 1) and
                                               row correspond to a new image. Only the sick patient are represented
                                               in this variable
                only_sick_labels (numpy array): 2D binary numpy array, each column correspond to a pathology. No Finding
                                                is not included as a pathology. Each row correspond to a new image. It
                                                only represent the pathology of sick patients.
        """
        image_list, id_list = self._import_png_folder()
        image_list_resample = [self.resample_image(image, self.resampled_width, self.resampled_height) for image in image_list]
        labels = self._import_csv()

        labels = labels[labels.iloc[:, 0].isin(id_list)]
        labels['ordered_id'] = pd.Categorical(labels.iloc[:, 0], categories=id_list, ordered=True)
        labels.sort_values('ordered_id')
        labels = labels.drop(labels='ordered_id', axis=1)
        labels = labels.iloc[:, 1:]

        image_list_flat = self.flatten(image_list_resample)
        image_flatten_array = np.asarray(image_list_flat)

        all_labels = copy.deepcopy(labels)
        all_labels = np.asarray(all_labels)

        bool_sick_labels = copy.deepcopy(labels)
        bool_sick_labels = np.asarray(bool_sick_labels)
        bool_sick_labels = 1 - bool_sick_labels[:, 5]

        only_sick_labels = copy.deepcopy(labels)
        only_sick_labels = np.asarray(only_sick_labels)
        sick_idx = np.nonzero(bool_sick_labels)
        only_sick_labels = only_sick_labels[sick_idx[0], :]
        only_sick_labels = np.delete(only_sick_labels, 5, axis=1)

        only_sick_image = image_flatten_array[sick_idx[0], :]
        self.label_ = labels
        return (image_flatten_array, all_labels),\
               (image_flatten_array, bool_sick_labels),\
               (only_sick_image, only_sick_labels)

    def get_all_data(self):
        """
        Function that return the instance variable self.all_data

        :arg
            self (DataHandler): instance of the class

        :return
            self.all_data (tuple of numpy array):
                self.all_data[0] (numpy array): 2D numpy array, each column is a normalized pixel value (0, 1) and
                                                   row correspond to a new image
                self.all_data[1] (numpy array): 2D binary numpy array, each column correspond to a pathology including
                                                the No Finding as a pathology and each row correspond to a new image

        """
        return self.all_data

    def get_sick_bool_data(self):
        """
        Function that return the instance variable self.sick_bool_data

        :arg
            self (DataHandler): instance of the class

        :return
            self.sick_bool_data (tuple of numpy array):
                self.sick_bool_data[0] (numpy array): 2D numpy array, each column is a normalized pixel value (0, 1) and
                                                   row correspond to a new image
                self.sick_bool_data[1] (numpy array): 1D binary numpy array, each value correspond to either 0 or 1. If
                                                the value is 1, it means that the image corresponding to the index is
                                                the image of a sick patient. 0 means a healthy patient

        """
        return self.sick_bool_data

    def get_only_sick_data(self):
        """
        Function that return the instance variable self.only_sick_data

        :arg
            self (DataHandler): instance of the class

        :return
            self.sick_bool_data (tuple of numpy array):
                self.sick_bool_data[0] (numpy array): 2D numpy array, each column is a normalized pixel value (0, 1) and
                                               row correspond to a new image. Only the sick patient are represented
                                               in this variable
                self.sick_bool_data[1] (numpy array): 2D binary numpy array, each column correspond to a pathology. No
                                                      Finding is not included as a pathology. Each row correspond to a
                                                      new image. It only represent the pathology of sick patients.

        """
        return self.only_sick_data

    def flatten(self, image: list):
        """
        Function that flatten a list of 2D numpy array

        :arg
            self (DataHandler): instance of the class
            image (list): list of 2D numpy array to be flatten

        :return
            image (list): list of 1D numpy array

        """
        image = [x.flatten(order='C') for x in image]
        return image

    def resample_image(self, image, width: int, heigth: int):
        """
        Function that resample a 2D numpy array

        :arg
            self (DataHandler): instance of the class
            image (numpy array): 2D numpy array
            width (int): width that the image will be resample
            heigth (int): heigth that the image will be resample

        :return
            resampled (numpy array): resampled 2D numpy array

        """
        resampled = cv2.resize(image, dsize=(width, heigth), interpolation=cv2.INTER_NEAREST)
        return resampled

    def plot_data(self):
        """
        Function that plot the bar plot of the pathology amongst patients including No Finding,
                           the bar plot excluding No Finding,
                           the bar plot os sick vs not sick

        :arg
            self (DataHandler): instance of the class

        :return
            value_hist_sick (list): list of the distrbution of sick people

        """
        array_labels = np.asarray(self.label_)

        value_hist = np.sum(array_labels, axis=0).tolist()
        label_names = self.label_.columns.values.tolist()

        value_hist, label_names = zip(*sorted(zip(value_hist, label_names)))

        # Dataset histogram
        y = np.arange(len(label_names))

        fig, ax = plt.subplots()
        rects = ax.barh(y, value_hist)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Number of people')
        ax.set_title('Distribution of the dataset')
        ax.set_yticks(y)
        ax.set_yticklabels(label_names)
        ax.legend()

        for rect in rects:
            width = rect.get_width()
            ax.annotate('{}'.format(width),
                        xy=(rect.get_x()+width, rect.get_y()),
                        xytext=(15, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


        plt.show(block=False)

        # Sick people histogram
        value_hist_sick = value_hist[:-1]
        label_names_sick = label_names[:-1]
        y = np.arange(len(label_names_sick))

        fig, ax = plt.subplots()
        rects = ax.barh(y, value_hist_sick)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Number of people')
        ax.set_title('Distribution of sick people')
        ax.set_yticks(y)
        ax.set_yticklabels(label_names_sick)
        ax.legend()

        for rect in rects:
            width = rect.get_width()
            ax.annotate('{}'.format(width),
                        xy=(rect.get_x()+width, rect.get_y()),
                        xytext=(15, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.show(block=False)

        # Sick vs not sick histogram
        sick_vs_not = [value_hist[-1], array_labels.shape[0] - value_hist[-1]]
        label_names_sick_not = ['Healthy', 'Sick']
        colors = ['g', 'r']
        y = np.arange(len(sick_vs_not))

        fig, ax = plt.subplots()
        rects = ax.barh(y, sick_vs_not, color=colors)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Number of people')
        ax.set_title('Sick vs Healthy people')
        ax.set_yticks(y)
        ax.set_yticklabels(label_names_sick_not)
        ax.legend()

        for rect in rects:
            width = rect.get_width()
            ax.annotate('{}'.format(width),
                        xy=(rect.get_x()+width/2, rect.get_y()),
                        xytext=(0, 50),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='center')

        plt.show(block=False)

        return value_hist_sick

    def show_samples(self):
        """
        Function that plot some sample of the image in the dataset

        :arg
            self (DataHandler): instance of the class

        :return
            None

        """
        idx = 0
        plt.figure()
        plt.rcParams['figure.figsize'] = (10.0, 10.0)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle('Samples of dataset')
        for image in self.image_list_[:16]:
            im = self.resample_image(image, 256, 256)
            plt.subplot(4, 4, idx + 1)  # .set_title(l)
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            idx += 1

        plt.show(block=False)
