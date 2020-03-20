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
    def __init__(self, label_full_path, image_path):
        self._download()
        self.label_ = None
        self.image_list_ = None
        self.label_full_path = label_full_path
        self.image_path = image_path
        self.all_data, self.sick_bool_data, self.only_sick_data = self._get_formated_data()

    def _download(self):
        """Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: The return value. True for success, False otherwise.

        .. _PEP 484:
            https://www.python.org/dev/peps/pep-0484/

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
        image_list, id_list = self._import_png_folder()
        image_list_resample = [self.resample_image(image, 64, 64) for image in image_list]
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
        return self.all_data

    def get_sick_bool_data(self):
        return self.sick_bool_data

    def get_only_sick_data(self):
        return self.only_sick_data

    def flatten(self, image: list):
        image = [x.flatten(order='C') for x in image]
        return image

    def resample_image(self, image, width: int, heigth: int):
        resampled = cv2.resize(image, dsize=(width, heigth), interpolation=cv2.INTER_NEAREST)
        return resampled

    def plot_data(self):
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

    def show_samples(self):
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
