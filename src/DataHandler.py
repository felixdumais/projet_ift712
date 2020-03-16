import os
import matplotlib.image as mpimg
import pandas as pd
from src.utils import unique
from tqdm import tqdm
import numpy as np
import cv2


class DataHandler:
    def __init__(self, label_full_path, image_path):
        self.label_full_path = label_full_path
        self.image_path = image_path
        self.data = self._get_formated_data()

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

                    image = self.resample_image(image, 128, 128)
                    image_list.append(image)
                    id_list.append(file)

                pbar.update(1)

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
        labels = self._import_csv()

        labels = labels[labels.iloc[:, 0].isin(id_list)]
        labels['ordered_id'] = pd.Categorical(labels.iloc[:, 0], categories=id_list, ordered=True)
        labels.sort_values('ordered_id')
        labels = labels.drop(labels='ordered_id', axis=1)
        labels = labels.iloc[:, 1:]

        image_list = self.flatten(image_list)
        image_flatten_array = np.asarray(image_list)
        labels = np.asarray(labels)

        return image_flatten_array, labels

    def get_data(self):
        return self.data

    def flatten(self, image: list):
        image = [x.flatten(order='C') for x in image]
        return image

    def resample_image(self, image, width: int, heigth: int):
        resampled = cv2.resize(image, dsize=(width, heigth), interpolation=cv2.INTER_NEAREST)
        return resampled














