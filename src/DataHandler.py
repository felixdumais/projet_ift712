import os
import matplotlib.image as mpimg
import pandas as pd

class DataHandler:
    def __init__(self, label_full_path, image_path):
        self.label_full_path = label_full_path
        self.image_path = image_path
        self.data = self._get_formated_data()



    def _import_png_folder(self):
        image_list = []
        id_list = []
        for file in os.listdir(self.image_path):
            if file.endswith(".png"):
                full_file_path = os.path.join(self.image_path, file)
                image = mpimg.imread(full_file_path)
                image_list.append(image)
                id_list.append(file)

        return image_list, id_list


    def _import_csv(self):
        try:
            df = pd.read_csv(self.label_full_path)
        except IOError as e:
            raise e


    def _get_formated_data(self):
        image_list, id_list = self._import_png_folder()
        labels = self._import_csv()









