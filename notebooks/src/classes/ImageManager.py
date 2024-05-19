from PIL import Image
import numpy as np
import struct
import matplotlib.pyplot as plt

import os
from pathlib import Path

class ImageManager:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_images_data(self, dataset_type_folder):
        images_data = {
            "paths": [],
            "labels": []
        }

        # Normal
        current_path = self.folder_path + dataset_type_folder
        folder_path = current_path + "/NORMAL/"
        
        file_list = os.listdir(folder_path)

        for file in file_list:
            if(file != ".DS_Store"):
                images_data["paths"].append(folder_path + file)
                images_data["labels"].append(0)

        folder_path = current_path + "/PNEUMONIA/"
        
        file_list = os.listdir(folder_path)

        for file in file_list:
            if(file != ".DS_Store"):
                images_data["paths"].append(folder_path + file)
                
                if file.split("_")[1] == "bacteria":
                    images_data["labels"].append(1)
                else:
                    images_data["labels"].append(2)


        return images_data
