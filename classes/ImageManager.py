from PIL import Image
import numpy as np
import struct
import matplotlib.pyplot as plt

import os
from pathlib import Path

class ImageManager:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def convert(self, image_sub_path):
        image_path = self.folder_path + "/" + image_sub_path

        with Image.open(image_path) as img:
            gray_image = img.convert("L")
            # gray_image = abs(np.array(gray_img).astype(np.int16)-255)

            return gray_image

    
    def print_image(self, dataset):
                
        plt.style.use('_mpl-gallery-nogrid')
        fig, ax = plt.subplots()
        
        ax.imshow(dataset)
        plt.savefig('./.pictures/radio.png')
        plt.show()

    def load_batch(self, dataset_type_folder):
        images_batch = {
            "images": [],
            "labels": []
        }

        print("Load normal batch")
        # Normal
        current_path = dataset_type_folder + "/NORMAL/"
        folder_path = Path( self.folder_path + "/" + current_path)
        
        file_list = os.listdir(folder_path)

        i = 0

        for file in file_list:
            print(f"{i}/{len(file_list)}\r", end='')
            if(file != ".DS_Store"):
                i += 1
                img = self.convert(current_path + file)
                images_batch["images"].append(np.array(img))
                images_batch["labels"].append("normal")
        
        print("\n batch loaded.")

        return images_batch

    def load_batch_p(self, dataset_type_folder):
        images_batch = {
            "images": [],
            "labels": []
        }

        # Pneumonia
        print("Load pneumonia batch")

        current_path = dataset_type_folder + "/PNEUMONIA/"
        folder_path = Path( self.folder_path + "/" + current_path)
        
        
        file_list = os.listdir(folder_path)

        i = 0

        
        for file in file_list:
            print(f"{i}/{len(file_list)}\r", end='')
            if(file != ".DS_Store"):
                i+=1
                img = self.convert(current_path + file)
                images_batch["images"].append(np.array(img))
                images_batch["labels"].append(file.split("_")[1])

        print("\n batch loaded.")

        return images_batch


            

