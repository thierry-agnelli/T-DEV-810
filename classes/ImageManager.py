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
            gray_image = abs(np.array(gray_image).astype(np.int16))

            return gray_image

    
    def print_image(self, dataset):
                
        plt.style.use('_mpl-gallery-nogrid')
        fig, ax = plt.subplots()
        
        ax.imshow(dataset)
        plt.savefig('./.pictures/radio.png')
        plt.show()

    def load_batch(self, dataset_type_folder, batch_size = None, batch_number=0):
        images_batch = {
            "images": [],
            "labels": []
        }

        ended = False

        # Normal
        current_path = dataset_type_folder + "/NORMAL/"
        folder_path = Path( self.folder_path + "/" + current_path)
        
        file_list = os.listdir(folder_path)    
        
        if batch_size == None:
            start = 0
            end = len(file_list)
        else:
            start = batch_size * batch_number
            end =  batch_size*batch_number + batch_size

        if end > len(file_list):
            end = len(file_list)

        qty = end - start

        if start < len(file_list) - 1:
            print("Load normal batch")

            # print("loading normal")

            i = 0
            for file in file_list[start : end]:
                i += 1
                print(f" {i}/{qty}      ", end='\r')

                if(file != ".DS_Store"):
                    img = self.convert(current_path + file)
                    
                    images_batch["images"].append(img.tolist())
                    images_batch["labels"].append(0)

            print("\nbatch loaded.")

            ended = False
        else:
            ended = True


        # Pneumonia
        current_path = dataset_type_folder + "/PNEUMONIA/"
        folder_path = Path( self.folder_path + "/" + current_path)
        
        
        file_list = os.listdir(folder_path)

        if batch_size == None:
            end = len(file_list)
        else:
            end =  batch_size*batch_number + batch_size

        if end > len(file_list):
            end = len(file_list)

        qty = end - start

        if start < len(file_list) - 1:  
            print("Load pneumonia batch")

            # print("loading pneumonia")

            i = 0
            for file in file_list[start : end]:
                i += 1
                print(f" {i}/{qty}      ", end='\r')
                

                if(file != ".DS_Store"):
                    img = self.convert(current_path + file)
                    images_batch["images"].append(img.tolist())
                    if file.split("_")[1] == "bacteria":
                        images_batch["labels"].append(1)
                    else:
                        images_batch["labels"].append(2)

            print("\n batch loaded.")

            ended = False
        else:
            ended = True

        return (images_batch , ended)


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
