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

    def load_batch(self, dataset_type_folder, batch_qty, batch_number):
        images_batch = {
            "images": [],
            "labels": []
        }

        ended = False

        print("Load normal batch")
        # Normal
        current_path = dataset_type_folder + "/NORMAL/"
        folder_path = Path( self.folder_path + "/" + current_path)
        
        file_list = os.listdir(folder_path)

        start = batch_qty*batch_number
        end =  batch_qty*batch_number + batch_qty

        if end >= len(file_list):
            end = len(file_list)

        if start < len(file_list) - 1:  
            print("loading normal")

            i = 0
            for file in file_list[start : end]:
                # print(f"{i}/{len(file_list)}\r", end='')
                print(f"{i}/{batch_qty}\r", end='')

                i += 1

                if(file != ".DS_Store"):
                    img = self.convert(current_path + file)
                    
                    images_batch["images"].append(img.tolist())
                    images_batch["labels"].append(0)

            ended = False
        else:
            ended = True

        print("\nbatch loaded.")

        # Pneumonia
        print("Load pneumonia batch")

        current_path = dataset_type_folder + "/PNEUMONIA/"
        folder_path = Path( self.folder_path + "/" + current_path)
        
        
        file_list = os.listdir(folder_path)

        end =  batch_qty*batch_number + batch_qty

        if end >= len(file_list):
            end = len(file_list)


        if start < len(file_list) - 1:  
            print("loading pneumonia")

            i = 0
            for file in file_list[start : end]:
                # print(f"{i}/{len(file_list)}\r", end='')
                print(f"{i}/{batch_qty}\r", end='')
                
                i+=1

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


            

