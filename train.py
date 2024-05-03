import numpy as np
from classes import ImageManager
from classes import TensorFlowManager


all_batch_loaded = False
i=0
batch_size = 750

print(" Get images Data")
img_manager = ImageManager("./assets/datasets/")
images_data = img_manager.get_images_data("train")

print("Trains model")
tf_manager = TensorFlowManager("pneumonia")
tf_manager.train(images_data)


print("learning done")
