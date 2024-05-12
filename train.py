import numpy as np
from classes import ImageManager
from classes import TensorFlowManager

batch_size = 750

print(" Get images Data")
img_manager = ImageManager("./assets/datasets/")
train_data = img_manager.get_images_data("train")
validation_data = img_manager.get_images_data("val")


print("Trains model")
tf_manager = TensorFlowManager("pneumonia")
tf_manager.train(train_data, validation_data)


print("learning done")
