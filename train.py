import numpy as np
import math
import matplotlib.pyplot as plt
import os

from classes import ImageManager
from classes import TensorFlowManager

print(" Get images Data")
img_manager = ImageManager("./assets/datasets/")
train_data = img_manager.get_images_data("train")
validation_data = img_manager.get_images_data("val")

batch_size = 4096
epochs = 1
shuffle = len(train_data["paths"])
repeat = 1
shape = 224


print("Trains model")
tf_manager = TensorFlowManager("pneumonia", batch_size=batch_size, epochs=epochs, shuffle=shuffle, repeat=repeat, shape=shape)
loss_history, accuracy_history, val_loss_history, val_accuracy_history = tf_manager.train(train_data, validation_data)

print("Training done")