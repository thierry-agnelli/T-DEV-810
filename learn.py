import numpy as np
from classes import ImageManager
from classes import TensorFlowManager

img_manager = ImageManager("./assets")

train_batch_normal = img_manager.load_batch("train")
train_batch_pneumonia = img_manager.load_batch_p("train")


print(train_batch)