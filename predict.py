import numpy as np
import math

from classes import ImageManager
from classes import TensorFlowManager


img_manager = ImageManager("./assets/datasets/")
images_data = img_manager.get_images_data("test")

tf_manager = TensorFlowManager("pneumonia")
predictions = tf_manager.predictions_by_batch(images_data)

comparaisons = np.array(predictions) == np.array(images_data["labels"])

succes_rate = math.floor(np.sum(comparaisons)/len(images_data["labels"])*10000)/100
print(succes_rate, "%")


# print(predictions)
# print(images_data["labels"])
