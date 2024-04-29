import numpy as np
from classes import ImageManager
from classes import TensorFlowManager

img_manager = ImageManager("./assets/datasets/")

all_batch_loaded = False
i=0
batch_size= 10

while not all_batch_loaded:
    print(f"batch #{i}")
    (train_batch, ended) = img_manager.load_batch("train", batch_size, i)
    
    tf_manager = TensorFlowManager()
    tf_manager.learning(train_batch)

    i += 1
    all_batch_loaded = ended


print("learning done")
