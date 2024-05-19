
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from .classes.ImageManager import ImageManager
from .classes.TensorFlowManager import TensorFlowManager

def comparaison(np_labels, np_predictions, mask):
    comparaisons =  np_labels[mask] == np_predictions[mask]
    succes_rate = math.floor(np.sum(comparaisons)/len(np_labels[mask])*10000)/100

    return succes_rate


def model_training(config):
    print(" Get images Data")
    img_manager = ImageManager("./src/assets/datasets/")
    train_data = img_manager.get_images_data("train")
    validation_data = img_manager.get_images_data("val")
    
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    shuffle = len(train_data["paths"])
    repeat = config["repeat"]
    shape = config["shape"]
    patience = config["patience"]


    print("Trains model")
    tf_manager = TensorFlowManager("pneumonia", batch_size=batch_size, epochs=epochs, shuffle=shuffle, repeat=repeat, shape=shape, patience=patience)
    loss_history, accuracy_history, val_loss_history, val_accuracy_history = tf_manager.train(train_data, validation_data)

    print("Training done")

    print("predictions")
    
    images_data = img_manager.get_images_data("test")

    tf_manager = TensorFlowManager("pneumonia", shape=shape)
    predictions = tf_manager.predictions_by_batch(images_data)

    np_labels = np.array(images_data["labels"])
    np_predictions = np.array(predictions)

    data = {
        "loss_history": loss_history,
        "accuracy_history": accuracy_history,
        "val_loss_history": val_loss_history,
        "val_accuracy_history": val_accuracy_history,
        "np_labels": np_labels,
        "np_predictions": np_predictions
    }

    return data

    