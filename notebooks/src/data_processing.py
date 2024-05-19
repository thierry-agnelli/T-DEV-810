import numpy as np
import math
import matplotlib.pyplot as plt
import os


from .classes.ImageManager import ImageManager
from .classes.TensorFlowManager import TensorFlowManager


def data_extraction(data):
    # loss_history = [
    #     15.172135353088379,
    #     14.172135353088379,
    #     13.172135353088379,
    #     12.172135353088379,
    #     11.172135353088379
    # ]
    # accuracy_history = [
    #         0.4334739148616791,
    #         0.5334739148616791,
    #         0.6334739148616791,
    #         0.7334739148616791,
    #         0.8334739148616791
    #     ]
    # val_loss_history = [
    #     28.31625747680664,
    #     27.31625747680664,
    #     26.31625747680664,
    #     25.31625747680664,
    #     24.31625747680664,
    #     23.31625747680664
    #     ]
    # val_accuracy_history = [
    #     0.5,
    #     0.4,
    #     0.3,
    #     0.2,
    #     0.1
    # ]
    # np_labels = np.array([
    #                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    #                 1, 1, 2, 2, 2, 1, 1, 1, 2, 1,
    #             ])
    # np_predictions = np.array([
    #                 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
    #                 0, 1, 0, 2, 0, 1, 2, 2, 1, 1,
    #                 ])
    
    loss_history = data["loss_history"]
    accuracy_history = data["accuracy_history"]
    val_loss_history = data["val_loss_history"]
    val_accuracy_history = data["val_accuracy_history"]
    np_labels = data["np_labels"]
    np_predictions = data["np_predictions"]

    return loss_history, accuracy_history, val_loss_history, val_accuracy_history, np_labels, np_predictions

def count_result(arr, index):
    return arr[index] if index < len(arr) else 0

def create_pie(labels, sizes, explode):
    fig, ax = plt.subplots()

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,      
        shadow=True,
        autopct='%1.1f%%'
    )


def train_history_chart(data):
    loss_history, accuracy_history, val_loss_history, val_accuracy_history, _, _ = data_extraction(data)

    plt.plot(loss_history, color="purple")
    plt.plot(val_loss_history, color="red")
    plt.plot(np.array(accuracy_history)*10, color="green")
    plt.plot(np.array(val_accuracy_history)*10, color="orange")

    plt.show()
    
    plt.close()

def total_chart(data):
    _, _, _, _, np_labels, np_predictions = data_extraction(data)

    # Total
    total_comparaisons =  np_labels == np_predictions
    total_succes_rate = math.floor(np.sum(total_comparaisons)/len(np_labels)*10000)/100

    labels = ["Sucess", "Failure"]
    sizes = [total_succes_rate,100-total_succes_rate]
    explode = (0.1, 0.1)

    create_pie(labels, sizes, explode)

def normal_chart(data):
    _, _, _, _, np_labels, np_predictions = data_extraction(data)
    
    mask = np_labels == 0
    counts = np.bincount(np_predictions[mask])

    labels = ["Normal", "Fake bacteria", "Fake virus"]
    sizes = [count_result(counts,0), count_result(counts,1), count_result(counts,2)]
    explode = (0.1, 0.1, 0.1)

    create_pie(labels, sizes, explode)

def pneumonia_chart(data):
    _, _, _, _, np_labels, np_predictions = data_extraction(data)
    
    mask = np_labels != 0
    counts = np.bincount(np_predictions[mask])

    labels = ["Faux négatifs", "Pneumonia"]
    sizes = [count_result(counts,0), len(np_labels[mask]) - count_result(counts,0)]
    explode = (0.1, 0.1)

    create_pie(labels, sizes, explode)


def bacteria_chart(data):
    _, _, _, _, np_labels, np_predictions = data_extraction(data)

    mask = np_labels == 1

    counts = np.bincount(np_predictions[mask])

    labels = ["Faux négatifs", "Bacteria", "Fake virus"]
    sizes = [count_result(counts,0), count_result(counts,1), count_result(counts,2)]
    explode = (0.1, 0.1, 0.1)

    create_pie(labels, sizes, explode)

def virus_chart(data):
    _, _, _, _, np_labels, np_predictions = data_extraction(data)

    mask = np_labels == 2

    counts = np.bincount(np_predictions[mask])

    labels = ["Faux négatifs", "Fake bacteria", "Virus"]
    sizes = [count_result(counts,0), count_result(counts,1), count_result(counts,2)]
    explode = (0.1, 0.1, 0.1)

    create_pie(labels, sizes, explode)