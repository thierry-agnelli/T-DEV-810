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
epochs = 40
shuffle = len(train_data["paths"])
repeat = 5
shape = 224


print("Trains model")
tf_manager = TensorFlowManager("pneumonia", batch_size=batch_size, epochs=epochs, shuffle=shuffle, repeat=repeat, shape=shape)
loss_history, accuracy_history = tf_manager.train(train_data, validation_data)


print("learning done")

def comparaison(np_labels, np_predictions, mask):
    comparaisons =  np_labels[mask] == np_predictions[mask]
    succes_rate = math.floor(np.sum(comparaisons)/len(np_labels[mask])*10000)/100

    return succes_rate

images_data = img_manager.get_images_data("test")

tf_manager = TensorFlowManager("pneumonia", shape=shape)
predictions = tf_manager.predictions_by_batch(images_data)

np_labels = np.array(images_data["labels"])
np_predictions = np.array(predictions)

chart_labels = []
chart_data = []

# Total
total_comparaisons =  np_labels == np_predictions
total_succes_rate = math.floor(np.sum(total_comparaisons)/len(np_labels)*10000)/100
chart_labels.append("Total")
chart_data.append(total_succes_rate)


# Normal
mask = np_labels == 0
normal_succes_rate = comparaison(np_labels, np_predictions, mask)
chart_labels.append("Normal")
chart_data.append(normal_succes_rate)

# Pneumonia
mask = np_labels != 0
pneumonia_succes_rate = comparaison(np_labels, np_predictions, mask)
chart_labels.append("Pneumonia")
chart_data.append(pneumonia_succes_rate)

# Bacteria
mask = np_labels == 1
bacteria_succes_rate = comparaison(np_labels, np_predictions, mask)
chart_labels.append("Bacteria")
chart_data.append(bacteria_succes_rate)

# Virus
mask = np_labels == 2
virus_succes_rate = comparaison(np_labels, np_predictions, mask)
chart_labels.append("Virus")
chart_data.append(virus_succes_rate)

print("Total:", total_succes_rate, "%")
print("   Normal:", normal_succes_rate, "%")
print("   Pneumonia:", pneumonia_succes_rate, "%")
print("      Bacteria:", bacteria_succes_rate, "%")
print("      Virus:", virus_succes_rate, "%")


fig, ax = plt.subplots()
bottom = np.zeros(5)
bar_labels = ["purple", "green", "red", "orange", "blue"]
bar_colors = ["tab:purple", "tab:green", "tab:red", "tab:orange", "tab:blue"]

bar = ax.bar(chart_labels, chart_data, label=bar_labels, bottom=bottom, color=bar_colors)
delta = 100 - np.array(chart_data)
bottom += chart_data
ax.bar_label(bar, label_type='center')

bar = ax.bar(chart_labels, delta, bottom=bottom, color="gray")

ax.set_ylabel("Result in %")
ax.set_title("Model prediction results")
# ax.legend(["this a legend.","blah blah blah"])
ax.set_xlabel(f"shape: {shape} / batch_size: {batch_size} / epochs: {epochs} / shuffle: {shuffle} / repeat: {repeat}")
# ax.legend(title="Legend", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# plt.show()
plt.savefig(f"./results/chart.png")

chart_count = len(os.listdir("./results"))
plt.savefig(f"./results/chart_#{chart_count}.png")

plt.close()

plt.plot(loss_history)
plt.plot(np.array(accuracy_history)*10)
# ax.set_xlabel(f"Nombre d'epochs : {len(loss_history)} / loss: {loss_history[-1]} / accuracy: {accuracy_history[-1]*100}")

plt.savefig(f"./results/plots/plots.png")

plot_count = len(os.listdir("./results/plots"))
plt.savefig(f"./results/plots/plot_#{plot_count}.png")

plt.close()