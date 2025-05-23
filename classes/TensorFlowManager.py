import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import math

from pathlib import Path

class TensorFlowManager:
    def __init__(self, model_name, epochs=5, batch_size=1, shuffle=None, repeat=None, shape=400, patience=None):
        self.model_name = model_name
        
        self.path = "./assets/datasets/train/NORMAL/"
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.repeat = repeat
        self.patience = patience
        self.shape = (shape, shape)
        self.index_count = 0

        if os.path.isfile(f"./models/{model_name}.keras"):
            self.model = load_model(f"./models/{model_name}.keras")
            print("model loaded")
        else:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=self.shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(10, activation="softmax")
            ])
            
        self.__compile_model()

    def __compile_model(self):
        self.model.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

    def save_model(self):
        self.model.save(f"./models/{self.model_name}.keras")

    def train(self, train_data, validation_data):
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        train_dataset = self.__create_dataset(train_data, repeat=self.repeat)
        
        validation_dataset = self.__create_dataset(validation_data)
        
        # self.model.fit(train_dataset, epochs=12, steps_per_epoch=250, callbacks=[early_stopping])
        history = self.model.fit(train_dataset, epochs=self.epochs, callbacks=[early_stopping], validation_data=validation_dataset)

        # Historique de la loss
        loss_history = history.history["loss"]
        val_loss_history = history.history["val_loss"]

        # Historique de l"accuracy
        accuracy_history = history.history["accuracy"]
        val_accuracy_history = history.history["val_accuracy"]

        self.save_model()

        print("repetion augmentation result")
        print(self.index_count)

        return (loss_history, accuracy_history, val_loss_history, val_accuracy_history)

    def evaluate(self, images_data):
        dataset = self.__create_dataset(images_data)

        loss, accuracy = self.model.evaluate(dataset)

        return (loss, accuracy)


    def predictions_by_batch(self, images_data):
        dataset = self.__create_dataset(images_data)

        predictions = self.model.predict(dataset)
        
        predicted_labels = []

        for prediction in predictions:
            predicted_labels.append(np.argmax(prediction))

        return predicted_labels

    # Private methods

    def __create_dataset(self, images_data, repeat=None):
        # Création dataset
        dataset = tf.data.Dataset.from_tensor_slices((images_data["paths"], images_data["labels"]))

        # Pre processing images
        dataset = dataset.map(self.__preprocess_image)

        # Mélange du dataset
        if self.shuffle != None:
            dataset = dataset.shuffle(self.shuffle)

        
        # Découpe du dataset en lot
        dataset = dataset.batch(self.batch_size)

        # Répétitions du dataset
        if repeat != None:
            if repeat == 0:
                dataset = dataset.repeat()
            else:
                dataset = dataset.repeat(self.repeat)

                # dataset = dataset.enumerate()
                # dataset = dataset.map(self.__cycle_augmentation)


        # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def __preprocess_image(self, path, label):
        
        # Get image data
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.adjust_jpeg_quality(image,100)
   
        # Get image greatest shape size
        if len(image) > len(image[0]):
            greatest_shape_size = len(image)
        else:
            greatest_shape_size = len(image[0])

        # Resize image 
        image = tf.image.resize_with_pad(image, greatest_shape_size, greatest_shape_size)
        image = tf.image.resize(image, [self.shape[0], self.shape[1]])
        
        return image, label

    def __cycle_augmentation(self, index, data):
        image, label = data

        
        if (index + 1) % 2 == 0:
            image = tf.image.adjust_contrast(image, 0.25)
        if (index + 1) % 3 == 0:
            image = tf.image.adjust_brightness(image, 0.25)

        return image, label
    
    # Static methods

    @staticmethod
    def about():
        print("tf version:", tf.__version__)
        print("Number of GPU avalaible:", len(tf.config.list_physical_devices("GPU")))
        print(tf.config.list_physical_devices("GPU"))

