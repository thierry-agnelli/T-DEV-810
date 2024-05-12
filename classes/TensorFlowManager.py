import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import math

from pathlib import Path

_shape = (400, 400)

class TensorFlowManager:
    def __init__(self, model_name, epochs=5, batch_size=1, shuffle=None, repeat=None, shape=400):
        self.model_name = model_name
        
        self.path = "./assets/datasets/train/NORMAL/"
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.repeat = repeat
        self.shape = (shape, shape)

        if os.path.isfile(f"./models/{model_name}.keras"):
            self.model = load_model(f"./models/{model_name}.keras")
            print("model loaded")
        else:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=self.shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
        self.__compile_model()

    def __compile_model(self):
        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def save_model(self):
        self.model.save(f"./models/{self.model_name}.keras")

    def train(self, train_data, validation_data):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        train_dataset = self.__create_dataset(train_data)

        if self.repeat != None:
            if self.repeat == 0:
                train_dataset = train_dataset.repeat()
            else:
                train_dataset = train_dataset.repeat(self.repeat)
        
        validation_dataset = self.__create_dataset(validation_data)
        # self.model.fit(train_dataset, epochs=self.epochs)
        # self.model.fit(train_dataset, epochs=12, steps_per_epoch=250, callbacks=[early_stopping])
        # self.model.fit(train_dataset, epochs=self.epochs, callbacks=[early_stopping])        
        self.model.fit(train_dataset, epochs=self.epochs, callbacks=[early_stopping], validation_data=validation_dataset)

        self.save_model()   

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

    def __create_dataset(self, images_data):
        dataset = tf.data.Dataset.from_tensor_slices((images_data["paths"], images_data["labels"]))
        dataset = dataset.map(self.__preprocess_image)
        
        if self.shuffle != None:
            dataset = dataset.shuffle(self.shuffle)

        dataset = dataset.batch(self.batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def __preprocess_image(self, path, label):
        
        # Get image data
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        # Get image greatest shape size
        if len(image) > len(image[0]):
            greatest_shape_size = len(image)
        else:
            greatest_shape_size = len(image[0])

        # Resize image 
        # image = tf.image.resize_with_pad(image, 2700,3000)
        image = tf.image.resize_with_pad(image, greatest_shape_size, greatest_shape_size)
        image = tf.image.resize(image, [self.shape[0], self.shape[1]])

        # Format gray level
        # image = image / 255.0
        
        return image, label
    
    def dev_preprocess(self, path):
        image = self.__preprocess_image(path, "")
        
        return image


    # Static methods
    
    @staticmethod
    def __preprocessing_images(image):
        img = np.array(image)
        img = tf.expand_dims(img, axis=-1)
        img = tf.image.resize_with_pad(img, 2700, 3000)
        img = tf.image.resize(img, [shape[0], shape[1]])
        img = tf.squeeze(img)
        return img

    @staticmethod
    def about():
        print("tf version:", tf.__version__)
        print("Number of GPU avalaible:", len(tf.config.list_physical_devices("GPU")))
        print(tf.config.list_physical_devices("GPU"))

