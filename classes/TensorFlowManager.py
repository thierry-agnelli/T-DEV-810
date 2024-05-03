import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import math

from pathlib import Path

shape = (270, 300)
# shape = (810, 900)
# shape = (1350, 1500)
# hsape = (2700, 3000)


class TensorFlowManager:
    def __init__(self, model_name):
        self.model_name = model_name
        
        self.path = "./assets/datasets/train/NORMAL/"
        self.batch_size = 32

        if os.path.isfile(f"./.models/{model_name}.h5"):
            self.model = load_model(f"./.models/{model_name}.h5")
            print("model loaded")
        else:
            self.model = tf.keras.models.Sequential([
                # tf.keras.layers.Flatten(input_shape=shape),
                tf.keras.layers.Input(shape=shape),
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

    #######################################################################################

    def old_train(self, dataset):
        # preprocessing
        print("preprocessing images...")
        
        i=0
        print("0%\r", end='')
        images_batch = []

        for image in dataset["images"]:
            img = self.__preprocessing_images(image)
            
            images_batch.append(img)
            i += 1
            print(f" {math.floor(i/len(dataset["images"])*10000)/100}%      ", end='\r')

        print("\nPreprocessing ended.")

        self.model.fit(np.array(images_batch),np.array(dataset["labels"]), epochs=5)
        
        self.__compile_model()

        # self.model.save(f"./.models/{self.model_name}.h5")
        
        # del model
        del images_batch
    
    def old_evaluate(self, dataset):
        # preprocessing
        print("preprocessing images...")
        
        i=0
        print("0%\r", end='')
        images_batch = []

        for image in dataset["images"]:
            img = self.__preprocessing_images(image)

            images_batch.append(img)
            i += 1
            print(f" {math.floor(i/len(dataset["images"])*10000)/100}%      ", end='\r')

        print("\nPreprocessing ended.")

        loss, accuracy = self.model.evaluate(np.array(images_batch), np.array(dataset["labels"]))
        
        return (loss, accuracy)

    def old_predictions(self, dataset):
        # preprocessing
        print("preprocessing images...")
        
        i=0
        print("0%\r", end='')
        images_batch = []

        for image in dataset["images"]:
            img = self.__preprocessing_images(image)

            images_batch.append(img)
            i += 1
            print(f" {math.floor(i/len(dataset["images"])*10000)/100}%      ", end='\r')

        print("\nPreprocessing ended.")
    
        predict_batch = np.array(images_batch)
        batch_size = len(predict_batch)
        succes_count = 0

        for i in range(batch_size):
            predict_input_tensor = tf.expand_dims(predict_batch[0], axis=0)

            predictions = self.model.predict(predict_input_tensor)
            predicted_label = np.argmax(predictions)

            if predicted_label == dataset["labels"][i]:
                succes_count += 1

        return math.floor(succes_count/batch_size*10000)/100

    #######################################################################################

    def save_model(self):
        self.model.save(f"./.models/{self.model_name}.h5")

    def train(self, images_data):
        # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        dataset = self.__create_dataset(images_data)
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat(5)

        self.model.fit(dataset, epochs=5)
        # self.model.fit(dataset, epochs=5, callbacks=[early_stopping], validation_data=val_dataset)

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


    def __create_dataset(self, images_data):
        dataset = tf.data.Dataset.from_tensor_slices((images_data["paths"], images_data["labels"]))
        dataset = dataset.map(self.__preprocess_image)
        dataset = dataset.batch(self.batch_size)

        return dataset

    def __preprocess_image(self, path, label):
        
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize_with_pad(image, 2700, 3000)
        image = tf.image.resize(image, [shape[0], shape[1]])
        image = image / 255.0
        
        return image, label

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

