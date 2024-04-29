import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

class TensorFlowManager:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def tf_learning(self, train_dataset, test_dataset):
        # mnist = tf.keras.datasets.mnist
        # (x_train, y_train),(x_test, y_test) = mnist.load_data()
        # x_train, x_test = x_train / 255.0, x_test / 255.0

        train_batch_images = np.array(train_dataset["images"])/255.0
        train_batch_labels = np.array(train_dataset["labels"])
        
        test_batch_images = np.array(test_dataset["images"])/255.0
        test_batch_labels = np.array(test_dataset["labels"])
        
        self.model.fit(train_batch_images, train_batch_labels, epochs=5)

        # self.model.evaluate(test_batch_images, test_batch_labels)
        self.model.save("./train_model/myModel.h5")

   
    def predict(self, dataset, predict_index):
        batch_images = np.array(dataset["images"])/255.0
        batch_labels = np.array(dataset["labels"])
        predict_input_tensor = tf.expand_dims(batch_images, 0)[0]

        if os.path.isfile("./train_model/myModel.h5"):        
            model = load_model("./train_model/myModel.h5")
            predictions = model.predict(predict_input_tensor[predict_index-1:predict_index])
            predicted_label = np.argmax(predictions)

            # print(predictions)
            # print(predicted_label)
            # print(batch_labels[predict_index-1:predict_index])
            return predicted_label

        else:
            print("### /!\\ Please train your model before /!\\ ###")

    def prediction(self, image):
        predict_input_tensor = tf.expand_dims(image/255.0, axis=0)
        
        if os.path.isfile("./train_model/myModel.h5"):        
            model = load_model("./train_model/myModel.h5")
            # print(predict_input_tensor)
            predictions = model.predict(predict_input_tensor)
            predicted_label = np.argmax(predictions)

            return predicted_label

        else:
            print("### /!\ Please train your model before /!\ ###")



    def about(self):
        print("tf version:", tf.__version__)

