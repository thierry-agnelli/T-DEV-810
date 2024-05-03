from classes import ImageManager
from classes import TensorFlowManager


img_manager = ImageManager("./assets/datasets/")
images_data = img_manager.get_images_data("test")

tf_manager = TensorFlowManager("pneumonia")
loss, accuracy = tf_manager.evaluate(images_data)


print("Test Loss:", loss)
print("Test Accuracy:", accuracy)