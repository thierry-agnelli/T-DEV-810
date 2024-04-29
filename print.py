from classes import ImageManager

img_manager = ImageManager("./assets/datasets")

img = img_manager.convert("/train/NORMAL/IM-0115-0001.jpeg")

img_manager.print_image(img)

