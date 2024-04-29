from classes import ImageManager

img_manager = ImageManager("./assets/")

img = img_manager.convert("/train/Normal/IM-0115-0001.jpeg")

img_manager.print_image(img)

