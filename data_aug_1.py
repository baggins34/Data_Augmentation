"""
Data Augmentation
by
İbrahim Halil Bayat
Department of Electronics and Communication Engineering
İstanbul Technical University
İstanbul, Turkey
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Defining the techniques for data augmentation

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True,
                             vertical_flip=True, fill_mode='nearest') # shear: cropping a part of the image

# Importing image

img = load_img("IMG_8055.jpg")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)


# Augmenting 50 images from 1 image and saving as .jpg

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir="Augmented", save_format=".jpeg"):
    i += 1
    if i > 50:
        break
