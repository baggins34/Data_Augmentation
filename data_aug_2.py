"""
Data Augmentation
by
İbrahim Halil Bayat
Department of Electronics and Communication Engineering
İstanbul Technical University
İstanbul, Turkey
"""

# Data augmentation and visualization with CIFAR 10 dataset

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import numpy as np


# Data downloading

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


# Visualizing original images

datagen = ImageDataGenerator()
datagen.fit(x_train)

for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=4, seed=499):
    # It takes 4 images randomly from the datset
    for i in range(0,4):
        plt.subplot(220 +1 +i)
        plt.imshow(x_batch[i])
    plt.show()
    break


# Rotating the images

datagen2 = ImageDataGenerator(rotation_range=359)
datagen2.fit(x_train)
for x_batch, y_batch in datagen2.flow(x_train, y_train, batch_size=4, seed=499):
    for i in range(0,4):
        plt.subplot(220 +1 +i)
        plt.imshow(x_batch[i])
    plt.show()
    break

# Shifting the images

datagen3 = ImageDataGenerator(height_shift_range=0.5)
datagen3.fit(x_train)
for x_batch, y_batch in datagen3.flow(x_train, y_train, batch_size=4, seed=499):
    for i in range(0,4):
        plt.subplot(220 +1 +i)
        plt.imshow(x_batch[i])
    plt.show()
    break

# Horizonral symmetry

datagen4 = ImageDataGenerator(horizontal_flip=True)
datagen4.fit(x_train)
for x_batch, y_batch in datagen4.flow(x_train, y_train, batch_size=4, seed=499):
    for i in range(0,4):
        plt.subplot(220 +1 +i)
        plt.imshow(x_batch[i])
    plt.show()
    break