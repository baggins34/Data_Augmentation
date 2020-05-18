"""
Data Augmentation Homework
by
İbrahim Halil Bayat
Department of Electronics and Communication Engineering
İstanbul Technical University
İstanbul, Turkey
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

datagen = ImageDataGenerator(channel_shift_range=70, zca_whitening=True,
                             fill_mode="reflect", rescale=0.2)
img = load_img("gandalf.jpg")
img = img_to_array(img)
img = img.reshape((1,) + img.shape)

i = 0

for batch in datagen.flow(img, batch_size=1, save_to_dir="homework_file", save_format="jpeg"):
    i += 1
    if i > 150:
        break