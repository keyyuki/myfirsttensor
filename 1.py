from PIL import Image

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 256, 256)
    image_32 = tf.cast(image_resized, tf.float32)
   
    return image_32, label

img_shape = (256, 256, 3)
batch_size = 3
epochs = 5 

train_filename = tf.constant([
    "./test/train/0/Alcil_j0.jpg",
    "./test/train/0/Alcil_j1.jpg",
    "./test/train/0/Alcil_j2.jpg",

    "./test/train/1/Amcla_f0.jpg",
    "./test/train/1/Amcla_f1.jpg",
    "./test/train/1/Amcla_j0.jpg",

    "./test/train/2/Caaub_u0.jpg",
    "./test/train/2/Caaug_u0.jpg",
    "./test/train/2/Caaur_l1.jpg",
])
train_labels = tf.constant([0,0,0, 1,1,1, 2,2,2])

dataTrain = tf.data.Dataset.from_tensor_slices((train_filename, train_labels))
dataTrain = dataTrain.map(_parse_function)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(256, 256)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(dataTrain, epochs=epochs, steps_per_epoch=9)
