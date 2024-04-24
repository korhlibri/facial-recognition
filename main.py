import tensorflow as tf
import keras
from keras_vggface.vggface import VGGFace
import mtcnn
import numpy as np
import matplotlib as mpl
import PIL
import os
import os.path

train_dataset = keras.utils.image_dataset_from_directory("./faces", image_size=(640,480))

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
])

vggface = VGGFace(model="vgg16", include_top=False, input_shape=(640,480,3))

nb_class = 20

vggface.trainable = False
# last_layer = vggface.get_layer("avg_pool").output

inputs = tf.keras.Input(shape=(640,480,3))
x = data_augmentation(inputs)
x = vggface(x)
x = keras.layers.Flatten(name="flatten")(x)
out = keras.layers.Dense(nb_class, name="classifier")(x)
custom_vgg = keras.Model(inputs, out)

learning_rate = 0.0001
custom_vgg.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                   loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=["accuracy"])

history = custom_vgg.fit(train_dataset, epochs=20)