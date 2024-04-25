from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, sys
from pathlib import Path
import keras
import tensorflow

def resize_face(path, pathtosave):
    for item in os.listdir(path):
        if os.path.isfile(path+"/"+item):
            img=cv2.imread(path+"/"+item, flags=cv2.IMREAD_UNCHANGED)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detector = MTCNN()
            results = detector.detect_faces(img)
            if results:
                x1, y1, width, height = results[0]['box']
                x2, y2 = x1 + width, y1 + height
                face = img[y1:y2, x1:x2]
                image = Image.fromarray(face)
                image = image.resize((224, 224))
                Path(pathtosave+"/").mkdir(parents=True, exist_ok=True)
                image.save(pathtosave+"/"+item, "PNG", quality=100)
            else:
                print("Face not detected!")

def extract_face(facefile):
    img=cv2.imread(facefile, flags=cv2.IMREAD_UNCHANGED)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(img)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    return face_array

def create_resized_faces():
    path = "./faces"
    dirs = os.listdir(path)

    for subdir in dirs:
        print(f"resizing {subdir}")
        resize_face(path+f"/{subdir}", f"./faces_resized/{subdir}")

train_dataset = keras.utils.image_dataset_from_directory(
    "./faces_resized",
    shuffle=True,
    batch_size=3,
    image_size=(224,224)
)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2)
])

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))

nb_class = 19

model.trainable = False
last_layer = model.get_layer("avg_pool").output

inputs = tensorflow.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)
x = model(x)
x = keras.layers.Flatten(name="flatten")(x)

out = keras.layers.Dense(nb_class, name="classifier")(x)
custom_vgg_model = keras.Model(inputs, out)

base_learning_rate = 0.001

custom_vgg_model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
history = custom_vgg_model.fit(train_dataset, epochs=20)