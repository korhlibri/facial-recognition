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
import time

# Detects and resizes the face in an image based on model specifications
# The model requires all input images to be of 224x224 resolution
# Once detected and resizes, saves the face in a new directory
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

# Extracts the face from an image.
# If a face is not detected, returns an empty list
def extract_face(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(img)
    if results:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        image = cv2.cvtColor(np.asarray(face), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.reshape(image, [1,224,224,3])
        return image
        # face_array = np.asarray(image)
        # return face_array
    else:
        return []

# Function to call to create resizes faces based on a directory in the following tree structure:
# faces/
#   name01/
#       01.png
#       02.png
#       03.png
#   name02/
#       01.png
#       02.png
#   ...
def create_resized_faces():
    path = "./faces"
    dirs = os.listdir(path)

    for subdir in dirs:
        print(f"resizing {subdir}")
        resize_face(path+f"/{subdir}", f"./faces_resized/{subdir}")

# If the resized directory does not exists, it means that the faces have not been resized yet.
# Therefore, calls the function to resize the faces.
if not os.path.isdir('./faces_resized'):
    create_resized_faces()

# Sets up a custom dataset based on the resized images of people that will be detected
train_dataset = keras.utils.image_dataset_from_directory(
    "./faces_resized",
    shuffle=True,
    batch_size=3,
    image_size=(224,224)
)

# Randomly changes the loaded images for easier detection
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2)
])

# Loads the model to be trained upon
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))

# Gets the number of people to detects and loads them into a variable
people_names = os.listdir("./faces_resized")
nb_class = len(people_names)

# Sets the loaded model to untrainable, allowing us to only train the custom loaded dataset
model.trainable = False
last_layer = model.get_layer("avg_pool").output

# Sets the inputs based on the model's image required resolution, which is 224x224
inputs = tensorflow.keras.Input(shape=(224,224,3))

# Parses several augmentations to the model
x = data_augmentation(inputs)
x = model(x)
x = keras.layers.Flatten(name="flatten")(x)

# These lines of code set the custom model based on the amount of new faces to detect
# The base model doesn't just provide faces, it also provides means of facial detection such as
# jawline, eyes, nose and other face features
# By adding custom faces to the base model, these detection features are used to easily detect
# new faces
out = keras.layers.Dense(nb_class, name="classifier")(x)
custom_vgg_model = keras.Model(inputs, out)

base_learning_rate = 0.001

# Compiles the custom model to be used for training the custom dataset
custom_vgg_model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
# Fits the custom model for our custom dataset
history = custom_vgg_model.fit(train_dataset, epochs=20)

# Activates the fitted model with the Softmax activation function
prob_model = keras.Sequential([
    custom_vgg_model,
    tensorflow.keras.layers.Softmax()
])

# After the model has been set up, the web camera is initialized
cam = cv2.VideoCapture(0)
cv2.namedWindow("preview")

# After initializing the camera, there is a delay of one second to allow for the camera to start
time.sleep(1)

if cam.isOpened(): # try to get the first frame
    rval, frame = cam.read()
else:
    rval = False

# While a frame is valid
while rval:
    rval, frame = cam.read()
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key%256 == 27: # exit on ESC
        break
    elif key%256 == 32: # capture on SPACE
        extracted_face = extract_face(frame)

        # If a face could be detected by extract_face() function
        # If a face could not be detected, the function returns an empty list
        if extracted_face != []:
            predictions = prob_model.predict(extracted_face)
            # These variables will calculate the best accuracy and position inside
            # the accuracy list. If there is a better accuracy, that person's face
            # is the face that was identified
            # This is checked exhaustively on every person inside the custom model
            best_x = 0.0
            best_i = 0
            for i, x in enumerate(predictions[0]):
                if x > best_x:
                    best_x = x
                    best_i = i
            # If an accuracy is below 70%, that means that the model cannot assert
            # confidently that the detected face is that person
            # The person in the image might be the same person, but most likely they
            # only share certain similarities
            if best_x > 0.70:
                print(f"Face detected: {people_names[best_i]}")
            else:
                print("Failed to identify face.")
        else:
            print("No face detected")

# Shutdown procedure for the web camera
cam.release()
cv2.destroyAllWindows()