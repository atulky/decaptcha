# import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from helpers import resize_to_fit
from skimage import io

# from keras.models import Sequential
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers.core import Flatten, Dense
# from helpers import resize_to_fit
# from skimage import io


LETTER_IMAGES_FOLDER = "extracted_letter_images1"
MODEL_FILENAME = "captcha_model2.hdf5"
MODEL_LABELS_FILENAME = "model_labels2.dat"


# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = io.imread(image_file)
    # image = rgb2gray(image)


    # image = cv2.imread(image_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter so it fits in a 20x20 pixel box
    image = rescale(image, 0.25, anti_aliasing=False)
    letter_image = resize_to_fit(image, 50, 50)
    # image = resize(image, (50, 50))

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(letter_image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2]

    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)
# data = np.array(data, dtype="float") / 255.0
data = np.array(data, dtype="float")
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_rem, Y_train, Y_rem) = train_test_split(data, labels, train_size=0.8, random_state=0)
test_size = 0.5
X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem,Y_rem, test_size=0.5)
# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
model = tf.keras.models.Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), strides=(2, 2), padding="same", input_shape=(50, 50, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 24 nodes (one for each possible letter/number we predict)
model.add(Dense(24, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=50, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
