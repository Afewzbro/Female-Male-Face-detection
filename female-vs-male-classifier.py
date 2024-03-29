# -*- coding: utf-8 -*-
"""CNN1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12PwWeofpP-HPAx0DiL2_QpAavQGVdk1e
"""

# Installing necessary libraries
!pip install tensorflow
!pip install numpy
!pip install matplotlib
!pip install kaggle

# importing the necessary libraries.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import pandas as pd

# mounting the google drive
from google.colab import drive

drive.mount('/content/drive')

train_data_dir = '/content/drive/MyDrive/Male and Female face dataset/train'
validation_data_dir = '/content/drive/MyDrive/Male and Female face dataset/validation'
test_data_dir = '/content/drive/MyDrive/Male and Female face dataset/test'

img_width, img_height = 150, 150
batch_size = 32

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGBA")
    image = image.resize((img_width, img_height))
    image_array = np.array(image)
    return image_array[:, :, :3]  # Keep only RGB channels, discarding the alpha channel

train_df = pd.DataFrame(columns=["filename", "class"])
validation_df = pd.DataFrame(columns=["filename", "class"])
test_df = pd.DataFrame(columns=["filename", "class"])

for class_label in os.listdir(train_data_dir):
    class_dir = os.path.join(train_data_dir, class_label)
    for filename in os.listdir(class_dir):
        train_df = train_df.append({"filename": os.path.join(class_dir, filename), "class": class_label}, ignore_index=True)

for class_label in os.listdir(validation_data_dir):
    class_dir = os.path.join(validation_data_dir, class_label)
    for filename in os.listdir(class_dir):
        validation_df = validation_df.append({"filename": os.path.join(class_dir, filename), "class": class_label}, ignore_index=True)

for class_label in os.listdir(test_data_dir):
    class_dir = os.path.join(test_data_dir, class_label)
    for filename in os.listdir(class_dir):
        test_df = test_df.append({"filename": os.path.join(class_dir, filename), "class": class_label}, ignore_index=True)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    x_col="filename",
                                                    y_col="class",
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    preprocessing_function=preprocess_image)

validation_generator = test_datagen.flow_from_dataframe(validation_df,
                                                        x_col="filename",
                                                        y_col="class",
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        preprocessing_function=preprocess_image)

test_generator = test_datagen.flow_from_dataframe(test_df,
                                                  x_col="filename",
                                                  y_col="class",
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='binary',
                                                  preprocessing_function=preprocess_image)

# Building CNN model using the Sequential API from Keras.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the Model using the fit function
epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps)

# evaluateingthe model's performance on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.n // batch_size)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualize the Training Process
# Ploting the accuracy and loss curves to visualize the training process.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

test_generator.reset()  # Reset the test generator
y_pred = model.predict(test_generator)
y_pred = np.round(y_pred).flatten()  # Convert probabilities to binary predictions (0 or 1)
y_true = test_generator.classes  # True labels from the test generator

cm = confusion_matrix(y_true, y_pred)

labels = ['Female', 'Male']  # Update with your class labels if different
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

image_path = '/content/Screenshot from 2023-07-05 19-31-21.png'
image = load_img(image_path, target_size=(img_width, img_height))
image_array = img_to_array(image) / 255.0  # Convert image to array and normalize pixel values
image_array = np.expand_dims(image_array, axis=0)  # Add an extra dimension to match the model's input shape

plt.imshow(image)

predictions = model.predict(image_array)
class_labels = ['Female', 'Male']  # Define your class labels

# Get the predicted class index (0 for cat, 1 for dog)
predicted_class_index = np.argmax(predictions[0])

# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class label
print(f"Predicted Class: {predicted_class_label}")