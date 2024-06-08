import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_x, test_x = train_x / 255.0, test_x / 255.0

# One-hot encode the labels
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10, batch_size=32,
          validation_data=(test_x, test_y))

# Evaluate the model

loss, accuracy = model.evaluate(test_x, test_y)

st.write(f"Loss: {loss}")

st.write(f"Accuracy: {accuracy}")

# Display the model summary

st.write(model.summary())

# Display the model architecture

st.write(tf.keras.utils.plot_model(model, to_file='model.png',
         show_shapes=True, show_layer_names=True))

image = Image.open('model.png')

st.image(image, caption='Model Architecture', use_column_width=True)
