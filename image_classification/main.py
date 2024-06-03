import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the dataset
train_x, train_y, test_x, test_y = tf.keras.datasets.cifar10.load_data()

# Normalize the data
train_x = train_x / 255.0
test_x = test_x / 255.0

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


TRAIN_SIZE = 10000
TEST_SIZE = 2000

train_x = train_x[:TRAIN_SIZE]
train_y = train_y[:TRAIN_SIZE]
test_x = test_x[:TEST_SIZE]
test_y = test_y[:TEST_SIZE]

model = Sequential(
    [
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax') # 10 classes
    ]
)



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=16, validation_data=(test_x, test_y))

# Save the model
model.save('model.h5')

# Test the model
predictions = model.predict(test_x)

for i in range(10):
    print(f"Prediction: {labels[np.argmax(predictions[i])]}")
    print(f"Actual: {labels[test_y[i][0]]}")
    plt.imshow(test_x[i])
    plt.show()

loss, accuracy = model.evaluate(test_x, test_y)
print(f"Loss: {loss},\n Accuracy: {accuracy}")

# TEST with a new image
image = cv.imread('image.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = cv.resize(image, (32, 32))

# Normalize the image
image = image 

prediction = model.predict(np.array([image])/ 255.0)

print(f"Prediction: {labels[np.argmax(prediction)]}")


