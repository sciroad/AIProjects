import os 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = tf.keras.utils.normalize(train_x, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')

    ]
)


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=8)

model.save('handwritten_digit_recognition.model')



image_number =1

while os.path.isfile(f'digits/{image_number}.png'):
    try:
        img = cv.imread(f'digits/{image_number}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f'The number is probably: {np.argmax(prediction)}')
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print('Error')
    finally:
        image_number += 1