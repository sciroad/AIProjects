from io import BytesIO

import cv2 as cv
import numpy as np
import tensorflow as tf
from telegram.ext import *

with open('token.txt', 'r') as f:
    token = f.read().strip()

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# model = tf.keras.models.load_model('model.h5')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


def start(update, context):
    update.message.reply_text('Send me an image')


def help(update, context):
    update.message.reply_text(
        """
        /start - start the bot
        /help - get help
        /train - train the model
        """
    )


def train(update, context):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=10)
    model.evaluate(test_x, test_y)

    model.save('model.h5')
    update.message.reply_text('Model trained and saved')


def handle_message(update, context):
    update.message.reply_text('Please train the model and send an image')


def handle_image(update, context):
    photo = update.message.photo[-1].get_file()
    img = cv.imdecode(np.frombuffer(
        photo.download_as_bytearray(), np.uint8), -1)
    img = cv.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_name = class_names[np.argmax(prediction)]

    update.message.reply_text(f'This is a {class_name}')


updater = Updater(token, use_context=True)
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('help', help))
dispatcher.add_handler(CommandHandler('train', train))
dispatcher.add_handler(MessageHandler(Filters.text, handle_message))
dispatcher.add_handler(MessageHandler(Filters.photo, handle_image))

updater.start_polling()
