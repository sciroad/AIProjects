import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#filepath
file_path = tf.keras.utils.get_file('dataset.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(file_path, 'rb').read().decode(encoding='utf-8').lower()

text = text[200000:700000]

chars = sorted(set(text))

char_to_index = {char: index for index, char in enumerate(chars)}

index_to_char = {index: char for index, char in enumerate(chars)}


sentences = [] # X
next_chars = [] # Y

SEQ_LEN = 64 # Length of the sequence
STEP_SIZE = 1 # Step to create a new sequence


for i in range(0, len(text) - SEQ_LEN, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LEN])
    next_chars.append(text[i + SEQ_LEN])


x = np.zeros((len(sentences), SEQ_LEN, len(chars)), dtype=np.bool) # One-hot 
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) # One-hot 

for s, sentence in enumerate(sentences):
    for c, char in enumerate(sentence):
        x[s, c, char_to_index[char]] = 1
    y[s, char_to_index[next_chars[s]]] = 1

model = Sequential(
    [
        LSTM(128, input_shape=(SEQ_LEN, len(chars))),
        Dense(len(chars)),
        Activation('softmax')
    ]
)

LEARNING_RATE = 0.01
LOSS = 'categorical_crossentropy'
BATCH_SIZE = 512
EPOCHS = 8

optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss=LOSS, optimizer=optimizer)

model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

model.save('model.h5')


def sample (preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature=1.0):
    start_index = random.randint(0, len(text) - SEQ_LEN - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LEN]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, SEQ_LEN, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

print(generate_text(1000, 0.5))





