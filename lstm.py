#!/usr/bin/env python
# LSTM classifier
# Uses LSTM in a variety of configurations for higher-accuracy labels
# Author: Wessel Heerema
# Latest build: 26/10/2024

import fasttext
import fasttext.util
import random as python_random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, BatchNormalization, Bidirectional
from keras.initializers import Constant
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_model(Y_train, adam):
    """Create the LSTM model with embedding matrix"""
    # Set parameters
    learning_rate = 0.005
    loss_function = 'categorical_crossentropy'
    if adam:
        optim = Adam(learning_rate=learning_rate)
    else:
        optim = SGD(learning_rate=learning_rate)
    # Take embedding dim and size from fasttext
    fasttext.util.download_model("en", if_exists='ignore')
    ft = fasttext.load_model("cc.en.300.bin")
    embedding_dim = 300
    num_tokens = len(ft)
    num_labels = len(set(Y_train))
    # Build model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(ft), trainable=False))
    # First LSTM layer (base layer with 64 units)
    # Return full sequence for the next LSTM layer
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    model.add(BatchNormalization())
    # Second LSTM layer with 32 units    
    model.add(Bidirectional(LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    model.add(BatchNormalization())
    # Third Bidirectional layer with 16 units
    model.add(Bidirectional(LSTM(16, dropout=0.3, recurrent_dropout=0.3)))
    model.add(BatchNormalization())
    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
    # Compile model, no scores just yet
    model.compile(loss=loss_function, optimizer=optim)
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev,
                batch_size=32, epochs=20, verb_bool=True, es_bool=True):
    """Wrapper for model fitting, with adjustable batch size, epochs,
    verbosity and early stopping
    Can also be used for the BERT classifier
    """
    if verb_bool:
        verbose = 1
    else:
        verbose = 0
    # Early stopping: stop training when there are three consecutive epochs without improving
    if es_bool:
        callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=3)]
    else:
        callback = None
    # Fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs,
              callbacks=callback, batch_size=batch_size,
              validation_data=(X_dev, Y_dev))
    return model
