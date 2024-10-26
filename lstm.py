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
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def set_vec_emb(X_train, X_dev, Y_train, Y_dev):
    """Create encodings and an embedding matrix for the train and dev sets
    using Fasttext
    """
    # Set up Fasttext
    fasttext.util.download_model("en", if_exists='ignore')
    ft = fasttext.load_model("cc.en.300.bin")
    # Use train and dev to create vocab
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)
    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Generate matrix
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Get embedding dimension from the word "the"
    embedding_dim = len(ft.get_word_vector("the"))
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = ft.get_word_vector(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector

    # Return final results
    return embedding_matrix, X_train_vect, X_dev_vect, Y_train_bin, Y_dev_bin


def create_model(emb_matrix, adam=True):
    """Create the LSTM model with embedding matrix"""
    # Set parameters
    learning_rate = 0.005
    loss_function = BinaryCrossentropy()
    if adam:
        optim = Adam(learning_rate=learning_rate)
    else:
        optim = SGD(learning_rate=learning_rate)
    # Take embedding dim and size from fasttext
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    # Build model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False))
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
    model.add(Dense(input_dim=embedding_dim, units=1, activation="softmax"))
    # Compile model, no scores just yet
    model.compile(loss=loss_function, optimizer=optim)
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev,
                batch_size=16, epochs=20, verb_bool=True, es_bool=True):
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


def result_transform(invec):
    """Transform gold/predicted encoding into numbers"""
    return np.argmax(invec, axis=1)
