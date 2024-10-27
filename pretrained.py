#!/usr/bin/env python
# BERT classifier
# Uses BERT-derived models in a variety of configurations 
# for higher-accuracy labels
# Author: Wessel Heerema
# Latest build: 26/10/2024

import random as python_random
import numpy as np
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_model(adam=True, lm="bert-base-uncased"):
    """Create the BERT or derived model"""
    # Set params
    learning_rate = 0.0005
    loss_function = BinaryCrossentropy(from_logits=True)
    if adam:
        optim = Adam(learning_rate=learning_rate)
    else:
        optim = SGD(learning_rate=learning_rate)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm,
                                                                 num_labels=1)
    model.compile(loss=loss_function, optimizer=optim)
    return model


def set_tok(X_train, X_dev, lm="bert-base-uncased", maxlen=100):
    """Tokenize the train and dev set"""
    tokenizer = AutoTokenizer.from_pretrained(lm)
    X_train_tok = tokenizer(X_train, padding=True, max_length=maxlen,
                            truncation=True, return_tensors="np").data
    X_dev_tok = tokenizer(X_dev, padding=True, max_length=maxlen,
                          truncation=True, return_tensors="np").data
    return X_train_tok, X_dev_tok
