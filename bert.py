#!/usr/bin/env python
# BERT classifier
# Uses BERT-derived models in a variety of configurations 
# for higher-accuracy labels
# Author: Wessel Heerema
# Latest build: 26/10/2024

import random as python_random
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def train_model(adam, lm="bert-base-uncased"):
    """Create the BERT or derived model"""
    # Set params
    learning_rate = 0.0005
    loss_function = CategoricalCrossentropy(from_logits=True)
    if adam:
        optim = Adam(learning_rate=learning_rate)
    else:
        optim = SGD(learning_rate=learning_rate)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=6)
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model
