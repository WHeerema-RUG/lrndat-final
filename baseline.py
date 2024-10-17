#!/usr/bin/env python
# Baseline classifier
# Uses one of a variety of algorithms to produce baseline labels
# Author: Wessel Heerema
# Latest build: 17/10/2024

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


def base_classifier(X_train, Y_train, X_test, model="nb", tfidf=True):
    '''Create and run baseline classifier'''
    # Vectorize data
    if tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity,
                              ngram_range=(1,3))
    else:
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity,
                              ngram_range=(1,3))
    # Select algorithm
    algorithms = {
        'nb': MultinomialNB(),
        'svc': LinearSVC(),
        'knn': KNeighborsClassifier(n_neighbors=5)
    }
    classifier = Pipeline([('vec', vec), ('cls', algorithms[model])])
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred
