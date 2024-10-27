#!/usr/bin/env python
# MAIN PROGRAM
# Compare four models for binary text classification:
# Import text, get labels, produce Precision, Recall and F1 scores
# Author: Wessel Heerema
# Latest build: 27/10/2024

import argparse
from collections import Counter
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import classic as ldc
import lstm as ldl
import pretrained as ldb


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='data/train.tsv', type=str,
                        help="Train file to learn from"
                        "(default data/train.tsv)")
    parser.add_argument("-d", "--dev_file", default='data/dev.tsv', type=str,
                        help="Dev file to evaluate on"
                        "(default data/dev.tsv)")
    parser.add_argument("-t", "--test_file", default='data/test.tsv', type=str,
                        help="Test file to evaluate on"
                        "(default data/test.tsv)")
    parser.add_argument("-o", "--out_file", default='out.txt', type=str,
                        help="Output file for predicted labels"
                        "(default out.txt)")
    parser.add_argument("-od", "--dev_out", action="store_true",
                        help="Output dev labels instead of test")
    parser.add_argument("-p", "--model_params", default='options.json',
                        type=str,
                        help="Set hyperparameters per model using a JSON file"
                        "(default options.json)")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file, unseen=False):
    """Extract docs and labels in a binary classification task"""
    documents = []
    tok_texts = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            if not unseen:
                tok_texts.append(tokens[:-1])
                documents.append(" ".join(tokens[:-1]).strip())
                labels.append(tokens[-1])
            else:
                tok_texts.append(tokens)
                documents.append(" ".join(tokens[:-1]).strip())
    return documents, tok_texts, labels


def evaluate(gold, pred):
    """Perform Precision, Recall and F1-score test"""
    prc, rec, f1, _ = precision_recall_fscore_support(gold, pred)
    print(f"Precision: {prc}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    args = create_arg_parser()

    # Read files
    X_train, X_traint, Y_train = read_corpus(args.train_file)
    X_dev, X_devt, Y_dev = read_corpus(args.dev_file)
    with open(args.model_params, "r") as fp:
        params = json.load(fp)
    bp = params["baseline"]
    op = params["optimized"]
    lp = params["lstm"]
    pp = params["pretrained"]

    # Baseline
    if bp["enable"]:
        print("BASELINE")
        b_pred = ldc.classic_classifier(X_traint, Y_train, X_devt,
                                        bp["model"], bp["tfidf"])
        evaluate(Y_dev, b_pred)

    # Optimized
    if op["enable"]:
        print("OPTIMIZED")
        o_pred = ldc.classic_classifier(X_traint, Y_train, X_devt,
                                        op["model"], op["tfidf"], op["ngram"])
        evaluate(Y_dev, o_pred)

    # LSTM
    l_emb, Xtv, Xdv, Ytb, Ydb = ldl.set_vec_emb(X_train, X_dev,
                                                Y_train, Y_dev)
    unique, counts = np.unique(Ydb, return_counts=True)
    print(dict(zip(unique, counts)))
    weights = {i: (1 / np.sum(Ytb == i)) * (len(Ytb) / 2.0) for i in range(2)}
    print(weights)
    if lp["enable"]:
        print("LSTM")
        l_model = ldl.create_model(l_emb, lp["adam"], lp["layers"],
                                   lp["nodes"], lp["decrement"])
        l_model = ldl.train_model(l_model, Xtv, Ytb, Xdv, Ydb,
                                  batch_size=32, epochs=lp["epochs"],
                                  weights=weights)
        l_pred = l_model.predict(Ydb)
        unique, counts = np.unique(l_pred, return_counts=True)
        print(dict(zip(unique, counts)))
        evaluate(np.argmax(Ydb, axis=1), np.argmax(l_pred, axis=1))

    # Pretrained
    if pp["enable"]:
        print("PRETRAINED")
        Xtt, Xdt = ldb.set_tok(X_train, X_dev, pp["model"])
        p_model = ldl.train_model(ldb.create_model(pp["adam"], pp["model"]),
                                  Xtt, Ytb, Xdt, Ydb, epochs=pp["epochs"],
                                  weights=weights)
        p_pred = p_model.predict(Ydb)["logits"]
        procpred = np.zeros((len(p_pred), 1))
        for i in range(len(p_pred)):
            if p_pred[i][0] > p_pred[i][1]:
                procpred[i] = 0
            else:
                procpred[i] = 1
        #unique, counts = np.unique(mod_pred, return_counts=True)
        #print(dict(zip(unique, counts)))
        #gold_t = np.argmax(Ydb, axis=1)
        #pred_t = np.argmax(p_pred, axis=1)
        evaluate(Ydb, procpred)
