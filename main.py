#!/usr/bin/env python
# MAIN PROGRAM
# Compare four models for binary text classification:
# Import text, get labels, produce Precision, Recall and F1 scores
# Author: Wessel Heerema
# Latest build: 29/10/2024

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
    parser.add_argument("-t", "--test_file", default='none', type=str,
                        help="Test file to evaluate on"
                        "(default none)")
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
    # Initialize output lists
    documents = []
    tok_texts = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            if not unseen:
                # Labeled
                tok_texts.append(tokens[:-1])
                documents.append(" ".join(tokens[:-1]).strip())
                labels.append(tokens[-1])
            else:
                # Unlabeled
                tok_texts.append(tokens)
                documents.append(" ".join(tokens[:-1]).strip())
    return documents, tok_texts, labels


def evaluate(gold, pred):
    """Perform Precision, Recall and F1-score test"""
    # Prepare predicted labels for testing
    # 2D array separated by type of prediction
    if pred.ndim == 2:
        if pred.shape[1] == 2:
            procpred = np.zeros((len(pred), 1))
            for i in range(len(pred)):
                if pred[i][0] > pred[i][1]:
                    procpred[i] = 0
                else:
                    procpred[i] = 1
        elif pred.shape[1] == 1:
            procpred = np.zeros(len(pred))
            for i in range(len(pred)):
                if pred[i] > 0.5:
                    procpred[i] = 1
                else:
                    procpred[i] = 0
        else:
            raise TypeError("Labels are incorrect shape")
    # If standard one-dimensional array, pass straight through
    else:
        procpred = pred
    prc, rec, f1, _ = precision_recall_fscore_support(gold, procpred)
    print(f"Precision: {prc}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    args = create_arg_parser()

    # Read files
    X_train, X_traint, Y_train = read_corpus(args.train_file)
    X_dev, X_devt, Y_dev = read_corpus(args.dev_file)
    if args.test_file == "none":
        test_data = False
    else:
        X_test, X_testt, Y_test = read_corpus(args.test_file)
        test_data = True
    with open(args.model_params, "r") as fp:
        params = json.load(fp)
    bp = params["baseline"]
    op = params["optimized"]
    lp = params["lstm"]
    pp = params["pretrained"]

    # Initialize test data if available
    if test_data:
        c_testable = X_testt
        c_gold = Y_test
    else:
        c_testable = X_devt
        c_gold = Y_dev

    # Baseline
    if bp["enable"]:
        print("BASELINE")
        b_pred = ldc.classic_classifier(X_traint, Y_train, c_testable,
                                        bp["model"], bp["tfidf"])
        evaluate(c_gold, b_pred)

    # Optimized
    if op["enable"]:
        print("OPTIMIZED")
        o_pred = ldc.classic_classifier(X_traint, Y_train, c_testable,
                                        op["model"], op["tfidf"], op["ngram"])
        evaluate(c_gold, o_pred)

    # Preprocess data
    l_emb, Xtv, Xdv, Ytb, Ydb,\
    enc, vec = ldl.set_vec_emb(X_train, X_dev, Y_train, Y_dev)
    Xtt, Xdt, tok = ldb.set_tok(X_train, X_dev, pp["model"])
    # Preprocess test data if available
    if test_data:
        l_testable = vec(np.array([[s] for s in X_test])).numpy()
        p_testable = tok(X_test, padding=True, max_length=100,
                        truncation=True, return_tensors="tf").data
        n_gold = enc.transform(Y_test)
    else:
        l_testable = Xdv
        p_testable = Xdt
        n_gold = Ydb
    # Get weights from training set
    weights = {i: (1 / np.sum(Ytb == i)) * (len(Ytb) / 2.0) for i in range(2)}

    # LSTM
    if lp["enable"]:
        print("LSTM")
        l_model = ldl.create_model(l_emb, lp["adam"], lp["layers"],
                                   lp["nodes"], lp["decrement"])
        l_model = ldl.train_model(l_model, Xtv, Ytb, Xdv, Ydb,
                                  batch_size=16, epochs=lp["epochs"],
                                  weights=weights)
        l_pred = l_model.predict(l_testable)
        evaluate(n_gold, l_pred)

    # Pretrained
    if pp["enable"]:
        print("PRETRAINED")
        p_model = ldl.train_model(ldb.create_model(pp["adam"], pp["model"]),
                                  Xtt, Ytb, Xdt, Ydb, epochs=pp["epochs"],
                                  weights=weights)
        p_pred = p_model.predict(p_testable)["logits"]
        evaluate(n_gold, p_pred)
