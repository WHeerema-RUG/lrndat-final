#!/usr/bin/env python
# MAIN PROGRAM
# Compare four models for binary text classification:
# Import text, get labels, produce Precision, Recall and F1 scores
# Author: Wessel Heerema
# Latest build: 17/10/2024

import argparse
import json
from sklearn.metrics import precision_recall_fscore_support

import baseline as ldb


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
    '''Extract docs and labels in a binary classification task'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            if not unseen:
                documents.append(tokens[:-1])
                labels.append(tokens[-1])
            else:
                documents.append(tokens)
    return documents, labels


def evaluate(gold, pred):
    '''Perform Precision, Recall and F1-score test'''
    prc, rec, f1, _ = precision_recall_fscore_support(gold, pred)
    print(f"Precision: {prc}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    args = create_arg_parser()

    # Read files
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.train_file)

    baseline_pred = ldb.base_classifier(X_train, Y_train, X_dev)
    evaluate(Y_dev, baseline_pred)
