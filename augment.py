#!/usr/bin/env python
# DATA AUGMENTATION
# Augment the data by way of named entity masking
# Author: Wessel Heerema
# Latest build: 29/10/2024

import argparse
import re
import spacy
from tqdm import tqdm


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", default='data/train.tsv', type=str,
                        help="File to augment data for"
                        "(default data/train.tsv)")
    parser.add_argument("-o", "--outfile", default='augmented.tsv', type=str,
                        help="File to save augmented data to"
                        "(default augmented.tsv)")
    args = parser.parse_args()
    return args


def iter_augment(records):
    """Iteratively augment the data to masked named entities and hashtags"""
    ner = spacy.load("en_core_web_sm")
    # Set words that need not be masked
    whitelist = ["CARDINAL", "DATE", "FAC", "MONEY", "ORDINAL", "PERCENT",
                 "QUANTITY", "TIME"]
    # Initialize new list for when they are all processed
    new_records = []
    for text in tqdm(records):
        # Copy text for data integrity
        new_text = text
        # Perform NER
        detected = ner(new_text)
        for word in detected.ents:
            if word.label_ not in whitelist:
                new_text = re.sub(re.escape(word.text), word.label_, new_text)
        # Remove hashtags, works better here
        new_text = re.sub("#\w+", "#HASHTAG", new_text)
        new_records.append(new_text)
    return new_records


def main(infile, outfile):
    """Augment a labeled file"""
    # Initialize output lists
    documents = []
    labels = []
    with open(infile, encoding='utf-8') as f1:
        for line in f1:
            tokens = line.strip().split()
            documents.append(" ".join(tokens[:-1]).strip())
            labels.append(tokens[-1])
    aug_docs = iter_augment(documents)
    outtext = ""
    for i in range(len(labels)):
        outtext += aug_docs[i] + "\t" + labels[i] + "\n"
    with open(outfile, "w") as f2:
        f2.write(outtext)


if __name__ == "__main__":
    args = create_arg_parser()
    main(args.infile, args.outfile)
