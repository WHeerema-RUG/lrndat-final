# lrndat-final

This repository is currently a work-in-progress.

Run the program using this command:
`python3 main.py`
While there are several options available, it should be able to run out of the box if you have at least `train.tsv` and `dev.tsv` in the folder `data/`.

## CURRENT ISSUES
Several days of work have not remedied the following issues:
- LSTM outputs only 1 for every record.
- BERT outputs only 0, unless the learning rate is higher (0.01), in which case it also only outputs 1.

### Diagnostics
- Gold data is legible and logical in all encodings, but its correctness for use in TensorFlow is uncertain.
- The baseline and optimized classifiers output sensible per-class F-scores (~0.8, ~0.5).
- Most code is copied from previous assignments. To my knowledge, all necessary adjustments for a binary classification task have been identified and implemented.
- Adding class weights and adjusting the learning rate have not remedied this issue so far.