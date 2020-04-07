Code for the research paper 'Subreddit classification of text posts' for the TxMM'19-'20 course. This repository contains the following important modules:

* `crawler.py` is the script that retrieves the dataset using the Python Reddit API Wrapper.
* `statistics.py` is a utility module in which we retrieve several statistics from the dataset. For instance, it computes the most discriminative terms for each subreddit.
* `document.py` contains a wrapper for documents, which performs simple tasks like tokenization and POS tagging. This way, these operations don't have to be repeated for each feature type that uses them.
* `features.py` contains a list of all feature types and the means to extract them from a document. It uses both the `statistics` and `document` helper module to compute each feature vector.
* `main.py` contains the actual experiments, in which we create the models and actually perform our classification.
