# how to turn a text into a feature vector.

from nltk.tokenize import wordpunct_tokenize


def extract_features(text):
    # num-tokens
    return [len(wordpunct_tokenize(text))]
