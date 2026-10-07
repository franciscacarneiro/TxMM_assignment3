# how to turn a text into a feature vector.
# features are grouped into categories; src/ablation.py leaves one group out at a time.

import re
import string
from collections import Counter
from math import log1p
from statistics import pstdev

import nltk

# the corpus writes apostrophes as '"' (I"m, didn"t); a quote between two letters is one
CONTRACTION = r'(?<=[A-Za-z])"(?=[A-Za-z])'

# coarse POS classes
POS_CLASSES = ["NN", "VB", "JJ", "RB", "PR", "DT", "IN", "CC", "MD", "TO", "W"]
FUNCTION_WORDS = [
    "the", "a", "an", "and", "but", "or", "so", "if", "then", "than",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "as",
    "that", "this", "there", "not", "no", "all", "just", "very", "only", "too",
    "was", "were", "is", "be", "been", "had", "have", "would", "could", "can",
]
PRONOUNS = {
    "first_person": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"},
    "second_person": {"you", "your", "yours", "yourself", "yourselves"},
    "third_person": {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"},
}
PUNCTUATION = {
    "comma": ",", "period": ".", "question": "?", "exclamation": "!",
    "semicolon": ";", "colon": ":", "apostrophe": "'", "quote": '"',
    "hyphen": "-", "open_parenthesis": "(", "close_parenthesis": ")",
}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _analyze(text):
    # tokenize/tag once per text; the feature functions reuse this
    text = re.sub(CONTRACTION, "'", text)
    sentences = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(text)]
    tokens = [t for s in sentences for t in s]
    words = [t.lower() for t in tokens if t.isalpha()]
    sentence_lengths = [sum(token.isalpha() for token in sentence) for sentence in sentences]
    sentence_lengths = [length for length in sentence_lengths if length]
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    return {
        "text": text,
        "words": words,
        "n_words": len(words),
        "counts": Counter(words),
        "sent_lengths": sentence_lengths,
        "n_paragraphs": len(paragraphs),
        "tags": Counter(tag for _, tag in nltk.pos_tag(tokens)),
        "n_tokens": len(tokens),
    }


def _type_token_ratio(ctx):
    return _ratio(len(ctx["counts"]), ctx["n_words"])


def _hapax_ratio(ctx):
    # words used exactly once / all words
    return _ratio(sum(c == 1 for c in ctx["counts"].values()), ctx["n_words"])


def _avg_word_length(ctx):
    return _ratio(sum(len(w) for w in ctx["words"]), ctx["n_words"])


def _long_word_ratio(ctx):
    return _ratio(sum(len(w) >= 7 for w in ctx["words"]), ctx["n_words"])


def _short_word_ratio(ctx):
    return _ratio(sum(len(w) <= 3 for w in ctx["words"]), ctx["n_words"])


def _avg_sentence_length(ctx):
    return _ratio(sum(ctx["sent_lengths"]), len(ctx["sent_lengths"]))


def _sentence_length_std(ctx):
    lengths = ctx["sent_lengths"]
    return pstdev(lengths) if len(lengths) > 1 else 0.0


def _pos_rate(prefix):
    # share of tokens whose Penn tag starts with prefix
    def feature(ctx):
        count = sum(n for tag, n in ctx["tags"].items() if tag.startswith(prefix))
        return _ratio(count, ctx["n_tokens"])
    return feature


def _word_rate(words):
    def feature(ctx):
        return _ratio(sum(ctx["counts"][w] for w in words), ctx["n_words"])
    return feature


def _character_rate(character):
    def feature(ctx):
        return _ratio(ctx["text"].lower().count(character), len(ctx["text"]))
    return feature


def _uppercase_ratio(ctx):
    return _ratio(sum(character.isupper() for character in ctx["text"]), len(ctx["text"]))


def _digit_ratio(ctx):
    return _ratio(sum(character.isdigit() for character in ctx["text"]), len(ctx["text"]))


def _whitespace_ratio(ctx):
    return _ratio(sum(character.isspace() for character in ctx["text"]), len(ctx["text"]))


def _newline_ratio(ctx):
    return _ratio(ctx["text"].count("\n"), len(ctx["text"]))


def _log_word_count(ctx):
    return log1p(ctx["n_words"])


def _log_sentence_count(ctx):
    return log1p(len(ctx["sent_lengths"]))


def _paragraph_count(ctx):
    return ctx["n_paragraphs"]


def _word_length_rate(length):
    def feature(ctx):
        count = sum(len(word) == length for word in ctx["words"])
        return _ratio(count, ctx["n_words"])
    return feature


def _long_word_bucket(ctx):
    return _ratio(sum(len(word) >= 10 for word in ctx["words"]), ctx["n_words"])


def _sentence_length_rate(minimum, maximum=None):
    def feature(ctx):
        lengths = ctx["sent_lengths"]
        count = sum(
            length >= minimum and (maximum is None or length <= maximum)
            for length in lengths
        )
        return _ratio(count, len(lengths))
    return feature


# (group, name, function)
FEATURES = []

FEATURES += [("letters", f"letter_{letter}", _character_rate(letter)) for letter in string.ascii_lowercase]
FEATURES += [("punctuation", name, _character_rate(mark)) for name, mark in PUNCTUATION.items()]
FEATURES += [
    ("character_style", "uppercase_ratio", _uppercase_ratio),
    ("character_style", "digit_ratio", _digit_ratio),
    ("character_style", "whitespace_ratio", _whitespace_ratio),
    ("character_style", "newline_ratio", _newline_ratio),
]
FEATURES += [
    ("document_structure", "log_word_count", _log_word_count),
    ("document_structure", "log_sentence_count", _log_sentence_count),
    ("document_structure", "paragraph_count", _paragraph_count),
]
FEATURES += [("word_length", f"word_length_{length}", _word_length_rate(length)) for length in range(1, 10)]
FEATURES += [
    ("word_length", "word_length_10_plus", _long_word_bucket),
    ("word_length", "avg_word_length", _avg_word_length),
    ("word_length", "short_word_ratio", _short_word_ratio),
    ("word_length", "long_word_ratio", _long_word_ratio),
]
FEATURES += [
    ("lexical", "type_token_ratio", _type_token_ratio),
    ("lexical", "hapax_ratio", _hapax_ratio),
]
FEATURES += [("function_words", f"w_{w}", _word_rate({w})) for w in FUNCTION_WORDS]
FEATURES += [("pronoun", name, _word_rate(words)) for name, words in PRONOUNS.items()]
FEATURES += [
    ("sentence_length", "avg_sentence_length", _avg_sentence_length),
    ("sentence_length", "sentence_length_std", _sentence_length_std),
    ("sentence_length", "short_sentence_ratio", _sentence_length_rate(1, 10)),
    ("sentence_length", "medium_sentence_ratio", _sentence_length_rate(11, 20)),
    ("sentence_length", "long_sentence_ratio", _sentence_length_rate(21)),
]
FEATURES += [("syntactic", f"pos_{prefix}", _pos_rate(prefix)) for prefix in POS_CLASSES]
FEATURE_NAMES = [name for _, name, _ in FEATURES]

# group name -> positions in the feature vector
FEATURE_GROUPS = {}
for i, (group, _, _) in enumerate(FEATURES):
    FEATURE_GROUPS.setdefault(group, []).append(i)


def extract_features(text):
    ctx = _analyze(text)
    return [fn(ctx) for _, _, fn in FEATURES]
