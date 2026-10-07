<<<<<<< HEAD
from nltk.tag import PerceptronTagger
from nltk.tokenize import TreebankWordTokenizer
from pathlib import Path
import numpy as np
import pandas as pd
import nltk

project = Path(__file__).resolve().parent.parent
nltk_dir = project / ".cache/nltk_data"
nltk.data.path.insert(0, str(nltk_dir))
tagger = PerceptronTagger(lang="eng")
tokenizer = TreebankWordTokenizer()

function_words = "a an the and or but if while because as until of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such".split()

letters = list("abcdefghijklmnopqrstuvwxyz")

punctuation = {
    "period": ".",
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "question": "?",
    "exclamation": "!",
    "apostrophe": "'",
    "quote": '"',
    "hyphen": "-",
    "open_parenthesis": "(",
    "close_parenthesis": ")",
    "ellipsis": "…"
}

style_names = [
    "mean_word_length",
    "std_word_length",
    "type_token_ratio",
    "hapax_ratio",
    "mean_sentence_words",
    "std_sentence_words",
    "short_word_ratio",
    "long_word_ratio",
    "uppercase_char_ratio",
    "digit_char_ratio",
    "whitespace_char_ratio",
    "newline_char_ratio"
]

pos_tags = "CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PRP$ RB RBR RBS RP SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WRB".split()

groups = {
    "function_words": function_words,
    "character_frequencies": letters,
    "idiosyncratic": list(punctuation),
    "authorship_stylistic": style_names,
    "pos_tags": pos_tags + ["OTHER"]
}

feature_names = []
feature_groups = {}

for group, names in groups.items():

    feature_groups[group] = []
    for name in names:
        feature_groups[group].append(len(feature_names))
        feature_names.append(group + "_" + name)

        
word_pattern = r"[^\W\d_]+(?:['’][^\W\d_]+)*"


def extract_features(texts):
    rows = []

    for text in texts:

        words = pd.Series([text.lower()]).str.findall(word_pattern).iloc[0]
        counts = pd.Series(words, dtype=str).value_counts()
        word_count = max(len(words), 1)
        char_count = max(len(text), 1)
        row = []

        for word in function_words:
            row.append(counts.get(word, 0) / word_count)

        for letter in letters:
            row.append(text.lower().count(letter) / char_count)

        for mark in punctuation.values():
            row.append(text.count(mark) / char_count)

        lengths = []

        for word in words:
            lengths.append(len(word))
        lengths = np.array(lengths)

        sentences = pd.Series([text]).str.split(r"[.!?]+").explode()
        sentence_words = sentences.str.findall(word_pattern)
        sentence_lengths = sentence_words.apply(len)    
        sentence_lengths = sentence_lengths[sentence_lengths > 0].to_numpy()

        row += [
            lengths.mean() if lengths.size else 0,
            lengths.std() if lengths.size else 0,
            len(counts) / word_count,
            (counts == 1).sum() / word_count,
            sentence_lengths.mean() if sentence_lengths.size else 0,
            sentence_lengths.std() if sentence_lengths.size else 0,
            (lengths <= 3).sum() / word_count,
            (lengths >= 7).sum() / word_count,
            sum(char.isupper() for char in text) / char_count,
            sum(char.isdigit() for char in text) / char_count,
            sum(char.isspace() for char in text) / char_count,
            text.count("\n") / char_count
        ]

        tokens = tokenizer.tokenize(text)
        tagged_words = tagger.tag(tokens)
        tags = []

        for word, tag in tagged_words:
            tags.append(tag)

        tags = pd.Series(tags, dtype=str)
        tag_counts = tags.value_counts()
        token_count = max(len(tokens), 1)

        for tag in pos_tags:
            row.append(tag_counts.get(tag, 0) / token_count)

        row.append((~tags.isin(pos_tags)).sum() / token_count)
        rows.append(row)
        
    return pd.DataFrame(rows, columns=feature_names, dtype=float)

=======
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
>>>>>>> d70ba2db8deff777489307bd1e0493b833ba0b52
