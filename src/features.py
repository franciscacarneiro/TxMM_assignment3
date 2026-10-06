# how to turn a text into a feature vector.
# features are grouped into categories; src/ablation.py leaves one group out at a time.

import io
import re
import string
import urllib.request
import zipfile
from collections import Counter
from functools import lru_cache
from math import log1p
from pathlib import Path
from statistics import mean, pstdev

import nltk
from joblib import Parallel, delayed
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# contractions
CONTRACTION = r'(?<=[A-Za-z])"(?=[A-Za-z])'

# coarse POS classes
POS_CLASSES = ["NN", "VB", "JJ", "RB", "PR", "DT", "IN", "CC", "MD", "TO", "W"]

PRONOUNS = {
    "first_person": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"},
    "second_person": {"you", "your", "yours", "yourself", "yourselves"},
    "third_person": {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"},
}
PUNCTUATION = {
    "comma": ",", "period": ".", "question": "?", "exclamation": "!",
    "semicolon": ";", "colon": ":",
    "open_parenthesis": "(", "close_parenthesis": ")",
}
# WordNet supersenses (lexicographer files) of nouns and verbs
SUPERSENSES = [f"noun.{s}" for s in [
    "Tops", "act", "animal", "artifact", "attribute", "body", "cognition", "communication", "event",
    "feeling", "food", "group", "location", "motive", "object", "person", "phenomenon", "plant",
    "possession", "process", "quantity", "relation", "shape", "state", "substance", "time",
]] + [f"verb.{s}" for s in [
    "body", "change", "cognition", "communication", "competition", "consumption", "contact",
    "creation", "emotion", "motion", "perception", "possession", "social", "stative", "weather",
]]
# NRC Emotion Lexicon: free for research, but may not be redistributed, so it is downloaded, not committed
NRC_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
NRC_FILE = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
NRC_PATH = Path(__file__).resolve().parent.parent / "lexicons" / Path(NRC_FILE).name
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive", "negative"]

# (name, regex) per group, counted per word
# quotes, apostrophes and hyphens are counted here, not in PUNCTUATION
PATTERNS = {
    "punctuation": [  # dialogue punctuation
        ("dialogue_quote", r'"'),  # contractions are already restored to "'"
        ("said", r"\bsaid\b"),
        ("comma_before_tag", r',"\s[a-z]'),
        ("period_before_tag", r'\."\s[a-z]'),
    ],
    "informal": [
        ("contraction", r"\w'\w"),
        ("ellipsis", r"\.\.\."),
        ("allcaps_word", r"\b[A-Z]{2,}\b"),
        ("cut_off", r"\w-(?=[\s\"]|$)"),
        ("double_hyphen", r"--"),
        ("spaced_hyphen", r" - "),
        ("stutter", r"(?i)\b([a-z])-\1"),
        ("punct_run", r"[!?]{2,}"),
        ("honorific", r"-(?:san|kun|chan|sama|senpai|sensei)\b"),
        ("tilde", r"~"),
        ("lowercase_after_stop", r"[.!?] [a-z]"),
        ("conjunction_start", r"[.!?] (?:And|But|So) "),
        ("scene_break", r"([^\w\s])\1{4,}"),
    ],
}


def download_resources():
    # NLTK data and the NRC lexicon; only downloads what is missing
    for resource in ["punkt_tab", "averaged_perceptron_tagger_eng", "wordnet", "vader_lexicon"]:
        nltk.download(resource, quiet=True)
    if not NRC_PATH.exists():
        NRC_PATH.parent.mkdir(exist_ok=True)
        # the server refuses Python's default user agent
        request = urllib.request.Request(NRC_URL, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(request) as response, zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            NRC_PATH.write_bytes(archive.read(NRC_FILE))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


@lru_cache(maxsize=None)
def _first_sense(word, pos):
    # (supersense, depth in the hypernym tree) of the most frequent sense, or None
    synsets = wn.synsets(word, pos)
    return (synsets[0].lexname(), synsets[0].min_depth()) if synsets else None


@lru_cache(maxsize=None)
def _sense_count(word):
    return len(wn.synsets(word))


@lru_cache(maxsize=None)
def _vader():
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=None)
def _nrc():
    # word -> emotions it is associated with
    lexicon = {}
    for line in NRC_PATH.read_text().splitlines():
        word, emotion, flag = line.split("\t")
        if flag == "1":
            lexicon.setdefault(word, set()).add(emotion)
    return lexicon


def _analyze(text):
    # tokenize/tag once per text; the feature functions reuse this
    text = re.sub(CONTRACTION, "'", text)
    raw_sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(s) for s in raw_sentences]
    tokens = [t for s in sentences for t in s]
    words = [t.lower() for t in tokens if t.isalpha()]
    sentence_lengths = [sum(token.isalpha() for token in sentence) for sentence in sentences]
    sentence_lengths = [length for length in sentence_lengths if length]
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    tagged = nltk.pos_tag(tokens)
    # common nouns and verbs with their first WordNet sense; proper nouns are names, not meaning
    senses = [
        _first_sense(token.lower(), wn.NOUN if tag.startswith("NN") else wn.VERB)
        for token, tag in tagged
        if token.isalpha() and (tag in ("NN", "NNS") or tag.startswith("VB"))
    ]
    senses = [sense for sense in senses if sense]
    nrc = _nrc()
    return {
        "text": text,
        "words": words,
        "n_words": len(words),
        "counts": Counter(words),
        "sent_lengths": sentence_lengths,
        "n_paragraphs": len(paragraphs),
        "tags": Counter(tag for _, tag in tagged),
        "n_tokens": len(tokens),
        "supersenses": Counter(lexname for lexname, _ in senses),
        "n_senses": len(senses),
        "noun_depths": [depth for lexname, depth in senses if lexname.startswith("noun.")],
        "sense_counts": [n for n in map(_sense_count, words) if n],
        "sentiment": [_vader().polarity_scores(s) for s in raw_sentences],
        "emotions": Counter(emotion for word in words for emotion in nrc.get(word, ())),
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


def _pattern_rate(pattern):
    def feature(ctx):
        return _ratio(len(re.findall(pattern, ctx["text"])), ctx["n_words"])
    return feature


def _supersense_rate(lexname):
    # share of the text's nouns and verbs whose first sense is in this supersense
    def feature(ctx):
        return _ratio(ctx["supersenses"][lexname], ctx["n_senses"])
    return feature


def _avg_noun_depth(ctx):
    # deeper in the hypernym tree = more specific nouns
    return mean(ctx["noun_depths"]) if ctx["noun_depths"] else 0.0


def _avg_sense_count(ctx):
    # number of WordNet senses per word: plain, ambiguous words vs precise ones
    return mean(ctx["sense_counts"]) if ctx["sense_counts"] else 0.0


def _sentiment_mean(key):
    def feature(ctx):
        return mean(s[key] for s in ctx["sentiment"]) if ctx["sentiment"] else 0.0
    return feature


def _sentiment_std(ctx):
    scores = [s["compound"] for s in ctx["sentiment"]]
    return pstdev(scores) if len(scores) > 1 else 0.0


def _emotion_rate(emotion):
    def feature(ctx):
        return _ratio(ctx["emotions"][emotion], ctx["n_words"])
    return feature


# (group, name, function)
# comments give the lecture category of each feature: lexical, character, syntactic, semantic, application-specific
FEATURES = []

# character
FEATURES += [("character", f"letter_{letter}", _character_rate(letter)) for letter in string.ascii_lowercase]
# character
FEATURES += [("punctuation", name, _character_rate(mark)) for name, mark in PUNCTUATION.items()]
# character
FEATURES += [
    ("character", "uppercase_ratio", _uppercase_ratio),
    ("character", "digit_ratio", _digit_ratio),
    ("character", "whitespace_ratio", _whitespace_ratio),
    ("character", "newline_ratio", _newline_ratio),
]
# lexical (paragraph_count: application-specific, layout)
FEATURES += [
    ("structure", "log_word_count", _log_word_count),
    ("structure", "log_sentence_count", _log_sentence_count),
    ("structure", "paragraph_count", _paragraph_count),
]
# lexical
FEATURES += [("lexical", f"word_length_{length}", _word_length_rate(length)) for length in range(1, 10)]
# lexical
FEATURES += [
    ("lexical", "word_length_10_plus", _long_word_bucket),
    ("lexical", "avg_word_length", _avg_word_length),
    ("lexical", "short_word_ratio", _short_word_ratio),
    ("lexical", "long_word_ratio", _long_word_ratio),
]
# lexical (vocabulary richness)
FEATURES += [
    ("lexical", "type_token_ratio", _type_token_ratio),
    ("lexical", "hapax_ratio", _hapax_ratio),
]
# lexical (word-list frequencies)
FEATURES += [("syntactic", name, _word_rate(words)) for name, words in PRONOUNS.items()]
# lexical (sentence length)
FEATURES += [
    ("structure", "avg_sentence_length", _avg_sentence_length),
    ("structure", "sentence_length_std", _sentence_length_std),
    ("structure", "short_sentence_ratio", _sentence_length_rate(1, 10)),
    ("structure", "medium_sentence_ratio", _sentence_length_rate(11, 20)),
    ("structure", "long_sentence_ratio", _sentence_length_rate(21)),
]
# syntactic
FEATURES += [("syntactic", f"pos_{prefix}", _pos_rate(prefix)) for prefix in POS_CLASSES]
# application-specific (fanfiction dialogue and informal writing)
FEATURES += [
    (group, name, _pattern_rate(pattern))
    for group, patterns in PATTERNS.items()
    for name, pattern in patterns
]
# semantic
FEATURES += [("semantic", f"supersense_{lexname}", _supersense_rate(lexname)) for lexname in SUPERSENSES]
FEATURES += [
    ("semantic", "avg_noun_depth", _avg_noun_depth),
    ("semantic", "avg_sense_count", _avg_sense_count),
]
FEATURES += [("semantic", f"sentiment_{key}", _sentiment_mean(key)) for key in ["neg", "neu", "pos", "compound"]]
FEATURES += [("semantic", "sentiment_std", _sentiment_std)]
FEATURES += [("semantic", f"emotion_{emotion}", _emotion_rate(emotion)) for emotion in EMOTIONS]
FEATURE_NAMES = [name for _, name, _ in FEATURES]

# group name -> positions in the feature vector
FEATURE_GROUPS = {}
for i, (group, _, _) in enumerate(FEATURES):
    FEATURE_GROUPS.setdefault(group, []).append(i)


def extract_features(text):
    ctx = _analyze(text)
    return [fn(ctx) for _, _, fn in FEATURES]


def extract_all(texts):
    # POS tagging dominates the runtime, so spread the texts over all cores
    return Parallel(n_jobs=-1, batch_size=32)(delayed(extract_features)(text) for text in texts)
