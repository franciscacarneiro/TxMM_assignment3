# how to turn a text into a feature vector.
# features are grouped into categories; src/ablation.py leaves one group out at a time.

from nltk.tokenize import sent_tokenize, wordpunct_tokenize


def _tokens(text):
    return wordpunct_tokenize(text)


def _num_tokens(text):
    # num-tokens
    return len(_tokens(text))


def _type_token_ratio(text):
    # unique tokens / total tokens
    toks = _tokens(text)
    return len(set(toks)) / len(toks) if toks else 0.0


def _avg_token_length(text):
    # mean characters per token
    toks = _tokens(text)
    return sum(len(t) for t in toks) / len(toks) if toks else 0.0


def _num_sentences(text):
    # num-sentences
    return len(sent_tokenize(text))


def _avg_sentence_length(text):
    # mean tokens per sentence
    sents = sent_tokenize(text)
    return len(_tokens(text)) / len(sents) if sents else 0.0


def _num_commas(text):
    return text.count(",")


def _num_questions(text):
    return text.count("?")


def _num_exclamations(text):
    return text.count("!")


def _num_quotes(text):
    return text.count('"')


def _uppercase_ratio(text):
    # uppercase letters / all letters
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def _digit_ratio(text):
    return sum(c.isdigit() for c in text) / len(text) if text else 0.0


# (group, name, function)
FEATURES = [
    ("lexical", "num_tokens", _num_tokens),
    ("lexical", "type_token_ratio", _type_token_ratio),
    ("lexical", "avg_token_length", _avg_token_length),
    ("syntactic", "num_sentences", _num_sentences),
    ("syntactic", "avg_sentence_length", _avg_sentence_length),
    ("punctuation", "num_commas", _num_commas),
    ("punctuation", "num_questions", _num_questions),
    ("punctuation", "num_exclamations", _num_exclamations),
    ("punctuation", "num_quotes", _num_quotes),
    ("character", "uppercase_ratio", _uppercase_ratio),
    ("character", "digit_ratio", _digit_ratio),
]

FEATURE_NAMES = [name for _, name, _ in FEATURES]

# group name -> positions in the feature vector
FEATURE_GROUPS = {}
for i, (group, _, _) in enumerate(FEATURES):
    FEATURE_GROUPS.setdefault(group, []).append(i)


def extract_features(text):
    return [fn(text) for _, _, fn in FEATURES]
