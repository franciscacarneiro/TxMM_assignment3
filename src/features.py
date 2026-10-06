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

