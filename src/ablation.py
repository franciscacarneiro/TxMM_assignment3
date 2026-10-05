# leave-one-group-out ablation over feature categories.

import matplotlib.pyplot as plt

from src.evaluation import f1, macro_f1
from src.model import classify


def _drop(features, columns):
    # remove the given column positions from every feature vector
    drop = set(columns)
    return [[v for i, v in enumerate(row) if i not in drop] for row in features]


def _score(train_features, train_labels, test_features, test_labels):
    pred = classify(train_features, train_labels, test_features)
    return {"micro_f1": f1(test_labels, pred), "macro_f1": macro_f1(test_labels, pred)}


def ablate(groups, train_features, train_labels, test_features, test_labels):
    # baseline with all features, then one run with each feature group left out
    scores = {"all": _score(train_features, train_labels, test_features, test_labels)}
    for name, columns in groups.items():
        scores[name] = _score(
            _drop(train_features, columns), train_labels, _drop(test_features, columns), test_labels
        )

    return scores


def plot_ablation(scores, ax=None):
    # bar per left-out group, with the full-feature score as a dashed baseline
    baseline = scores["all"]["macro_f1"]
    names = [name for name in scores if name != "all"]
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    ax.bar(names, [scores[name]["macro_f1"] for name in names])
    ax.axhline(baseline, color="black", linestyle="--", label=f"All features ({baseline:.3f})")
    ax.set_xlabel("Feature group left out")
    ax.set_ylabel("Macro-F1")
    ax.legend()
    return ax
