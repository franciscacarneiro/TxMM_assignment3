# leave-one-group-out ablation over feature categories.

import matplotlib.pyplot as plt
from sklearn.base import clone

from src.evaluation import f1, macro_f1
from src.model import Classifier


def _drop(features, columns):
    # remove the given column positions from every feature vector
    drop = set(columns)
    return [[v for i, v in enumerate(row) if i not in drop] for row in features]


def _score(train_features, train_labels, test_features, test_labels, clf=None):
    pred = Classifier(clone(clf) if clf is not None else None).train(train_features, train_labels).predict(test_features)
    return {"micro_f1": f1(test_labels, pred), "macro_f1": macro_f1(test_labels, pred)}


def ablate(groups, train_features, train_labels, test_features, test_labels, clf=None):
    # baseline with all features, then one run with each feature group left out
    scores = {"all": _score(train_features, train_labels, test_features, test_labels, clf)}
    for name, columns in groups.items():
        scores[name] = _score(
            _drop(train_features, columns), train_labels, _drop(test_features, columns), test_labels, clf
        )

    return scores


def plot_ablation(scores, ax=None):
    # get all ablation groups
    groups = sorted((n for n in scores if n != "all"), key=lambda n: -scores[n]["macro_f1"])

    # add "all" to group names
    names = ["all", *groups]

    # Get all scores and labels
    values = [scores[n]["macro_f1"] for n in names]
    labels = ["None", *(n.replace("_", " ") for n in groups)]

    # allow using in subplot later, or default to single plot
    ax = ax or plt.subplots(figsize=(5, 3))[1]

    # Set colours, unique for all
    colors = ["#52514e"] + ["#2a78d6"] * len(groups)

    # Add bar chart
    ax.bar(range(len(names)), values, width=0.65, color=colors)

    # Add horizontal line at height of all
    ax.axhline(values[0], color="#52514e", linewidth=0.8, linestyle="--", zorder=0)

    # Add labels inside bars
    for x, v in enumerate(values):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, -3), textcoords="offset points",
                    ha="center", va="top", fontsize=6.5, color="white")

    # zoomed y-axis
    ax.set_ylim(round(min(values) - 0.05, 2), round(max(values) + 0.01, 2))

    # styling
    ax.set_xticks(range(len(names)), labels=labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Removed feature group")
    ax.set_ylabel("Macro-F1 (dev)")
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax
