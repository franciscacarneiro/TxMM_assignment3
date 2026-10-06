import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="micro")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


# one-hue sequential ramp; starts light-but-visible so small error rates still show
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
)
BLUES.set_bad("#f0efec")  # zero cells


def author_codes(authors):
    # short letter per author id (A, B, ...), in sorted-id order so it is stable across runs
    return {author: chr(ord("A") + i) for i, author in enumerate(sorted(set(authors)))}


def confusion(y_true, y_pred, ax=None):
    # Row-normalized confusion matrix in percent.
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true") * 100

    # allow using in subplot later, or default to single plot
    ax = ax or plt.subplots(figsize=(5.5, 5))[1]

    # confusion matrix
    mesh = ax.pcolormesh(np.ma.masked_equal(cm, 0), cmap=BLUES, vmin=0, vmax=100,
                         edgecolors="white", linewidth=1)

    # Add color bar
    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("% of the true author's snippets")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)

    # fix surrounding
    codes = author_codes(labels)
    names = [codes[label] for label in labels]
    ticks = [k + 0.5 for k in range(len(labels))]
    ax.set_xticks(ticks, labels=names)
    ax.set_yticks(ticks, labels=names)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Predicted author")
    ax.set_ylabel("True author")
    return ax
