import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="micro")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def confusion(y_true, y_pred, ax=None):
    # Row-normalized confusion matrix.
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    ax = ax or plt.subplots(figsize=(10, 8))[1]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Proportion")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted author")
    ax.set_ylabel("True author")
    return ax
