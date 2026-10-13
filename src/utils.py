"""Utilidades: plots, metricas, guardado."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def save_class_names(path, class_names):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")


def load_class_names(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_history(history, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, acc, label="train acc")
    ax[0].plot(epochs, val_acc, label="val acc")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoca")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(epochs, loss, label="train loss")
    ax[1].plot(epochs, val_loss, label="val loss")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoca")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, normalize=False):
    import seaborn as sns
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion" + (" (normalizada)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def print_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(report)
    return report
