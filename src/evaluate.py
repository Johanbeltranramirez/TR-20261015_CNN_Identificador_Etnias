"""Evaluacion del modelo en el split de test."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES
from .dataset import build_datasets_from_folder
from .utils import load_class_names, plot_confusion_matrix, print_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="model/ethnicity_cnn.h5")
    ap.add_argument("--class-names", default="model/class_names.json")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    class_names = load_class_names(args.class_names) if Path(args.class_names).exists() else CLASS_NAMES
    model = tf.keras.models.load_model(args.model)
    try:
        img_size = int(model.input_shape[1]) or 128
    except (TypeError, IndexError):
        img_size = 128
    _, _, test_ds, meta = build_datasets_from_folder(
        args.data_dir, batch_size=args.batch_size, augment=False, img_size=img_size)
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"Prueba: pérdida={loss:.4f} exactitud={acc:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)
    print_report(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names, "results/confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, "results/confusion_matrix_norm.png", normalize=True)


if __name__ == "__main__":
    main()
