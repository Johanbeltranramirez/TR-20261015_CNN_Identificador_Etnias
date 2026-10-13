"""Entrenamiento de la CNN para clasificacion de etnias (UTKFace).

Uso tipico (Colab / local):
    python -m src.train --data-dir data/raw/UTKFace --epochs 25 --model transfer
    python -m src.train --demo --epochs 5   # sin imágenes: datos sintéticos para probar el flujo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import (BATCH_SIZE, CLASS_NAMES, CLASS_NAMES_ES, CLASS_NAMES_PATH, EPOCHS, IMG_SIZE,
                     MODEL_PATH, RESULTS_DIR)
from .dataset import build_datasets_from_folder, make_tf_dataset
from .model import build_model
from .utils import plot_confusion_matrix, plot_history, print_report, save_class_names


def make_demo_datasets(num_per_class=60, img_size=64, batch_size=32):
    """Datos sintéticos (ruido estructurado por clase) para validar el flujo sin UTKFace."""
    rng = np.random.default_rng(42)
    n_classes = len(CLASS_NAMES)
    paths_train, paths_val, paths_test = [], [], []
    # Generamos arrays en memoria y los servimos con tf.data (sin archivos).
    def gen(n):
        X = rng.normal(0.5, 0.2, size=(n * n_classes, img_size, img_size, 3)).astype("float32")
        y = np.repeat(np.arange(n_classes), n)
        # Tinte por clase para que sea aprendible
        tints = np.array([[0.9, 0.7, 0.7], [0.7, 0.9, 0.7], [0.7, 0.7, 0.9],
                          [0.9, 0.9, 0.7], [0.8, 0.7, 0.9]], dtype="float32")
        X = np.clip(X * 0.5 + 0.5 * tints[y][:, None, None, :], 0, 1)
        return X, y
    Xtr, ytr = gen(num_per_class)
    Xva, yva = gen(num_per_class // 4)
    Xte, yte = gen(num_per_class // 4)
    train_ds = tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(2048, seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((Xte, yte)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    meta = {"train_n": len(Xtr), "val_n": len(Xva), "test_n": len(Xte),
            "y_test": yte, "demo": True}
    return train_ds, val_ds, test_ds, meta


def plot_sample_predictions(model, test_ds, class_names, out_path, n=12):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images, labels = [], []
    for x, y in test_ds.unbatch().take(n):
        images.append(x.numpy())
        labels.append(int(y.numpy()))
    images = np.stack(images)
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    nombres_es = [CLASS_NAMES_ES.get(c, c) for c in class_names]
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.atleast_1d(axes).flatten()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(np.clip(images[i], 0, 1))
        ok = preds[i] == labels[i]
        ax.set_title(f"Real: {nombres_es[labels[i]]}\nPredicha: {nombres_es[preds[i]]} ({probs[i][preds[i]]:.2f})",
                     fontsize=8, color="green" if ok else "red")
        ax.axis("off")
    fig.suptitle("Predicciones de ejemplo (conjunto de prueba)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Entrenamiento de la CNN (UTKFace) — interfaz en español.")
    ap.add_argument("--data-dir", default=None, help="Carpeta con imágenes *.jpg de UTKFace")
    ap.add_argument("--demo", action="store_true", help="Usar datos sintéticos de prueba (sin imágenes)")
    ap.add_argument("--model", choices=["transfer", "simple"], default="transfer",
                    help="Arquitectura: transfer (MobileNetV2) o simple (desde cero)")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="Número de épocas")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Tamaño de lote")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE, help="Tamaño de imagen (px)")
    ap.add_argument("--no-augment", action="store_true", help="Desactivar el aumento de datos")
    ap.add_argument("--model-out", default=str(MODEL_PATH), help="Ruta de salida del modelo")
    ap.add_argument("--demo-n", type=int, default=60, help="Muestras por clase en modo demo")
    ap.add_argument("--class-names-out", default=str(CLASS_NAMES_PATH), help="Ruta del JSON de clases")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="Carpeta de resultados")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    if args.demo or not args.data_dir:
        print(">> Modo de demostración: datos sintéticos de prueba.")
        train_ds, val_ds, test_ds, meta = make_demo_datasets(
            num_per_class=args.demo_n, batch_size=args.batch_size, img_size=args.img_size)
        class_names = CLASS_NAMES
    else:
        print(f">> Cargando UTKFace desde {args.data_dir}")
        train_ds, val_ds, test_ds, meta = build_datasets_from_folder(
            args.data_dir, batch_size=args.batch_size, augment=not args.no_augment,
            img_size=args.img_size)
        print(f"Conjuntos: entrenamiento={meta['train_n']} validación={meta['val_n']} prueba={meta['test_n']}")
        class_names = CLASS_NAMES

    # Pesos por clase ante el desbalance (UTKFace concentra la mayoría en White)
    if not meta.get("demo"):
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=meta["y_train"])
        class_weight = {i: float(w) for i, w in enumerate(cw)}
        print("Pesos por clase:", class_weight)
    else:
        class_weight = None

    weights = None if args.demo else "imagenet"
    if args.model == "transfer":
        model = build_model(args.model, img_size=args.img_size, weights=weights)
    else:
        model = build_model("simple", img_size=args.img_size)
    # build_model con weights=None en demo evita descargar ImageNet si no hay internet
    model.summary(print_fn=print)

    cbs = [
        keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True, monitor="val_accuracy", mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.CSVLogger(str(results_dir / "training_log.csv")),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=class_weight, callbacks=cbs, verbose=1)
    model.save(args.model_out)
    save_class_names(args.class_names_out, class_names)
    print(f"Modelo guardado en {args.model_out}")

    plot_history(history, results_dir / "training_history.png")

    # Evaluación en el conjunto de prueba
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"Prueba: pérdida={loss:.4f} exactitud={acc:.4f}")
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)
    print_report(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names, results_dir / "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, results_dir / "confusion_matrix_norm.png", normalize=True)
    plot_sample_predictions(model, test_ds, class_names, results_dir / "sample_predictions.png", n=12)
    (results_dir / "metrics.txt").write_text(f"perdida={loss:.4f}\nexactitud={acc:.4f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
