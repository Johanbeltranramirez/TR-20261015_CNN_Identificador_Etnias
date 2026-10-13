"""Carga y preprocesamiento del dataset UTKFace."""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import BATCH_SIZE, IMG_SIZE, NUM_CLASSES, SEED

FILENAME_RE = re.compile(r"^(?P<age>\d+)_(?P<gender>[01])_(?P<race>[0-4])_.*\.jpg$", re.IGNORECASE)


def parse_utkface_file(path: Path):
    """Devuelve (age, gender, race) o None si el nombre no cumple el formato."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group("age")), int(m.group("gender")), int(m.group("race"))


def scan_dataset(data_dir: str | Path):
    """Recorre data_dir y devuelve listas (paths, races, ages, genders)."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.jpeg")) + sorted(data_dir.glob("*.png"))
    paths, races, ages, genders = [], [], [], []
    for f in files:
        parsed = parse_utkface_file(f)
        if parsed is None:
            continue
        age, gender, race = parsed
        paths.append(str(f))
        races.append(race)
        ages.append(age)
        genders.append(gender)
    return paths, np.array(races), np.array(ages), np.array(genders)


def stratified_split(paths, races, val_size=0.15, test_size=0.15, seed=SEED):
    """Division estratificada train/val/test. Devuelve 3 tuplas (paths, labels)."""
    p_train, p_temp, y_train, y_temp = train_test_split(
        paths, races, test_size=val_size + test_size, random_state=seed, stratify=races
    )
    rel_test = test_size / (val_size + test_size)
    p_val, p_test, y_val, y_test = train_test_split(
        p_temp, y_temp, test_size=rel_test, random_state=seed, stratify=y_temp
    )
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


def _decode(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, try_recover_truncated=True)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label


def get_augmentation_layer():
    from tensorflow import keras
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.12),
            keras.layers.RandomContrast(0.12),
            keras.layers.RandomTranslation(0.08, 0.08),
        ],
        name="augmentation",
    )


def make_tf_dataset(paths, labels, training: bool, batch_size=BATCH_SIZE, augment: bool = False):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(paths), 4096), seed=SEED)
    if augment and training:
        aug = get_augmentation_layer()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    # Normalizacion: [0,255] -> [0,1]
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets_from_folder(data_dir, batch_size=BATCH_SIZE, val_size=0.15, test_size=0.15, augment=True):
    paths, races, _, _ = scan_dataset(data_dir)
    if len(paths) == 0:
        raise FileNotFoundError(f"No se encontraron imagenes UTKFace en {data_dir}")
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = stratified_split(paths, races, val_size, test_size)
    train_ds = make_tf_dataset(p_tr, y_tr, training=True, batch_size=batch_size, augment=augment)
    val_ds = make_tf_dataset(p_va, y_va, training=False, batch_size=batch_size)
    test_ds = make_tf_dataset(p_te, y_te, training=False, batch_size=batch_size)
    meta = {"train_n": len(p_tr), "val_n": len(p_va), "test_n": len(p_te),
            "train_paths": p_tr, "val_paths": p_va, "test_paths": p_te,
            "y_train": y_tr, "y_val": y_va, "y_test": y_te}
    return train_ds, val_ds, test_ds, meta
