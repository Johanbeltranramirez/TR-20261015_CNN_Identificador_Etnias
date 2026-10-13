"""Inferencia con el modelo entrenado."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CLASS_NAMES, IMG_SIZE
from .utils import load_class_names


def load_model_and_labels(model_path, class_names_path=None):
    model = tf.keras.models.load_model(model_path)
    if class_names_path and Path(class_names_path).exists():
        class_names = load_class_names(class_names_path)
    else:
        class_names = CLASS_NAMES
    return model, class_names


def model_input_size(model, default=IMG_SIZE):
    try:
        shape = model.input_shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 3 and shape[1]:
            return int(shape[1])
    except Exception:
        pass
    return default


def preprocess_pil(img: Image.Image, img_size=None, model=None):
    if img_size is None:
        img_size = model_input_size(model) if model is not None else IMG_SIZE
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)


def predict_image(model, class_names, img: Image.Image):
    x = preprocess_pil(img, model=model)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs
