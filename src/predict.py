"""Inferencia con el modelo entrenado (en español).

Antes de predecir se recorta automáticamente el rostro: el modelo se entrenó
con rostros ajustados (UTKFace) y las fotos con hombros, ropa o fondo lo
desorientan. Si no se detecta rostro, se analiza la imagen completa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CLASS_NAMES, IMG_SIZE
from .utils import load_class_names

_CASCADA = None


def _cascada():
    global _CASCADA
    if _CASCADA is None:
        import cv2
        _CASCADA = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADA


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


def recortar_rostro(img: Image.Image, margen=0.25):
    """Recorta el rostro principal en cuadrado con margen.

    Devuelve (imagen, encontrado). Si OpenCV no está instalado o no hay
    rostro, devuelve la imagen original y False.
    """
    try:
        import cv2
    except ImportError:
        return img, False
    cascada = _cascada()
    if cascada.empty():
        return img, False
    base = img.convert("RGB")
    gris = cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2GRAY)
    rostros = cascada.detectMultiScale(gris, scaleFactor=1.08, minNeighbors=4,
                                       minSize=(30, 30))
    if len(rostros) == 0:
        return img, False
    x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])
    cx, cy = x + w / 2, y + h / 2
    lado = max(w, h) * (1 + margen)
    x0 = int(max(0, cx - lado / 2))
    y0 = int(max(0, cy - lado / 2))
    x1 = int(min(base.width, cx + lado / 2))
    y1 = int(min(base.height, cy + lado / 2))
    if x1 <= x0 or y1 <= y0:
        return img, False
    return base.crop((x0, y0, x1, y1)), True


def predict_image(model, class_names, img: Image.Image):
    cara, _ = recortar_rostro(img)
    x = preprocess_pil(cara, model=model)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs
