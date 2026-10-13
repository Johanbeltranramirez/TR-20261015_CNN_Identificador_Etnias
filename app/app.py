"""App Streamlit: clasificador de etnias (CNN UTKFace).

Ejecucion:
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import CLASS_NAMES_ES, IMG_SIZE

MODEL_CANDIDATES = [
    ROOT / "model" / "ethnicity_cnn.h5",
    ROOT / "model" / "ethnicity_cnn.keras",
    ROOT / "ethnicity_cnn.h5",
]
CLASS_CANDIDATES = [
    ROOT / "model" / "class_names.json",
    ROOT / "class_names.json",
]

ES_ORDER = ["White", "Black", "Asian", "Indian", "Others"]


@st.cache_resource
def load_assets():
    import tensorflow as tf
    model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "No se encontro el modelo. Esperado en: "
            + ", ".join(str(p) for p in MODEL_CANDIDATES)
        )
    model = tf.keras.models.load_model(model_path)
    class_names = ES_ORDER
    for c in CLASS_CANDIDATES:
        if c.exists():
            class_names = json.loads(c.read_text(encoding="utf-8"))
            break
    return model, class_names, model_path


def preprocess(img: Image.Image, model=None):
    size = IMG_SIZE
    try:
        shape = model.input_shape if model is not None else None
        if shape and len(shape) >= 3 and shape[1]:
            size = int(shape[1])
    except Exception:
        pass
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0), img


st.set_page_config(page_title="CNN Identificador de Etnias", page_icon="🧬", layout="centered")
st.title("🧬 CNN — Identificador de Etnias Culturales")
st.caption("Clasificacion de rasgos faciales en 5 grupos del dataset UTKFace (White, Black, Asian, Indian, Others). Uso educativo.")

with st.expander("ℹ️ Sobre el modelo y uso responsable", expanded=False):
    st.markdown(
        "- **Dataset:** UTKFace (Aligned & Cropped, ~20k rostros, solo investigacion no comercial).\n"
        "- **Modelo:** MobileNetV2 (transfer learning) + cabeza densa, entrada 128×128, Data Augmentation.\n"
        "- **Advertencia:** la etnia no se reduce a rasgos fisicos; el modelo refleja sesgos del dataset "
        "(mayoria White) y **no debe usarse** para discriminar, vigilar o tomar decisiones sobre personas."
    )

try:
    model, class_names, model_path = load_assets()
    st.success(f"Modelo cargado: `{model_path.name}` ({len(class_names)} clases)")
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo cargar el modelo: {e}")
    st.info("Entrena con `python -m src.train --data-dir data/raw/UTKFace` o `python -m src.train --demo` y reintenta.")
    st.stop()

uploaded = st.file_uploader("Sube una foto de rostro (jpg/png)", type=["jpg", "jpeg", "png"])
use_camera = st.checkbox("O usa la camara", value=False)
cam_img = st.camera_input("Toma una foto") if use_camera else None

img_file = cam_img if cam_img is not None else uploaded

if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption="Imagen ingresada", use_container_width=True)
    x, preview = preprocess(img, model)
    with st.spinner("Prediciendo..."):
        probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label_en = class_names[idx]
    label_es = CLASS_NAMES_ES.get(label_en, label_en)
    st.metric("Clase predicha", f"{label_en} — {label_es}", f"{probs[idx]*100:.2f}% confianza")
    st.progress(float(probs[idx]), text="Nivel de confianza")
    st.subheader("Probabilidades por clase")
    import pandas as pd
    df = pd.DataFrame({
        "Clase": class_names,
        "Descripcion": [CLASS_NAMES_ES.get(c, c) for c in class_names],
        "Probabilidad": probs,
    }).sort_values("Probabilidad", ascending=False)
    st.bar_chart(df.set_index("Clase")["Probabilidad"])
    st.dataframe(df.style.format({"Probabilidad": "{:.4f}"}), use_container_width=True)
else:
    st.info("Sube una imagen para obtener la prediccion + probabilidad por clase.")
    st.caption("Tip: usa fotos frontales, buena luz, un solo rostro (como UTKFace Aligned & Cropped).")
