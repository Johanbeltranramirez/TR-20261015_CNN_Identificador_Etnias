# CNN Identificador de Etnias Culturales 🧬

Clasificación de imágenes de rostros en **5 grupos étnicos del conjunto de datos UTKFace** (Blanca, Negra, Asiática, India y Otras) con una **Red Neuronal Convolucional (CNN)** + aprendizaje por transferencia (MobileNetV2). Toda la interfaz del sistema está en **español**.

> ⚠️ **Uso responsable:** la etnia no se reduce a rasgos físicos. Este proyecto es **educativo**, refleja los sesgos del conjunto de datos (mayoría de clase *Blanca*) y **no debe usarse** para discriminar, vigilar ni tomar decisiones sobre personas.

## 1. Contenido del repositorio

```
├── app/app.py                  # Aplicación Streamlit en español (carga modelo .h5 y predice + probabilidad)
├── src/
│   ├── config.py               # Clases, IMG_SIZE=128, rutas
│   ├── dataset.py              # Parseo UTKFace, split estratificado, tf.data + normalización
│   ├── model.py                # CNN simple + MobileNetV2 transfer
│   ├── train.py                # Entrenamiento, gráficas, matriz de confusión, 12 predicciones
│   ├── evaluate.py             # Evaluación en test
│   ├── extraer_muestra.py      # Descarga del espejo UTKFace y extracción balanceada a /data
│   └── predict.py              # Inferencia para una imagen
├── notebooks/
│   └── experimentacion.ipynb   # Notebook de experimentación (Google Colab)
├── model/
│   ├── ethnicity_cnn.h5        # Modelo entrenado (CNN simple 128px con 700 imágenes reales;
│   │                           #  re-entrena en Colab con más datos + transfer para producción)
│   └── class_names.json        # ["White","Black","Asian","Indian","Others"]
├── results/                    # training_history.png, confusion_matrix*.png, sample_predictions.png
├── data/
│   ├── raw/UTKFace/*.jpg       # conjunto de trabajo real (700 imágenes, ignorado por git)
│   ├── muestra/UTKFace/*.jpg   # muestra real incluida en el repo (125 imágenes)
│   └── README.md               # Fuentes y regeneración de datos
├── requirements.txt
└── README.md
```

## 2. Dataset: UTKFace

- **Fuente:** [UTKFace — Large Scale Face Dataset](https://susanqq.github.io/UTKFace/) (Zhang et al., CVPR 2017). Solo investigación no comercial.
- **Versión usada:** *Aligned & Cropped Faces* (~107 MB, ~20k+ rostros, un rostro por imagen, variada pose/iluminación/edad 0–116).
- **Etiquetas en el nombre:** `[age]_[gender]_[race]_[datetime].jpg`
  - `age`: 0–116 · `gender`: 0=hombre, 1=mujer · `race`: `0=White, 1=Black, 2=Asian, 3=Indian, 4=Others (Hispanic/Latino/Middle Eastern)`.
- **Por qué sirve:** es el benchmark estándar para etnia/edad/género, ya recortado a rostro, con desbalance real (mayoría Blanca) que obliga a usar pesos por clase y aumento de datos.
- **Distribución aprox.:** Blanca ~40%, Negra ~19%, Asiática ~15%, India ~15%, Otras ~11% (ver EDA en el notebook).

### Obtención de datos (fuente funcional)

> ⚠️ El enlace original de Drive (`.../0BxYys69jI14kU0I1YUQyY1ZDRUE`) está
> caído (devuelve 404). Fuente verificada: espejo
> **Hugging Face `vtsouval/utkface-cropped`** (parquet con imagen + edad +
> género + etnia, 23.704 filas).

```bash
pip install -r requirements.txt
python -m src.extraer_muestra --descargar
# Opciones: --cuota-raw 140 --cuota-muestra 25
```

Esto genera `data/raw/UTKFace/*.jpg` (trabajo) y `data/muestra/UTKFace/*.jpg`
(muestra del repo) con nombres en formato `25_0_2_2024010100000000001.jpg`.
Detalle en `data/README.md`. El repo ya incluye la muestra (`data/muestra/`),
así que puedes entrenar sin descargar nada (ver abajo).

## 3. Instalación y ejecución local

```bash
git clone <este-repo> && cd TR-20261015_CNN_Identificador_Etnias
python -m venv .venv; .\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

### Entrenar (con UTKFace real)
```bash
python -m src.train --data-dir data/raw/UTKFace --epochs 25 --model transfer
# Rápido en CPU o con la muestra incluida:
python -m src.train --data-dir data/muestra/UTKFace --epochs 5 --model simple
# Opcional: --no-augment | --img-size 128 | --batch-size 32
```

### Probar el pipeline sin dataset (datos sintéticos)
```bash
python -m src.train --demo --epochs 5 --model simple
```

### Evaluar
```bash
python -m src.evaluate --data-dir data/raw/UTKFace --model model/ethnicity_cnn.h5
```

### App Streamlit
```bash
streamlit run app/app.py
```
Sube un `jpg/png` → verás **clase predicha + % confianza + barras por clase**.

## 4. Google Colab (notebook)

1. Sube `notebooks/experimentacion.ipynb` a Colab.
2. `Entorno > Acelerador GPU T4`.
3. Ejecuta de arriba a abajo: instala deps, descarga UTKFace, EDA, train/val/test (70/15/15 estratificado), Data Augmentation, CNN, 25 épocas, métricas, matriz de confusión, 10+ predicciones, guardado `.h5`.
4. Descarga el modelo desde `model/ethnicity_cnn.h5` si quieres usarlo en Streamlit local.

## 5. Modelo y preprocesamiento

- **Entrada:** 128×128×3, `float32 [0,1]` (`/255`). Para MobileNetV2 se reescala a `[-1,1]` internamente.
- **Aumento de datos (solo entrenamiento):** `RandomFlip + Rotation(0.08) + Zoom(0.12) + Contrast(0.12) + Translation(0.08)`.
- **División:** 70/15/15 estratificada por etnia (`sklearn.train_test_split`, semilla 42).
- **Arquitectura principal:** MobileNetV2 ImageNet congelado + `Dropout + Dense(256,relu) + Dropout + Dense(5,softmax)`, `Adam 1e-4`, `sparse_categorical_crossentropy`, `class_weight='balanced'`.
- **Llamadas de retorno:** `ModelCheckpoint(val_accuracy)`, `EarlyStopping(p=7)`, `ReduceLROnPlateau`, `CSVLogger`.
- **Métricas:** exactitud, matriz de confusión (cruda + normalizada), informe por clase (precisión/exhaustividad/F1).
- **Referencia esperada (UTKFace real, 128px, 25 épocas, transfer):** exactitud en prueba ~0.78–0.86; clases minoritarias con F1 menor por desbalance.

## 6. Resultados

Tras entrenar se generan en `results/`:
- `training_history.png` (exactitud/pérdida de entrenamiento frente a validación)
- `confusion_matrix.png` y `confusion_matrix_norm.png`
- `sample_predictions.png` (12 ejemplos de prueba con clase real frente a predicha + confianza)
- `training_log.csv`, `metrics.txt`

## 7. Predicción por código

```python
from PIL import Image
from src.predict import load_model_and_labels, predict_image
model, clases = load_model_and_labels("model/ethnicity_cnn.h5", "model/class_names.json")
etiqueta, prob, vector = predict_image(model, clases, Image.open("foto.jpg"))
print(etiqueta, prob)
```

## 8. Referencias

- Zhang et al. *Age Progression/Regression by Conditional Adversarial Autoencoder*, CVPR 2017. Dataset: https://susanqq.github.io/UTKFace/
- FairFace (alternativa más balanceada): https://github.com/joojs/fairface
- TensorFlow / Keras, scikit-learn, Streamlit.

## 9. Licencia y ética

- Código: MIT (uso educativo). **Las imágenes UTKFace no se redistribuyen** (licencia original: solo investigación no comercial; © de sus dueños).
- No usar para fines discriminatorios. Ver sección de uso responsable en la app.
