# CNN Identificador de Etnias Culturales 🧬

Clasificación de imágenes de rostros en **5 grupos étnicos del dataset UTKFace** (White, Black, Asian, Indian, Others) con una **Red Neuronal Convolucional (CNN)** + Transfer Learning (MobileNetV2).

> ⚠️ **Uso responsable:** la etnia no se reduce a rasgos físicos. Este proyecto es **educativo**, refleja los sesgos del dataset (mayoría *White*) y **no debe usarse** para discriminar, vigilar ni tomar decisiones sobre personas.

## 1. Contenido del repositorio

```
├── app/app.py                  # Aplicación Streamlit (carga modelo .h5 y predice + probabilidad)
├── src/
│   ├── config.py               # Clases, IMG_SIZE=128, rutas
│   ├── dataset.py              # Parseo UTKFace, split estratificado, tf.data + normalización
│   ├── model.py                # CNN simple + MobileNetV2 transfer
│   ├── train.py                # Entrenamiento, gráficas, matriz de confusión, 12 predicciones
│   ├── evaluate.py             # Evaluación en test
│   └── predict.py              # Inferencia para una imagen
├── notebooks/
│   └── experimentacion.ipynb   # Notebook de experimentación (Google Colab)
├── model/
│   ├── ethnicity_cnn.h5        # Modelo entrenado (incluido: pesos demo smoke-test 64px;
│   │                           #  re-entrena en Colab con UTKFace real para pesos de producción 128px)
│   └── class_names.json        # ["White","Black","Asian","Indian","Others"]
├── results/                    # training_history.png, confusion_matrix*.png, sample_predictions.png
├── data/README.md              # Cómo descargar UTKFace (no incluido por licencia)
├── requirements.txt
└── README.md
```

## 2. Dataset: UTKFace

- **Fuente:** [UTKFace — Large Scale Face Dataset](https://susanqq.github.io/UTKFace/) (Zhang et al., CVPR 2017). Solo investigación no comercial.
- **Versión usada:** *Aligned & Cropped Faces* (~107 MB, ~20k+ rostros, un rostro por imagen, variada pose/iluminación/edad 0–116).
- **Etiquetas en el nombre:** `[age]_[gender]_[race]_[datetime].jpg`
  - `age`: 0–116 · `gender`: 0=hombre, 1=mujer · `race`: `0=White, 1=Black, 2=Asian, 3=Indian, 4=Others (Hispanic/Latino/Middle Eastern)`.
- **Por qué sirve:** es el benchmark estándar para etnia/edad/género, ya recortado a rostro, con desbalance real (mayoría White) que obliga a usar `class_weight` y aumento de datos.
- **Distribución aprox.:** White ~40%, Black ~19%, Asian ~15%, Indian ~15%, Others ~11% (ver EDA en el notebook).

### Descarga (elegir una)

Opción A — Google Drive oficial (requiere `gdown`):
```bash
pip install gdown
# Carpeta oficial Aligned&Cropped: https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE
gdown --folder https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE -O data/raw/UTKFace
```

Opción B — Kaggle (espejo `jangedoo/utkface-new`):
```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('jangedoo/utkface-new'))"
# Copia los .jpg a data/raw/UTKFace/
```

La carpeta final debe contener `*.jpg` como `25_0_2_20170116174525157.jpg`.

## 3. Instalación y ejecución local

```bash
git clone <este-repo> && cd TR-20261015_CNN_Identificador_Etnias
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

### Entrenar (con UTKFace real)
```bash
python -m src.train --data-dir data/raw/UTKFace --epochs 25 --model transfer
# Opcional: --model simple | --no-augment | --img-size 128 | --batch-size 32
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
- **Augmentation (solo train):** `RandomFlip + Rotation(0.08) + Zoom(0.12) + Contrast(0.12) + Translation(0.08)`.
- **Split:** 70/15/15 estratificado por `race` (`sklearn.train_test_split`, seed 42).
- **Arquitectura principal:** MobileNetV2 ImageNet congelado + `Dropout + Dense(256,relu) + Dropout + Dense(5,softmax)`, `Adam 1e-4`, `sparse_categorical_crossentropy`, `class_weight='balanced'`.
- **Callbacks:** `ModelCheckpoint(val_accuracy)`, `EarlyStopping(p=7)`, `ReduceLROnPlateau`, `CSVLogger`.
- **Métricas:** accuracy, matriz de confusión (cruda + normalizada), `classification_report` (precision/recall/F1 por clase).
- **Baseline esperado (UTKFace real, 128px, 25 épocas):** accuracy test ~0.78–0.86; clases minoritarias con F1 menor por desbalance.

## 6. Resultados

Tras entrenar se generan en `results/`:
- `training_history.png` (accuracy/loss train vs val)
- `confusion_matrix.png` y `confusion_matrix_norm.png`
- `sample_predictions.png` (12 ejemplos test con real vs predicho + confianza)
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
