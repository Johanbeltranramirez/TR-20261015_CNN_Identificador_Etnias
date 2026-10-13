"""Configuracion central del proyecto CNN Identificador de Etnias."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Clases UTKFace (race 0-4). Se mantienen los nombres originales en ingles
# para compatibilidad con el dataset + etiqueta en espanol para la UI.
CLASS_NAMES = ["White", "Black", "Asian", "Indian", "Others"]
CLASS_NAMES_ES = {
    "White": "Blanca / Caucasica",
    "Black": "Negra / Afrodescendiente",
    "Asian": "Asiatica",
    "Indian": "India / Sudasiatica",
    "Others": "Otras (Latina, Oriente Medio, mixta...)",
}
RACE_TO_LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}

IMG_SIZE = 128          # (IMG_SIZE, IMG_SIZE, 3). MobileNetV2 acepta >= 32.
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

MODEL_PATH = PROJECT_ROOT / "model" / "ethnicity_cnn.h5"
CLASS_NAMES_PATH = PROJECT_ROOT / "model" / "class_names.json"
RESULTS_DIR = PROJECT_ROOT / "results"
