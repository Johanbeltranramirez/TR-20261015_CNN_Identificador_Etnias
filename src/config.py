"""Configuracion central del proyecto CNN Identificador de Etnias."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Clases UTKFace (race 0-4). Los nombres originales en ingles se conservan
# solo como clave tecnica (compatibilidad con el dataset); toda la
# interfaz de usuario muestra las etiquetas en espanol.
CLASS_NAMES = ["White", "Black", "Asian", "Indian", "Others"]
CLASS_NAMES_ES = {
    "White": "Blanca / Caucásica",
    "Black": "Negra / Afrodescendiente",
    "Asian": "Asiática",
    "Indian": "India / Sudasiática",
    "Others": "Otras (latina, Oriente Medio, mixta…)",
}
CLASES_ES = [CLASS_NAMES_ES[c] for c in CLASS_NAMES]
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
