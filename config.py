import os

# === Hiperparámetros Generales ===
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
SPLIT = 0.15

# === Rutas de Datos ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(BASE_DIR, "chest_xray_resized")

TRAIN_DIR = os.path.join(DATASET_BASE, "train")
VAL_DIR = os.path.join(DATASET_BASE, "val")
TEST_DIR = os.path.join(DATASET_BASE, "test")

# === Rutas de Artefactos ===
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# === Parámetros de Modelo ===
NUM_CLASSES = 1  # Binario
DROPOUT_RATE = 0.4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4