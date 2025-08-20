import os

# Reproducibilidad
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
SPLIT = 0.15

DATASET_ORIGINAL = os.path.abspath("./chest_xray")

# 2) Ruta de salida para imágenes reescaladas con letterbox
RESIZED_BASE = os.path.abspath("./chest_xray_resized")
TRAIN_DIR_R = os.path.join(RESIZED_BASE, "train")
VAL_DIR_R   = os.path.join(RESIZED_BASE, "val")
TEST_DIR_R  = os.path.join(RESIZED_BASE, "test")

# Directorio para artefactos (modelos, gráficos, etc.)
ARTIFACTS_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
