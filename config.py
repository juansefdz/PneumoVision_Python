import os

# Reproducibilidad
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# Rutas
# 1) Ruta al dataset original de Kaggle (si lo tienes descargado)
#    Estructura esperada:
#    chest_xray/
#      ├─ train/{NORMAL,PNEUMONIA}
#      ├─ val/{NORMAL,PNEUMONIA}
#      └─ test/{NORMAL,PNEUMONIA}
DATASET_ORIGINAL = os.path.abspath("./chest_xray")

# 2) Ruta de salida para imágenes reescaladas con letterbox
RESIZED_BASE = os.path.abspath("./chest_xray_resized")
TRAIN_DIR_R = os.path.join(RESIZED_BASE, "train")
VAL_DIR_R   = os.path.join(RESIZED_BASE, "val")
TEST_DIR_R  = os.path.join(RESIZED_BASE, "test")

# Directorio para artefactos (modelos, gráficos, etc.)
ARTIFACTS_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
