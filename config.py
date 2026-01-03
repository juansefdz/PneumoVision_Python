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

# --- CAMBIO CRITICO PARA PLAN B ---
# La carpeta "val" original solo tiene 16 fotos, lo que causa métricas locas.
# Usamos "test" (624 fotos) para validar mientras entrenamos.
VAL_DIR = os.path.join(DATASET_BASE, "test") 
TEST_DIR = os.path.join(DATASET_BASE, "test")

# === Rutas de Artefactos ===
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# === Parámetros de Modelo ===
NUM_CLASSES = 1  # Binario
DROPOUT_RATE = 0.4

# --- CAMBIO CRITICO PARA PLAN B ---
# Bajamos el LR de 1e-3 a 1e-4 para un aprendizaje más suave y estable
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 1e-4