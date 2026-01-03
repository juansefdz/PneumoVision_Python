# data_manager.py
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import config

AUTOTUNE = tf.data.AUTOTUNE

def get_augmenter():
    """Genera el pipeline de aumentación de datos para entrenamiento."""
    # Nota: Estos layers pueden correr en GPU si se integran al modelo,
    # pero aquí usamos map() optimizado para preparar batch en paralelo.
    return keras.Sequential([
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="robust_aug")

def count_files_per_class(directory):
    """Cuenta archivos rápidamente sin iterar el dataset de TensorFlow."""
    counts = {}
    if not os.path.exists(directory):
        return {0: 1, 1: 1} # Fallback
        
    # Keras ordena clases alfabéticamente: NORMAL (0), PNEUMONIA (1)
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(directory, class_name)
        # Contar solo archivos de imagen válidos
        num_files = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        counts[i] = num_files
        
    return counts

def load_datasets(model_type="custom"):
    """
    Carga y preprocesa los datasets con pipeline optimizado.
    
    Args:
        model_type (str): 'custom' (escala 1/255) o 'effnet' (preprocess nativo).
    """
    # 1. Configuración común
    common_args = dict(
        label_mode="binary",
        color_mode="grayscale" if model_type == "custom" else "rgb",
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
    )

    # 2. Cargar datasets (Lectura de disco)
    print("Cargando datasets...")
    train_ds = keras.preprocessing.image_dataset_from_directory(
        config.TRAIN_DIR, shuffle=True, seed=config.SEED, **common_args
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        config.VAL_DIR, shuffle=False, **common_args
    )
    test_ds = keras.preprocessing.image_dataset_from_directory(
        config.TEST_DIR, shuffle=False, **common_args
    )

    # 3. Cálculo de pesos de clase (Optimizado: <1 segundo)
    print("Calculando pesos de clase (método rápido)...")
    counts = count_files_per_class(config.TRAIN_DIR)
    
    # Asumiendo 0: NORMAL, 1: PNEUMONIA
    cnt_normal = counts.get(0, 0)
    cnt_pneumo = counts.get(1, 0)
    total = cnt_normal + cnt_pneumo
    
    if total > 0 and cnt_normal > 0 and cnt_pneumo > 0:
        class_weight = {
            0: total / (2.0 * cnt_normal),
            1: total / (2.0 * cnt_pneumo),
        }
    else:
        print("⚠️ Advertencia: No se encontraron suficientes imágenes para calcular pesos. Usando 1:1.")
        class_weight = {0: 1.0, 1: 1.0}
        
    print(f"Pesos: Normal({cnt_normal}): {class_weight[0]:.2f}, Pneumonia({cnt_pneumo}): {class_weight[1]:.2f}")

    # 4. Definir operaciones
    augmenter = get_augmenter()
    normalization = layers.Rescaling(1./255)

    def apply_augmentation(x, y):
        # Aplica aumentación. training=True es vital para que las capas Random* funcionen
        return augmenter(x, training=True), y

    def apply_normalization(x, y):
        if model_type == "effnet":
            return tf.keras.applications.efficientnet.preprocess_input(x), y
        else:
            return normalization(x), y

    # 5. Construcción del Pipeline de Alto Rendimiento
    # ORDEN CRÍTICO: Cache -> Shuffle -> Augment -> Normalize -> Prefetch
    
    # --- TRAIN ---
    # a) Cachear imágenes crudas en RAM (o disco) para evitar lectura repetitiva
    train_ds = train_ds.cache()
    # b) Mezclar el buffer cacheado
    train_ds = train_ds.shuffle(1000)
    # c) Aumentación (debe ir DESPUÉS del cache para que sea dinámica cada época)
    train_ds = train_ds.map(apply_augmentation, num_parallel_calls=AUTOTUNE)
    # d) Normalización
    train_ds = train_ds.map(apply_normalization, num_parallel_calls=AUTOTUNE)
    # e) Prefetch para solapar CPU/GPU
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    # --- VAL / TEST ---
    # Solo normalización, sin aumentación
    val_ds = val_ds.cache().map(apply_normalization, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.cache().map(apply_normalization, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_weight