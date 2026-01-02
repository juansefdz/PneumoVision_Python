# data_manager.py
import tensorflow as tf
from tensorflow import keras
from keras import layers
import config

AUTOTUNE = tf.data.AUTOTUNE

def get_augmenter():
    """Genera el pipeline de aumentación de datos para entrenamiento."""
    return keras.Sequential([
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2), # Clave para rayos X con diferente exposición
    ], name="robust_aug")

def load_datasets(model_type="custom"):
    """
    Carga y preprocesa los datasets.
    
    Args:
        model_type (str): 'custom' (escala 1/255) o 'effnet' (preprocess nativo).
    """
    # 1. Cargar desde directorios
    common_args = dict(
        label_mode="binary",
        color_mode="grayscale" if model_type == "custom" else "rgb",
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=False # Se hace shuffle manual en train
    )

    train_ds = keras.preprocessing.image_dataset_from_directory(
        config.TRAIN_DIR, shuffle=True, seed=config.SEED, **common_args
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        config.VAL_DIR, **common_args
    )
    test_ds = keras.preprocessing.image_dataset_from_directory(
        config.TEST_DIR, **common_args
    )

    # 2. Calcular pesos de clase (solo con train)
    print("Calculando pesos de clase...")
    counts = {0: 0, 1: 0}
    for _, y in train_ds.unbatch():
        counts[int(y.numpy())] += 1
    
    total = counts[0] + counts[1]
    class_weight = {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1]),
    }
    print(f"Pesos calculados: {class_weight}")

    # 3. Definir función de preprocesamiento
    augmenter = get_augmenter()
    normalization = layers.Rescaling(1./255)

    def _preprocess(x, y, training=False):
        # Aumentación solo en training
        if training:
            x = augmenter(x, training=True)
        
        # Normalización dependiente del modelo
        if model_type == "effnet":
            # EfficientNet espera 0-255 inputs si usamos los pesos por defecto de TF,
            # pero su preprocess_input interno se encarga. 
            # Si usamos 'imagenet', usamos la funcion de keras applications.
            x = tf.keras.applications.efficientnet.preprocess_input(x)
        else:
            # Para nuestro modelo custom, normalizamos a [0, 1]
            x = normalization(x)
            
        return x, y

    # 4. Optimización (Map -> Cache -> Shuffle -> Prefetch)
    # Train
    train_ds = train_ds.map(lambda x, y: _preprocess(x, y, training=True), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    
    # Val / Test
    val_ds = val_ds.map(lambda x, y: _preprocess(x, y, training=False), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    test_ds = test_ds.map(lambda x, y: _preprocess(x, y, training=False), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_weight