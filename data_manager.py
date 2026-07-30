import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import config

AUTOTUNE = tf.data.AUTOTUNE

def get_augmenter():
  
    return keras.Sequential([
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="robust_aug")

def count_files_per_class(directory):
    
    counts = {}
    if not os.path.exists(directory):
        return {0: 1, 1: 1} # Fallback
        
   
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(directory, class_name)
     
        num_files = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        counts[i] = num_files
        
    return counts

def load_datasets(model_type="custom"):
  
    common_args = dict(
        label_mode="binary",
        color_mode="grayscale" if model_type == "custom" else "rgb",
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
    )

   
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
        print("No se encontraron suficientes imágenes para calcular pesos. Usando 1:1.")
        class_weight = {0: 1.0, 1: 1.0}
        
    print(f"Pesos: Normal({cnt_normal}): {class_weight[0]:.2f}, Pneumonia({cnt_pneumo}): {class_weight[1]:.2f}")


    normalization = layers.Rescaling(1./255)

    def apply_normalization(x, y):
        if model_type == "effnet":
            return tf.keras.applications.efficientnet.preprocess_input(x), y
        else:
            return normalization(x), y

    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.map(apply_normalization, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    val_ds = val_ds.cache().map(apply_normalization, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.cache().map(apply_normalization, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_weight