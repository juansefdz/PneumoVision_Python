
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers

from config import (
    IMG_SIZE, BATCH_SIZE, SEED,
    TRAIN_DIR_R, VAL_DIR_R, TEST_DIR_R
)

# --- Carga datasets (val oficial) ---
common = dict(
    label_mode="binary",
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_names=["NORMAL", "PNEUMONIA"],
)

train_ds_raw = keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR_R, shuffle=True, seed=SEED, **common
)
val_ds_raw = keras.preprocessing.image_dataset_from_directory(
    VAL_DIR_R, shuffle=False, **common
)
test_ds_raw = keras.preprocessing.image_dataset_from_directory(
    TEST_DIR_R, shuffle=False, **common
)

# --- Preprocess (sin flip H) ---
AUTOTUNE = tf.data.AUTOTUNE
normalization = layers.Rescaling(1./255)

aug = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
    layers.RandomTranslation(0.02, 0.02),
], name="xray_aug")

def preprocess(ds, training=False, as_rgb=False, cache_path=None):
    if not as_rgb:
        ds = ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache(cache_path) if cache_path else ds.cache()
    if training:
        ds = ds.shuffle(1000, seed=SEED)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

# Salidas de pipeline
train_ds_prep = preprocess(train_ds_raw, training=True,  as_rgb=False)
val_ds_prep   = preprocess(val_ds_raw,   training=False, as_rgb=False)
test_ds_prep  = preprocess(test_ds_raw,  training=False, as_rgb=False)

train_ds_rgb = preprocess(train_ds_raw, training=True,  as_rgb=True)
val_ds_rgb   = preprocess(val_ds_raw,   training=False, as_rgb=True)
test_ds_rgb  = preprocess(test_ds_raw,  training=False, as_rgb=True)

# Pesos de clase desde el dataset real
def count_from_ds(ds):
    counts = {0: 0, 1: 0}
    for _, y in ds.unbatch():
        counts[int(y.numpy())] += 1
    return counts

counts = count_from_ds(train_ds_raw)
total = counts[0] + counts[1]
class_weight = {
    0: total / (2 * counts[0]),
    1: total / (2 * counts[1]),
}

print("Class counts:", counts)
print("Class weights:", class_weight)
