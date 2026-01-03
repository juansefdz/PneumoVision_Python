import tensorflow as tf
print("Versión de TF:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ ¡GPU DETECTADA! ({len(gpus)} encontradas)")
    for gpu in gpus:
        print("  -", gpu)
else:
    print("⚠️  NO se detectó GPU. Se usará la CPU.")