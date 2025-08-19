"""
Carga los modelos ya entrenados desde ./artifacts y los evalúa en el test set.
También puedes probar Grad-CAM en imágenes específicas.
"""

import tensorflow as tf
from data_pipeline import test_ds_prep, test_ds_rgb
from models import compile_model, compile_eff
from utils import evaluate_on_test, show_prediction_with_cam
from config import ARTIFACTS_DIR

def main():
    # === CNN ===
    print("\n== Evaluando CNN ==")
    cnn_path = f"{ARTIFACTS_DIR}/cnn_best.keras"
    cnn_model = tf.keras.models.load_model(cnn_path, compile=False)
    compile_model(cnn_model, lr=1e-4)
    cnn_model.evaluate(test_ds_prep, verbose=1)
    evaluate_on_test(cnn_model, test_ds_prep, threshold=0.5, name="CNN")

    # === EfficientNetB0 ===
    print("\n== Evaluando EfficientNetB0 ==")
    eff_path = f"{ARTIFACTS_DIR}/eff_b0_best.keras"
    eff_model = tf.keras.models.load_model(eff_path, compile=False)
    compile_eff(eff_model, lr=1e-4)
    eff_model.evaluate(test_ds_rgb, verbose=1)
    evaluate_on_test(eff_model, test_ds_rgb, threshold=0.5, name="EffNetB0")

    # === Ejemplo Grad-CAM ===
    # Cambia esta ruta a una radiografía de tu test set
    img_path = "./chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"
    try:
        show_prediction_with_cam(img_path, eff_model, model_name="EffNetB0", threshold=0.5)
    except Exception as e:
        print("No se pudo generar Grad-CAM:", e)

if __name__ == "__main__":
    main()