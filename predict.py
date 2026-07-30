import os
import cv2
import numpy as np
import tensorflow as tf
from utils import prepare_image, encode_heatmap_to_base64
from model_core import AIModelManager

def predict_and_explain(model_path, image_path, model_type="custom"):
    """
    Analiza una radiografía desde línea de comandos e imprime los resultados.
    """
    if not os.path.exists(image_path):
        print(f"Error: La imagen no existe en '{image_path}'")
        return

    if not os.path.exists(model_path):
        print(f"Error: El modelo no existe en '{model_path}'")
        return

    print(f"Cargando imagen desde: {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    img_array = prepare_image(image_bytes)
    
    print(f"Cargando modelo ({model_type}) desde: {model_path}")
    manager = AIModelManager()
    
    # Asignar modelo manualmente
    try:
        if model_type == "custom" and os.path.isdir(model_path):
            manager.models[model_type] = tf.saved_model.load(model_path)
        else:
            manager.models[model_type] = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    res = manager.analyze(model_type, img_array)
    
    print("\n================ RESULTADOS ================")
    print(f"Modelo: {res.get('model_name')}")
    print(f"Predicción: {res.get('prediction')}")
    print(f"Confianza: {res.get('confidence') * 100:.2f}%")
    if res.get('heatmap'):
        print("GradCAM: Mapa de calor generado exitosamente en base64.")
    print("============================================\n")
    return res

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        predict_and_explain(sys.argv[1], sys.argv[2])
    else:
        print("Uso: python predict.py <ruta_modelo> <ruta_imagen> [tipo_modelo]")
