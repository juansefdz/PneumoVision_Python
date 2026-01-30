import os


os.environ["TF_USE_LEGACY_KERAS"] = "0" 

import keras
import tensorflow as tf
import shutil


base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "artifacts", "custom_best.keras")
output_path = os.path.join(base_dir, "artifacts", "custom_graph")

print(f"Exportando {input_path} a Grafo Universal...")


if os.path.exists(output_path):
    shutil.rmtree(output_path)

try:
 
    model = keras.models.load_model(input_path)
 
    model.export(output_path)
    
    print(f" Modelo exportado a la carpeta: {output_path}")

except Exception as e:
    print(f"Error: {e}")