import os


os.environ["TF_USE_LEGACY_KERAS"] = "0"

import keras
import tensorflow as tf

print("IA TRAINER: Iniciando conversión de compatibilidad...")


base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "artifacts", "custom_best.keras")
output_path = os.path.join(base_dir, "artifacts", "custom_legacy.h5") 

try:
    print(f"Leyendo modelo moderno: {input_path}")
  
    model = keras.models.load_model(input_path)
    
    print("Guardando en formato universal (.h5)...")
    
    model.save(output_path, save_format='h5')
    
    print(f"Archivo generado: {output_path}")

except Exception as e:
    print(f"Error: {e}")