
"""
Script maestro: redimensiona (si hace falta), entrena CNN y EfficientNet,
y evalúa ambos modelos. Puedes comentar pasos según tu necesidad.
"""

import os
from config import DATASET_ORIGINAL, RESIZED_BASE

def maybe_resize():
    # Ejecuta el reescalado sólo si no existe la carpeta destino
    if not os.path.isdir(RESIZED_BASE):
        from scan_and_resize import main as resize_main
        print(">> No existe carpeta reescalada. Ejecutando letterbox...")
        resize_main()
    else:
        print(">> Carpeta reescalada ya existe. Saltando letterbox.")

def train_all():
    import train_cnn, train_effnet
    print("\n>> Entrenamiento CNN")
    train_cnn.main()
    print("\n>> Entrenamiento EfficientNetB0")
    train_effnet.main()

if __name__ == "__main__":
    if not os.path.isdir(DATASET_ORIGINAL):
        raise SystemExit(
            f"No se encuentra el dataset original en {DATASET_ORIGINAL}.\n"
            "Colócalo allí con la estructura chest_xray/{train,val,test}."
        )
    maybe_resize()
    train_all()
    print("\n✅ Pipeline completo finalizado.")
