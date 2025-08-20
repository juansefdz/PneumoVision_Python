import argparse, os, tensorflow as tf
from config import ARTIFACTS_DIR
from models import compile_model, compile_eff
from utils import show_prediction_with_cam

def load_model_choice(model_name):
    if model_name == "cnn":
        path = os.path.join(ARTIFACTS_DIR, "cnn_best.keras")
        model = tf.keras.models.load_model(path, compile=False)
        compile_model(model, lr=1e-4)
        return model, "CNN"
    elif model_name == "effnet":
        path = os.path.join(ARTIFACTS_DIR, "eff_b0_best.keras")
        model = tf.keras.models.load_model(path, compile=False)
        compile_eff(model, lr=1e-4)
        return model, "EffNetB0"
    else:
        raise ValueError("Modelo inválido")

def parse_args():
    ap = argparse.ArgumentParser(description="Predicción única con Grad-CAM")
    ap.add_argument("--model", choices=["cnn", "effnet"], required=True,
                    help="Modelo a usar")
    ap.add_argument("--img", required=True,
                    help="Ruta a la imagen de rayos X")
    ap.add_argument("--img_size", type=int, default=224,
                    help="Tamaño de entrada (default 224)")
    ap.add_argument("--save_dir", default="resultados",
                    help="Carpeta donde guardar resultados")
    return ap.parse_args()

def main():
    args = parse_args()
    model, name = load_model_choice(args.model)

    show_prediction_with_cam(
        args.img, model, model_name=name,
        threshold=0.5, img_size=args.img_size,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
