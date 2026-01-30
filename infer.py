

import os
import argparse
import tensorflow as tf
from models import compile_model, compile_eff
from utils import show_prediction_with_cam, load_xray_for_model, predict_xray

def parse_args():
    ap = argparse.ArgumentParser(description="Inferencia con modelos PneumoVision")
    ap.add_argument("--model", required=True, help="Ruta al modelo .keras (CNN o EfficientNet)")
    ap.add_argument("--imgs",  nargs="+", required=True, help="Ruta(s) a imagen(es) a inferir")
    ap.add_argument("--img_size", type=int, default=224, help="Tamaño de entrada (default: 224)")
    ap.add_argument("--thr", type=float, default=0.5, help="Umbral de decisión para PNEUMONIA (default: 0.5)")
    ap.add_argument("--cam", action="store_true", help="Generar Grad-CAM y overlays")
    ap.add_argument("--outdir", default="infer_outputs", help="Carpeta para guardar outputs CAM")
    return ap.parse_args()

def is_effnet(model):
 
    try:
        model.get_layer("efficientnetb0")
        return True
    except Exception:
        return False

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model = tf.keras.models.load_model(args.model, compile=False)

    if is_effnet(model):
        compile_eff(model, lr=1e-4)
        model_name = "EfficientNetB0"
        base = model.get_layer("efficientnetb0")
        last_conv = "top_conv"
    else:
        compile_model(model, lr=1e-4)
        model_name = "CNN"
        base = None
        last_conv = None

    print(f"Modelo cargado: {model_name} — {args.model}")


    for img_path in args.imgs:
        if not os.path.isfile(img_path):
            print(f"No existe: {img_path}")
            continue

        try:
            label, prob, im_vis, arr = predict_xray(img_path, model, threshold=args.thr, img_size=args.img_size)
            print(f"[{model_name}] {os.path.basename(img_path)} → {label}  (Prob PNEUMONIA={prob*100:.2f}%, thr={args.thr})")

            if args.cam:
               
                img_stem = os.path.splitext(os.path.basename(img_path))[0]
                save_dir = os.path.join(args.outdir, f"{img_stem}_{model_name}")
                show_prediction_with_cam(
                    img_path, model, model_name=model_name,
                    threshold=args.thr, alpha=0.45,
                    last_conv_name=last_conv, owner_model=base,
                    target="auto", img_size=args.img_size, save_dir=save_dir
                )

        except Exception as e:
            print(f"Error procesando {img_path}: {e}")

if __name__ == "__main__":
    main()
