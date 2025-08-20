"""
Carga los modelos ya entrenados desde ./artifacts y los evalúa en TEST.
También calcula umbral óptimo en VALIDACIÓN (máx F1 o recall objetivo),
y permite generar Grad-CAM para imágenes sueltas o una carpeta completa.
"""

import argparse
import os
import glob
import tensorflow as tf

from config import ARTIFACTS_DIR
from data_pipeline import (
    val_ds_prep, test_ds_prep,     # CNN (grayscale)
    val_ds_rgb,  test_ds_rgb       # EfficientNet (RGB)
)
from models import compile_model, compile_eff
from utils import (
    evaluate_on_test,
    evaluate_with_val_threshold,
    get_probs_from_ds,
    plot_pr_curve,
    show_prediction_with_cam,
)

# --- Recolector de imágenes para CAM ---
def collect_images(cam_imgs, cam_dir):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.gif","*.tif","*.tiff")
    paths = set(cam_imgs or [])
    if cam_dir:
        for ext in exts:
            for p in glob.glob(os.path.join(cam_dir, "**", ext), recursive=True):
                paths.add(p)
    paths = [p for p in sorted(paths) if os.path.isfile(p)]
    if not paths:
        print("No se encontraron imágenes para CAM.")
    return paths

# --- Evaluación CNN ---
def eval_cnn(mode, target_recall, cam_list, img_size, save_dir=None):
    cnn_path = os.path.join(ARTIFACTS_DIR, "cnn_best.keras")
    if not os.path.exists(cnn_path):
        print(f"No existe {cnn_path}. Entrena primero la CNN.")
        return

    model = tf.keras.models.load_model(cnn_path, compile=False)
    compile_model(model, lr=1e-4)
    print("\n== Evaluando CNN ==")

    # 1) Elegir / calcular umbral
    thr = 0.5
    if mode == "fixed":
        evaluate_on_test(model, test_ds_prep, threshold=thr, name="CNN (thr=0.5)")
    elif mode == "f1":
        thr, *_ = evaluate_with_val_threshold(model, val_ds_prep, test_ds_prep, mode="f1", name="CNN")
    elif mode == "recall":
        thr, *_ = evaluate_with_val_threshold(
            model, val_ds_prep, test_ds_prep, mode="recall",
            target_recall=target_recall, name="CNN"
        )

    # 2) PR-curve validación (opcional)
    yv, pv = get_probs_from_ds(model, val_ds_prep)
    plot_pr_curve(yv, pv, title="CNN – PR (validación)")

    # 3) Grad-CAM con el umbral elegido
    if cam_list:
        for p in cam_list:
            try:
                show_prediction_with_cam(
                    p, model, model_name="CNN",
                    threshold=thr, img_size=img_size,
                    last_conv_name=None, owner_model=None,
                    save_dir=save_dir or "resultados"
                )
            except Exception as e:
                print(f"Grad-CAM CNN falló para {p}: {e}")

# --- Evaluación EfficientNetB0 ---
def eval_effnet(mode, target_recall, cam_list, img_size, save_dir=None):
    eff_path = os.path.join(ARTIFACTS_DIR, "eff_b0_best.keras")
    if not os.path.exists(eff_path):
        print(f"No existe {eff_path}. Entrena primero EfficientNet.")
        return

    model = tf.keras.models.load_model(eff_path, compile=False)
    compile_eff(model, lr=1e-4)
    print("\n== Evaluando EfficientNetB0 ==")

    # 1) Elegir / calcular umbral
    thr = 0.5
    if mode == "fixed":
        evaluate_on_test(model, test_ds_rgb, threshold=thr, name="EffNetB0 (thr=0.5)")
    elif mode == "f1":
        thr, *_ = evaluate_with_val_threshold(model, val_ds_rgb, test_ds_rgb, mode="f1", name="EffNetB0")
    elif mode == "recall":
        thr, *_ = evaluate_with_val_threshold(
            model, val_ds_rgb, test_ds_rgb, mode="recall",
            target_recall=target_recall, name="EffNetB0"
        )

    # 2) PR-curve validación (opcional)
    yv, pv = get_probs_from_ds(model, val_ds_rgb)
    plot_pr_curve(yv, pv, title="EffNetB0 – PR (validación)")

    # 3) Grad-CAM con el umbral elegido
    if cam_list:
        base = None
        try:
            base = model.get_layer("efficientnetb0")
        except Exception:
            pass
        for p in cam_list:
            try:
                show_prediction_with_cam(
                    p, model, model_name="EffNetB0",
                    threshold=thr, img_size=img_size,
                    last_conv_name="top_conv", owner_model=base,
                    save_dir=save_dir or "resultados"
                )
            except Exception as e:
                print(f"Grad-CAM EffNetB0 falló para {p}: {e}")

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluación de modelos PneumoVision")
    ap.add_argument("--model", choices=["cnn", "effnet", "both"], required=True,
                    help="Modelo a evaluar")
    ap.add_argument("--mode", choices=["fixed", "f1", "recall"], default="fixed",
                    help="Umbral: fixed=0.5, f1=máx F1, recall=objetivo")
    ap.add_argument("--target_recall", type=float, default=0.90,
                    help="Recall objetivo si --mode recall (default 0.90)")
    ap.add_argument("--cam_imgs", nargs="*", default=None,
                    help="Rutas a imágenes para Grad-CAM")
    ap.add_argument("--cam_dir", default=None,
                    help="Carpeta con imágenes para Grad-CAM (recursivo)")
    ap.add_argument("--img_size", type=int, default=224,
                    help="Tamaño de entrada para inferencia/CAM")
    ap.add_argument("--save_cam", default=None,
                    help="Carpeta donde guardar overlays/heatmaps (opcional)")
    return ap.parse_args()

def main():
    args = parse_args()

    # preparar lista de imágenes y carpeta de salida
    cam_list = collect_images(args.cam_imgs, args.cam_dir)
    out_dir = args.save_cam or "resultados"
    os.makedirs(out_dir, exist_ok=True)

    if args.model in ("cnn", "both"):
        eval_cnn(args.mode, args.target_recall, cam_list, args.img_size, save_dir=out_dir)

    if args.model in ("effnet", "both"):
        eval_effnet(args.mode, args.target_recall, cam_list, args.img_size, save_dir=out_dir)

if __name__ == "__main__":
    main()
