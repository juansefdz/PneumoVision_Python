import os
import io
import uuid
from typing import Optional, Literal

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import keras

# Importa de tu proyecto
from config import ARTIFACTS_DIR, IMG_SIZE
from data_pipeline import val_ds_prep, val_ds_rgb
from models import compile_model, compile_eff
from utils import (
    resize_crop_pil,
    predict_xray,
    show_prediction_with_cam,
    get_probs_from_ds,
    threshold_max_f1,
    threshold_for_recall,
)

# ---------- Config ----------
RESULTS_DIR = os.path.join(os.getcwd(), "resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="PneumoVision API", version="1.0")

# CORS (ajusta origins si tienes front aparte)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=RESULTS_DIR), name="static")

# ---------- Modelos en memoria ----------
CNN_PATH = os.path.join(ARTIFACTS_DIR, "cnn_best.keras")
EFF_PATH = os.path.join(ARTIFACTS_DIR, "eff_b0_best.keras")

CNN_MODEL = None
EFF_MODEL = None

# Cache de umbrales
THRESHOLDS = {
    "cnn": {"f1": None, "recall": None},      
    "effnet": {"f1": None, "recall": None},
}
DEFAULT_TARGET_RECALL = 0.90


def _load_models_once():
    """Carga los modelos en memoria si existen y aún no están cargados."""
    global CNN_MODEL, EFF_MODEL

    try:
        if os.path.exists(CNN_PATH) and CNN_MODEL is None:
            model = keras.models.load_model(CNN_PATH, compile=False)
            compile_model(model, lr=1e-4)
            CNN_MODEL = model
            print("[INFO] Modelo CNN cargado exitosamente.")

        if os.path.exists(EFF_PATH) and EFF_MODEL is None:
            model = keras.models.load_model(EFF_PATH, compile=False)
            compile_eff(model, lr=1e-4)
            EFF_MODEL = model
            print("[INFO] Modelo EfficientNet cargado exitosamente.")
    except Exception as e:
        print(f"[ERROR] No se pudieron cargar los modelos. Causa: {e}")
        CNN_MODEL = None
        EFF_MODEL = None


def _compute_thresholds_once():
    """Calcula y cachea los umbrales de clasificación."""
    
    if CNN_MODEL is not None:
        y, p = get_probs_from_ds(CNN_MODEL, val_ds_prep)
        thr_f1, *_ = threshold_max_f1(y, p)
        THRESHOLDS["cnn"]["f1"] = float(thr_f1)
        thr_rec, *_ = threshold_for_recall(y, p, target_recall=DEFAULT_TARGET_RECALL)
        THRESHOLDS["cnn"]["recall"] = float(thr_rec)

    if EFF_MODEL is not None:
        y, p = get_probs_from_ds(EFF_MODEL, val_ds_rgb)
        thr_f1, *_ = threshold_max_f1(y, p)
        THRESHOLDS["effnet"]["f1"] = float(thr_f1)
        thr_rec, *_ = threshold_for_recall(y, p, target_recall=DEFAULT_TARGET_RECALL)
        THRESHOLDS["effnet"]["recall"] = float(thr_rec)


@app.on_event("startup")
def startup_event():
    _load_models_once()
    if CNN_MODEL is None and EFF_MODEL is None:
        # Sin modelos, la API seguirá levantando, pero /predict fallará
        print("[WARN] No se encontraron modelos en ./artifacts. La API se iniciará, pero las predicciones fallarán.")
    else:
        _compute_thresholds_once()
        print("[INFO] Umbrales iniciales:", THRESHOLDS)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "cnn_loaded": CNN_MODEL is not None,
            "effnet_loaded": EFF_MODEL is not None,
        },
        "thresholds": THRESHOLDS,
    }


# ---------- Utils locales ----------
def _save_bytes_to_tmp(upload: UploadFile) -> str:

    base = os.path.splitext(upload.filename or "upload")[0]
    uid = uuid.uuid4().hex[:8]
    fname = f"{base}_{uid}.png"
    fpath = os.path.join(RESULTS_DIR, fname)

    data = upload.file.read()
    Image.open(io.BytesIO(data)).save(fpath)  
    return fpath


def _choose_model(which: str):
    which = (which or "effnet").lower()
    if which == "cnn":
        if CNN_MODEL is None:
            raise HTTPException(400, "Modelo CNN no cargado. Revisa los logs de inicio para más detalles.")
        return CNN_MODEL, "CNN"
    elif which in ("effnet", "efficientnet", "b0", "eff"):
        if EFF_MODEL is None:
            raise HTTPException(400, "Modelo EfficientNet no cargado. Revisa los logs de inicio para más detalles.")
        return EFF_MODEL, "EffNetB0"
    else:
        raise HTTPException(400, "Parámetro 'model' debe ser 'cnn' o 'effnet'.")


def _choose_threshold(model_name: str, mode: str, target_recall: float) -> float:
    
    mode = (mode or "f1").lower()
    if mode == "fixed":
        return 0.5
    elif mode == "f1":
        thr = THRESHOLDS[model_name.lower()].get("f1")
        if thr is None:
            raise HTTPException(500, "Umbral F1 no calculado. Revisa los logs de inicio.")
        return float(thr)
    elif mode == "recall":
        thr = THRESHOLDS[model_name.lower()].get("recall")
        if thr is None:
            raise HTTPException(500, "Umbral recall no calculado. Revisa los logs de inicio.")
        return float(thr)
    else:
        raise HTTPException(400, "mode debe ser 'fixed', 'f1' o 'recall'.")


# ---------- Endpoint principal ----------
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    model: Literal["cnn", "effnet"] = Form("effnet"),
    mode: Literal["fixed", "f1", "recall"] = Form("f1"),
    target_recall: Optional[float] = Form(None),  
    alpha: float = Form(0.45),                 
):
    
    # Seguridad mínima de entrada
    if not file.filename:
        raise HTTPException(400, "Debes adjuntar una imagen.")
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        raise HTTPException(400, "Formato de imagen no soportado.")

    # Elegir modelo
    mdl, mdl_name = _choose_model(model)

    # Elegir umbral
    thr = _choose_threshold(model, mode, target_recall or DEFAULT_TARGET_RECALL)

    # Guardar temporalmente
    try:
        tmp_path = _save_bytes_to_tmp(file)
    except Exception as e:
        raise HTTPException(400, f"Imagen inválida: {e}")

    # Generar overlay + JSON usando util de tu proyecto
    try:
        show_prediction_with_cam(
            path=tmp_path,
            model=mdl,
            model_name=mdl_name,
            threshold=thr,
            alpha=alpha,
            last_conv_name=None,          
            owner_model=None,
            target="auto",
            img_size=IMG_SIZE,
            save_dir=RESULTS_DIR
        )
    except Exception as e:
        raise HTTPException(500, f"Fallo generando CAM: {e}")

    
    base = os.path.splitext(os.path.basename(tmp_path))[0]
    overlay_name = f"{base}_overlay.png"
    json_name = f"{base}_result.json"

    # Respuesta
    return {
        "ok": True,
        "model": mdl_name,
        "mode": mode,
        "threshold": thr,
        # URLs (servidas en /static)
        "overlay_url": f"/static/{overlay_name}",
        "result_json_url": f"/static/{json_name}",
        # Nota: el JSON ya incluye diagnosis y probabilidad en %.
        "note": "Esto es una ayuda de software, no un diagnóstico médico."
    }