import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
import tensorflow as tf

try:
    import tensorflow_addons as tfa
    HAS_ADAMW = True
except ImportError:
    HAS_ADAMW = False

import config
from data_manager import load_datasets
from models import build_resnet_se, build_efficientnet
from utils import get_callbacks, plot_history, evaluate_on_test

# === Estado y Verificación de GPU ===
def setup_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU] CUDA Device Detected: {gpu_name}")
        print("[GPU] Keras running on PyTorch CUDA Backend.")
    else:
        print("[CPU] Running training on CPU...")

setup_gpu()

def get_optimizer(lr, wd):
    if HAS_ADAMW:
        print(f"Usando AdamW (lr={lr}, wd={wd})")
        return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    else:
        print(f"Usando Adam (lr={lr})")
        return tf.keras.optimizers.Adam(learning_rate=lr)

def run_training(model_name="custom"):
    print(f"=== Iniciando entrenamiento para: {model_name} ===")
    
   
    train_ds, val_ds, test_ds, class_weights = load_datasets(model_type=model_name)

  
    if model_name == "custom":
        model = build_resnet_se(config.IMG_SIZE, config.DROPOUT_RATE)
        base_model = None # No hay base pre-entrenada
    elif model_name == "effnet":
        model, base_model = build_efficientnet(config.IMG_SIZE, config.DROPOUT_RATE)
    else:
        raise ValueError("Modelo desconocido. Usa 'custom' o 'effnet'.")
    
    model.summary()

  
    optimizer = get_optimizer(config.LEARNING_RATE, config.WEIGHT_DECAY)
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    
    metrics_list = [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)

   
    best_path = f"{config.ARTIFACTS_DIR}/{model_name}_best.keras"
    cbs = get_callbacks(best_path, patience=10)

 
    print("\n--- Fase de Entrenamiento ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40, # Ajustable
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1
    )
    
 
    if model_name == "effnet" and base_model is not None:
        print("\n--- Fase de Fine-Tuning (Descongelando últimas capas) ---")
        base_model.trainable = True
        
        for layer in base_model.layers[:-50]:
            layer.trainable = False
    
        optimizer_ft = get_optimizer(config.LEARNING_RATE / 10, config.WEIGHT_DECAY)
        model.compile(optimizer=optimizer_ft, loss=loss_fn, metrics=metrics_list)
        
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            class_weight=class_weights,
            callbacks=cbs,
            verbose=1
        )
     
    print("\n--- Evaluando Mejor Modelo en Test ---")
    best_model = tf.keras.models.load_model(best_path, compile=False)
    best_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)
    
    evaluate_on_test(best_model, test_ds, threshold=0.5, name=model_name)
    plot_history(history, title=f"Entrenamiento - {model_name}")

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "effnet"], help="Modelo a entrenar")
    args = parser.parse_args()
    
    run_training(args.model)