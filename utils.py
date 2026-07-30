import os
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf

def resize_letterbox(img_np, target_size=(224, 224), color=(0, 0, 0)):
    h, w = img_np.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    return canvas

def prepare_image(image_bytes, target_size=(224, 224)):
    # Convertir bytes a objeto de imagen PIL
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)
    
    # Redimensionar con letterbox para mantener relación de aspecto
    letterbox_np = resize_letterbox(img_np, target_size=target_size)
    
    # Convertir a array de Numpy con batch dimension
    img_array = np.expand_dims(letterbox_np.astype(np.float32), axis=0)
    
    return img_array

def encode_heatmap_to_base64(heatmap_array, original_img_size=(224, 224)):
    heatmap = np.uint8(255 * heatmap_array)
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet_heatmap = cv2.resize(jet_heatmap, original_img_size)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(jet_heatmap)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return "data:image/png;base64," + new_image_string

def get_callbacks(best_path, patience=10):
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            best_path, 
            monitor="val_auc", 
            mode="max", 
            save_best_only=True, 
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", 
            mode="max", 
            patience=patience, 
            restore_best_weights=True, 
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", 
            mode="max", 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6, 
            verbose=1
        )
    ]

def evaluate_on_test(model, test_ds, threshold=0.5, name="model"):
    print(f"\n--- Evaluando {name} (Umbral: {threshold}) ---")
    results = model.evaluate(test_ds, verbose=1)
    metrics_names = model.metrics_names
    for m, val in zip(metrics_names, results):
        print(f" - {m}: {val:.4f}")
    return results

def plot_history(history, title="Historial de Entrenamiento"):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        if "loss" in history.history:
            axes[0].plot(history.history["loss"], label="Train Loss")
        if "val_loss" in history.history:
            axes[0].plot(history.history["val_loss"], label="Val Loss")
        axes[0].set_title(f"{title} - Loss")
        axes[0].legend()
        
        # AUC / Accuracy
        metric_key = "auc" if "auc" in history.history else "accuracy"
        if metric_key in history.history:
            axes[1].plot(history.history[metric_key], label=f"Train {metric_key.upper()}")
        if f"val_{metric_key}" in history.history:
            axes[1].plot(history.history[f"val_{metric_key}"], label=f"Val {metric_key.upper()}")
        axes[1].set_title(f"{title} - {metric_key.upper()}")
        axes[1].legend()
        
        out_path = f"artifacts/{title.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f" Gráfica de entrenamiento guardada en: {out_path}")
    except Exception as e:
        print(f"No se pudo generar la gráfica de entrenamiento: {e}")