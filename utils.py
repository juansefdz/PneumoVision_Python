import os, json, datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from PIL import Image


# Callbacks unificados (PR-AUC) 
def get_callbacks(best_path, patience=8):
    es = callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True
    )
    rlrop = callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc", mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    ckpt = callbacks.ModelCheckpoint(
        best_path, monitor="val_pr_auc", mode="max", save_best_only=True
    )
    return [es, rlrop, ckpt]

# History plots 
def plot_history(h, title="Historia de entrenamiento"):
    if h is None:
        print("No hay history para graficar."); return
    hist = h.history
    metrics_to_plot = [
        ("loss", "val_loss", "Loss"),
        ("accuracy", "val_accuracy", "Accuracy"),
        ("auc", "val_auc", "AUC"),
        ("pr_auc", "val_pr_auc", "PR-AUC"),
        ("precision", "val_precision", "Precision"),
        ("recall", "val_recall", "Recall")
    ]
    metrics_to_plot = [(m, vm, lab) for (m, vm, lab) in metrics_to_plot if (m in hist and vm in hist)]
    n = len(metrics_to_plot)
    if n == 0:
        print("No hay métricas para graficar en este history."); return

    plt.figure(figsize=(5.5*n, 4))
    for i, (m, vm, lab) in enumerate(metrics_to_plot, 1):
        plt.subplot(1, n, i)
        plt.plot(hist[m], label=f"train_{m}")
        plt.plot(hist[vm], label=f"val_{vm}")
        plt.title(lab); plt.xlabel("Época")
        plt.legend(); plt.grid(alpha=0.2)
    plt.suptitle(title); plt.tight_layout(); plt.show()

def plot_cm(cm, title):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar()
    tick_marks = np.arange(2)
    classes = ["NORMAL","PNEUMONIA"]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Etiqueta real'); plt.xlabel('Predicción')
    plt.tight_layout(); plt.show()

# Evaluaciones 
def get_probs_from_ds(model, ds):
    y_true, y_prob = [], []
    for xb, yb in ds:
        y_prob.extend(model.predict(xb, verbose=0).ravel())
        y_true.extend(yb.numpy().astype(int))
    return np.array(y_true), np.array(y_prob)

def threshold_max_f1(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / (p + r + 1e-8)
    idx = np.argmax(f1)
    thr = t[idx] if idx < len(t) else 0.5
    return thr, p[idx], r[idx], f1[idx]

def threshold_for_recall(y_true, y_prob, target_recall=0.90):
    p, r, t = precision_recall_curve(y_true, y_prob)
    r, p = r[:-1], p[:-1]
    idx = np.where(r >= target_recall)[0]
    if len(idx) == 0:
        thr_f1, pf1, rf1, f1 = threshold_max_f1(y_true, y_prob)
        return thr_f1, pf1, rf1
    i = idx[-1]
    return t[i], p[i], r[i]

def evaluate_on_test(model, test_ds, threshold=0.5, name="Model"):
    y_true, y_prob = [], []
    for xb, yb in test_ds:
        y_prob.extend(model.predict(xb, verbose=0).ravel())
        y_true.extend(yb.numpy().astype(int))
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=["NORMAL","PNEUMONIA"])

    print(f"\n== {name} ==")
    print("ROC AUC:", roc)
    print("PR AUC :", pr_auc)
    print(cm)
    print(cr)

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "cm": cm,
        "report": cr,
        "threshold": float(threshold),
        "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred
    }

# Preprocesamiento (center-crop + resize)
def resize_crop_pil(im, size=(224,224)):
    im = im.convert("L")
    w, h = im.size
    target_w, target_h = size
    aspect = w / h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        im = im.crop((offset, 0, offset + new_w, h))
    else:
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        im = im.crop((0, offset, w, offset + new_h))

    return im.resize(size, Image.Resampling.BICUBIC)

def load_xray_for_model(path, model, img_size=224):
    with Image.open(path) as im:
        im = resize_crop_pil(im, (img_size, img_size))
        arr = np.array(im, dtype=np.float32) / 255.0
    in_ch = model.input_shape[-1]
    if in_ch == 1:
        arr = arr[..., np.newaxis]
    elif in_ch == 3:
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        raise ValueError(f"Canales de entrada no soportados: {in_ch}")
    return arr, im

# ===== Grad-CAM =====
def find_last_conv_and_owner(model):
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )

    # 1) Buscar de atrás hacia adelante en el modelo principal
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name, model

    # 2) Si no, buscar dentro de submodelos
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, conv_types):
                    return sub.name, layer

    raise ValueError("No se encontró capa tipo conv para Grad-CAM.")

def make_gradcam_heatmap_logits_owner(model, img_array,
                                      last_conv_name=None, owner_model=None,
                                      target="auto"):
    if last_conv_name is None or owner_model is None:
        last_conv_name, owner_model = find_last_conv_and_owner(model)
    conv_layer = owner_model.get_layer(last_conv_name)
    conv_output_tensor = conv_layer.output
    grad_model = tf.keras.models.Model(inputs=model.inputs,
                                       outputs=[conv_output_tensor, model.output])
    with tf.GradientTape() as tape:
        conv_out, prob = grad_model(img_array[np.newaxis, ...], training=False)
        p = tf.clip_by_value(prob, 1e-7, 1. - 1e-7)
        logit = tf.math.log(p / (1. - p))
        if target == "PNEUMONIA":
            loss = logit[:, 0]
        elif target == "NORMAL":
            loss = -logit[:, 0]
        else:
            loss = tf.where(prob[:, 0] >= 0.5, logit[:, 0], -logit[:, 0])
    grads   = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam     = tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)
    cam     = tf.nn.relu(cam[0])
    cam = cam - tf.reduce_min(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None],
                          (img_array.shape[0], img_array.shape[1])).numpy().squeeze()
    return cam

def predict_xray(path, model, threshold=0.5, img_size=224):
    arr, im_vis = load_xray_for_model(path, model, img_size)
    prob = float(model.predict(arr[np.newaxis, ...], verbose=0).ravel()[0])
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    return label, prob, im_vis, arr

def show_prediction_with_cam(path, model, model_name,
                             threshold=0.5, alpha=0.45,
                             last_conv_name=None, owner_model=None,
                             target="auto", img_size=224,
                             save_dir="resultados"):
    # --- Predicción ---
    label, prob, im_vis, arr = predict_xray(path, model, threshold, img_size)

    # --- Grad-CAM ---
    heat = make_gradcam_heatmap_logits_owner(
        model, arr,
        last_conv_name=last_conv_name,
        owner_model=owner_model,
        target=target
    )
    heat = heatmap_remove_letterbox(heat, arr)

    # --- Crear carpeta destino ---
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    # --- Guardar overlay ---
    overlay_path = os.path.join(save_dir, f"{base}_overlay.png")
    plt.figure(figsize=(6,6))
    plt.imshow(im_vis, cmap="gray")
    plt.imshow(heat, cmap="jet", alpha=alpha)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0); plt.close()

    # --- Preparar diagnóstico ---
    diagnosis = "NEUMONÍA" if prob >= threshold else "NO NEUMONÍA"

    # --- Guardar JSON ---
    meta = {
        "input_image": os.path.basename(path),
        "model_name": getattr(model, "name", model_name),
        "diagnosis": diagnosis,
        "probability": f"{prob*100:.1f}%",
        "threshold": f"{threshold*100:.1f}%",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "result_image": os.path.basename(overlay_path),
    }
    json_path = os.path.join(save_dir, f"{base}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --- Mostrar en consola ---
    print(f"\n Diagnóstico con {model_name}: {diagnosis}")
    print(f"    Probabilidad neumonía: {prob*100:.1f}% (umbral {threshold*100:.1f}%)")
    print(f" Guardado en {save_dir}: {os.path.basename(overlay_path)} y {os.path.basename(json_path)}")


def evaluate_with_val_threshold(model, val_ds, test_ds,
                                mode="f1", target_recall=0.90,
                                target_precision=0.90, name="Model"):
    yv, pv = get_probs_from_ds(model, val_ds)

    if mode == "f1":
        thr, P, R, F1 = threshold_max_f1(yv, pv)
        print(f"[{name}] Umbral VALIDACIÓN (máx F1): {thr:.3f} | P={P:.2f} R={R:.2f} F1={F1:.2f}")
    elif mode == "recall":
        thr, P, R = threshold_for_recall(yv, pv, target_recall=target_recall)
        print(f"[{name}] Umbral VALIDACIÓN (recall>={target_recall:.2f}): {thr:.3f} | P={P:.2f} R={R:.2f}")
    elif mode == "precision":
        thr, P, R = threshold_for_precision(yv, pv, target_precision=target_precision)
        print(f"[{name}] Umbral VALIDACIÓN (precision>={target_precision:.2f}): {thr:.3f} | P={P:.2f} R={R:.2f}")
    else:
        raise ValueError("mode debe ser 'f1', 'recall' o 'precision'")

    yt, pt = get_probs_from_ds(model, test_ds)
    y_pred = (pt >= thr).astype(int)

    print(f"\n== {name} (TEST con thr={thr:.3f}) ==")
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
    print("ROC AUC:", roc_auc_score(yt, pt))
    cm = confusion_matrix(yt, y_pred); print(cm)
    print(classification_report(yt, y_pred, target_names=["NORMAL","PNEUMONIA"]))

    return thr, (yt, pt, y_pred), cm

def plot_pr_curve(y_true, y_prob, title="Precision–Recall (validación)"):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5.5, 4))
    plt.plot(r, p, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def heatmap_remove_letterbox(heat, img_array):
    """
    Ajusta el mapa de calor para que no marque las zonas negras del letterbox.
    """
    # máscara: píxeles donde hay información real (no relleno negro)
    mask = (img_array.sum(axis=-1) > 0)

    # aplica máscara
    heat = heat * mask

    # normaliza de 0 a 1
    hmax = heat.max()
    if hmax > 0:
        heat = heat / hmax

    return heat


def threshold_for_precision(y_true, y_prob, target_precision=0.90):
    p, r, t = precision_recall_curve(y_true, y_prob)
    # p,r tienen len = len(t)+1 → alinear con t
    p, r = p[:-1], r[:-1]
    idx = np.where(p >= target_precision)[0]
    if len(idx) == 0:
        # cae a F1 si no alcanza esa precisión
        thr, pf1, rf1, f1 = threshold_max_f1(y_true, y_prob)
        return thr, pf1, rf1
    # Elegimos el UMBRAL MÁS ALTO que aún cumple la precisión:
    i = idx[-1]
    return t[i], p[i], r[i]