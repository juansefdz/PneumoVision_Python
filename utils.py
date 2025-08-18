# utils.py
import os, json, datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# ===== Callbacks unificados (PR-AUC) =====
def get_callbacks(best_path, patience=8):
    es = callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                 patience=patience, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max",
                                        factor=0.5, patience=3, min_lr=1e-6)
    ckpt = callbacks.ModelCheckpoint(best_path, monitor="val_pr_auc",
                                     mode="max", save_best_only=True)
    return [es, rlrop, ckpt]

# ===== History plots =====
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

def evaluate_on_test(model, test_ds, threshold=0.5, name="Model"):
    y_true, y_prob = [], []
    for xb, yb in test_ds:
        y_prob.extend(model.predict(xb, verbose=0).ravel())
        y_true.extend(yb.numpy().astype(int))
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n== {name} ==")
    print("ROC AUC:", roc_auc_score(y_true, y_prob))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["NORMAL","PNEUMONIA"]))

# ===== Grad-CAM =====
def resize_letterbox_pil(im, size=(224,224)):
    from PIL import Image
    im = im.convert("L")
    w, h = im.size
    scale = min(size[0]/w, size[1]/h)
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    im_res = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("L", size, color=0)
    off_x, off_y = (size[0]-nw)//2, (size[1]-nh)//2
    canvas.paste(im_res, (off_x, off_y))
    return canvas

def load_xray_for_model(path, model, img_size=224):
    from PIL import Image
    with Image.open(path) as im:
        im = resize_letterbox_pil(im, (img_size, img_size))
        arr = np.array(im, dtype=np.float32) / 255.0
    in_ch = model.input_shape[-1]
    if in_ch == 1:
        arr = arr[..., np.newaxis]
    elif in_ch == 3:
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        raise ValueError(f"Canales de entrada no soportados: {in_ch}")
    return arr, im

def find_last_conv_and_owner(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, model
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name, layer
    raise ValueError("No se encontró capa Conv2D para Grad-CAM.")

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

def heatmap_remove_letterbox(heat, img_array):
    mask = (img_array.sum(axis=-1) > 0)
    heat = heat * mask
    hmax = heat.max()
    if hmax > 0: heat = heat / hmax
    return heat

def show_prediction_with_cam(path, model, model_name,
                             threshold=0.5, alpha=0.45,
                             last_conv_name=None, owner_model=None,
                             target="auto", img_size=224,
                             save_dir=None):
    label, prob, im_vis, arr = predict_xray(path, model, threshold, img_size)
    heat = make_gradcam_heatmap_logits_owner(model, arr,
                                             last_conv_name=last_conv_name,
                                             owner_model=owner_model,
                                             target=target)
    heat = heatmap_remove_letterbox(heat, arr)

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Análisis con {model_name}: {os.path.basename(path)}", fontsize=16)

    plt.subplot(1, 3, 1); plt.imshow(im_vis, cmap="gray"); plt.title("Radiografía"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(heat, cmap="jet");   plt.title("Grad-CAM");    plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(im_vis, cmap="gray"); plt.imshow(heat, cmap="jet", alpha=alpha)
    plt.title(f"Pred: {label} — {prob*100:.1f}%"); plt.axis("off")

    plt.tight_layout(); plt.show()
    print(f"Diagnóstico de {model_name}: {label} — Prob PNEUMONIA = {prob*100:.2f}% (umbral {threshold*100:.0f}%)")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path_img = os.path.join(save_dir, "image.png")
        im_vis.save(path_img)
        path_heat = os.path.join(save_dir, "heatmap.png")
        plt.figure(figsize=(4,4)); plt.imshow(heat, cmap="jet"); plt.axis("off")
        plt.tight_layout(); plt.savefig(path_heat, bbox_inches="tight", pad_inches=0); plt.close()
        path_overlay = os.path.join(save_dir, "overlay.png")
        plt.figure(figsize=(4,4)); plt.imshow(im_vis, cmap="gray")
        plt.imshow(heat, cmap="jet", alpha=alpha); plt.axis("off")
        plt.tight_layout(); plt.savefig(path_overlay, bbox_inches="tight", pad_inches=0); plt.close()
        meta = {
            "input_image": os.path.basename(path),
            "model_name": getattr(model, "name", model_name),
            "predicted_label": label,
            "probability_pneumonia": float(prob),
            "threshold": float(threshold),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "assets": {"image": "image.png", "heatmap": "heatmap.png", "overlay": "overlay.png"},
        }
        with open(os.path.join(save_dir, "result.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"✅ Assets guardados en: {save_dir}")

def predict_xray(path, model, threshold=0.5, img_size=224):
    arr, im_vis = load_xray_for_model(path, model, img_size)
    prob = float(model.predict(arr[np.newaxis, ...], verbose=0).ravel()[0])
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    return label, prob, im_vis, arr
