# models.py
import tensorflow as tf
from keras import layers, models, optimizers, metrics

# ====== (opcional) AdamW de tensorflow-addons ======
try:
    import tensorflow_addons as tfa
    HAS_ADAMW = True
except Exception:
    HAS_ADAMW = False


# ===================================================
#                CNN v2 (grayscale)
# ===================================================

def _conv_bn_relu(x, filters, k=3):
    """Bloque conv separable + BN + ReLU."""
    x = layers.SeparableConv2D(filters, k, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_cnn(img_size=224, channels=1, dropout=0.3):
    """
    CNN liviana y robusta para rayos X en escala de grises.
    Devuelve un modelo con nombre 'PneumoniaCNN_Grayscale_v2'.
    """
    inp = layers.Input((img_size, img_size, channels))
    x = _conv_bn_relu(inp, 32);  x = layers.MaxPooling2D()(x)
    x = _conv_bn_relu(x, 64);    x = layers.MaxPooling2D()(x)
    x = _conv_bn_relu(x, 128);   x = layers.MaxPooling2D()(x)
    x = _conv_bn_relu(x, 256);   x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inp, out, name="PneumoniaCNN_Grayscale_v2")


def compile_model(model, lr=1e-3, wd=1e-4, use_adamw=True):
    """
    Compila con AdamW si está disponible; si no, cae a Adam estándar.
    Métricas incluyen AUC-ROC y PR-AUC (recomendado con desbalance).
    """
    if use_adamw and HAS_ADAMW:
        opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    else:
        opt = optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.AUC(name="auc"),
            metrics.AUC(name="pr_auc", curve="PR"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )


# ===================================================
#        EfficientNetB0 (para train_effnet.py)
# ===================================================

def build_efficientnet_b0(img_size=224, dropout=0.3):
    """
    Backbone preentrenado en ImageNet. Espera entrada RGB (3 canales).
    Devuelve (model, base) para facilitar fine-tuning.
    """
    inp = layers.Input(shape=(img_size, img_size, 3), name="input_rgb")
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_tensor=inp, weights="imagenet"
    )
    base.trainable = False  # fase 1 congelada

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="pred")(x)
    model = models.Model(inp, out, name="EffNetB0_RGB")
    return model, base


def compile_eff(model, lr=1e-3, wd=1e-4, use_adamw=True):
    """Compilación para EfficientNetB0."""
    if use_adamw and HAS_ADAMW:
        opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    else:
        opt = optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.AUC(name="auc"),
            metrics.AUC(name="pr_auc", curve="PR"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )
