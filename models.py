# models.py
from tensorflow.keras import layers, models, optimizers, metrics, regularizers, applications

def build_cnn(img_size=224):
    inp = layers.Input(shape=(img_size, img_size, 1))
    def conv_block(x, f, p=0.0):
        x = layers.Conv2D(f, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        if p: x = layers.SpatialDropout2D(p)(x)
        x = layers.MaxPooling2D()(x)
        return x

    x = conv_block(inp, 32)
    x = conv_block(x,   64)
    x = conv_block(x,  128, p=0.1)
    x = conv_block(x,  256, p=0.1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="PneumoniaCNN_Grayscale_v2")

def build_efficientnet_b0(img_size=224, dropout=0.3):
    inp = layers.Input(shape=(img_size, img_size, 3), name="input_rgb")
    base = applications.EfficientNetB0(include_top=False, input_tensor=inp, weights="imagenet")
    base.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="pred")(x)
    return models.Model(inp, out, name="EffNetB0_RGB"), base

def compile_model(m, lr=1e-3):
    m.compile(
        optimizer=optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.AUC(name="auc"),
            metrics.AUC(name="pr_auc", curve="PR"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )

# Alias útil (misma firma que compile_model)
compile_eff = compile_model
