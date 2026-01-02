# models.py
import tensorflow as tf
from keras import layers, models, regularizers

# === Bloques de Construcción ===

def squeeze_excite_block(input_tensor, ratio=16):
    """Módulo de atención de canales (Squeeze-and-Excitation)."""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.Multiply()([input_tensor, se])

def residual_block(x, filters, stride=1):
    """Bloque Residual con Separable Convolutions y SE."""
    shortcut = x
    # Ajuste de dimensiones si hay stride o cambio de filtros
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Rama principal
    x = layers.SeparableConv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Atención
    x = squeeze_excite_block(x)
    
    # Suma y activación final
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# === Modelos ===

def build_resnet_se(img_size=224, dropout=0.4):
    """Arquitectura personalizada de alto rendimiento (Grayscale)."""
    inp = layers.Input((img_size, img_size, 1))
    
    # Stem (Entrada)
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Backbone
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 256, stride=2)
    
    # Head (Clasificador)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    
    out = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inp, out, name="PneumoResNet_SE")

def build_efficientnet(img_size=224, dropout=0.4):
    """Wrapper para EfficientNetB0 (RGB). Devuelve (model, base_model)."""
    inp = layers.Input((img_size, img_size, 3))
    
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_tensor=inp, weights="imagenet"
    )
    base.trainable = False # Congelado inicial

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inp, out, name="EffNetB0_Transfer")
    return model, base