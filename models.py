import os
import tensorflow as tf
from keras import layers, models, regularizers


def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(
        max(8, filters // ratio), 
        activation='relu', 
        kernel_initializer='he_normal', 
        use_bias=False
    )(se)
    se = layers.Dense(
        filters, 
        activation='sigmoid', 
        kernel_initializer='he_normal', 
        use_bias=False
    )(se)
    return layers.Multiply()([input_tensor, se])

def residual_block(x, filters, stride=1, drop_rate=0.1):
    shortcut = x
    
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding='same', 
            kernel_initializer='he_normal', use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.SeparableConv2D(
        filters, 3, strides=stride, padding='same', 
        depthwise_initializer='he_normal', pointwise_initializer='he_normal', 
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if drop_rate > 0:
        x = layers.SpatialDropout2D(drop_rate)(x)

    x = layers.SeparableConv2D(
        filters, 3, padding='same', 
        depthwise_initializer='he_normal', pointwise_initializer='he_normal', 
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = squeeze_excite_block(x)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def get_augmenter():
    return models.Sequential([
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="robust_aug")

def build_resnet_se(img_size=224, dropout=0.4):
    inp = layers.Input((img_size, img_size, 1))
    x = get_augmenter()(inp)
    
    # Stem (Entrada de alta definición)
    x = layers.Conv2D(
        32, 3, strides=2, padding='same', 
        kernel_initializer='he_normal', use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Backbone Residual Profundo con SE Attention
    x = residual_block(x, 64, stride=1, drop_rate=0.05)
    x = residual_block(x, 64, stride=2, drop_rate=0.1)
    x = residual_block(x, 128, stride=1, drop_rate=0.1)
    x = residual_block(x, 128, stride=2, drop_rate=0.15)
    x = residual_block(x, 256, stride=1, drop_rate=0.15)
    x = residual_block(x, 256, stride=2, drop_rate=0.2)
    
    # Head de Clasificación: Dual Pooling (Global Average + Global Max Pooling)
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    x = layers.Dense(
        128, 
        kernel_initializer='he_normal', 
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    
    out = layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal', dtype='float32')(x)
    
    return models.Model(inp, out, name="PneumoResNet_SE_Advanced")

def build_efficientnet(img_size=224, dropout=0.4):
    
    inp = layers.Input((img_size, img_size, 3))
    
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_tensor=inp, weights="imagenet"
    )
    base.trainable = False 

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", dtype='float32')(x)
    
    model = models.Model(inp, out, name="EffNetB0_Transfer")
    return model, base