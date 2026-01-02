# PneumoVision – Python 🫁

Sistema avanzado para la clasificación de radiografías de tórax (NORMAL vs. PNEUMONIA) utilizando arquitecturas de Deep Learning modernas y técnicas de visión artificial.

## 🚀 Características Principales

### 🧠 Modelos Híbridos

El proyecto implementa dos enfoques de modelado distintos:

1.  **PneumoResNet_SE (Custom)**:

    - Arquitectura diseñada desde cero para eficiencia.
    - Usa **Convoluciones Separables** y **Bloques Residuales**.
    - Integra módulos **Squeeze-and-Excitation (SE)** para atención de canales.
    - Entrada: Escala de grises (Grayscale) normalizada.

2.  **EfficientNetB0 (Transfer Learning)**:
    - Modelo preentrenado en ImageNet.
    - Estrategia de entrenamiento en dos fases: **Feature Extraction** (congelado) + **Fine-Tuning** (descongelado parcial de últimas capas).
    - Entrada: RGB con preprocesamiento nativo de EfficientNet.

### 🛠 Pipeline de Datos Robusto (`data_pipeline.py`)

- **Aumentación Dinámica**: Aplica rotaciones, zoom, traslaciones, contraste y brillo aleatorio solo durante el entrenamiento para mejorar la generalización.
- **Balanceo Automático**: Calcula `class_weights` inversos para manejar el desbalance de clases en el dataset.
- **Eficiencia**: Uso de `tf.data.AUTOTUNE`, caché y prefetch para maximizar el uso de GPU.

### ⚙️ Entrenamiento Avanzado (`trainer.py`)

- **Optimizadores**: Soporte automático para **AdamW** (si está disponible) o Adam.
- **Regularización**: Uso de **Label Smoothing** (0.1) para prevenir sobreconfianza en las predicciones.
- **Callbacks**: Checkpointing del mejor modelo y monitorización constante.

---

## 📋 Requisitos

- Python 3.9+ (recomendado entorno virtual).
- GPU recomendada para entrenamiento.

### Instalación

```bash
pip install -r requirements.txt
```

---

## 📂 Estructura del Dataset

El sistema espera que los datos estén en la carpeta `chest_xray_resized/` (definido en `config.py`), o puedes usar el script de escaneo para generarla desde el original. Estructura esperada:

```
chest_xray_resized/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> **Nota**: Puedes ajustar las rutas y parámetros como `IMG_SIZE` o `BATCH_SIZE` directamente en `config.py`.

---

## 🏃‍♂️ Ejecución

El flujo de trabajo se ha unificado en scripts modulares.

### 1. Entrenamiento (`train.py`)

Usa el script de entrenamiento indicando qué modelo deseas entrenar:

**Opción A: Modelo Personalizado (PneumoResNet_SE)**

```bash
python train.py --model custom
```

_Mejor para:_ Entrenamiento rápido, inferencia ligera.

**Opción B: EfficientNetB0**

```bash
python train.py --model effnet
```

_Mejor para:_ Máxima precisión utilizando conocimiento previo de ImageNet.

### 2. Evaluación

Para evaluar los modelos guardados en el set de prueba y generar métricas:

```bash
python evaluate.py
```

### 3. Backend (API)

Para servir el modelo y hacer predicciones vía API:

```bash
uvicorn backend.app:app --reload
```

---

## � Resultados y Métricas

El sistema monitorea múltiples métricas durante el entrenamiento para asegurar un rendimiento balanceado:

- **Accuracy**
- **AUC (Area Under Curve)**
- **Precision & Recall** (Crítico en diagnósticos médicos)

Los mejores modelos se guardan automáticamente en la carpeta `artifacts/`:

- `custom_best.keras`
- `effnet_best.keras`

## 🔍 Interpretabilidad

El proyecto incluye utilidades para **Grad-CAM**, permitiendo visualizar qué áreas de la radiografía activaron la decisión del modelo, proporcionando transparencia en el diagnóstico automatizado.
