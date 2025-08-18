# PneumoVision – Python

Clasificación de radiografías de tórax (NORMAL vs. PNEUMONIA) usando CNN personalizada y EfficientNetB0 preentrenada, con visualización mediante Grad-CAM.

### ⚙️ Requisitos

- Python 3.9+ (se recomienda usar entorno virtual).

#### Instalar dependencias:

`pip install -r requirements.txt`

###  Estructura esperada del dataset📂

Debes colocar el dataset en la carpeta ./chest_xray/ con esta estructura:

    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/


> Si usas el dataset de Kaggle "Chest X-Ray Images (Pneumonia)", la estructura ya coincide.⚠️ Si usas el dataset de Kaggle "Chest X-Ray Images (Pneumonia)", la estructura ya coincide.

### ▶️ Ejecución

Crear y activar entorno virtual (opcional, pero recomendado):

`python -m venv .venv`
# Windows
`.venv\Scripts\activate`
# macOS/Linux
`source .venv/bin/activate`


## Instalar dependencias:

`pip install -r requirements.txt`


### Ejecutar el pipeline completo:

`python main.py`

### 🔄 Qué hace main.py

- Escaneo y preprocesamiento:

- Filtra imágenes problemáticas (muy oscuras, blancas, pequeñas).

- Redimensiona todas las imágenes con letterbox a 224×224, guardándolas en ./chest_xray_resized/.

- Entrenamiento de modelos:

- 🧩 CNN personalizada (grayscale + normalización propia).

🧠-  EfficientNetB0 (preentrenada en ImageNet, convertida a RGB, entrenada en dos fases: congelada + fine-tuning).

### Evaluación:

- Calcula métricas: Accuracy, AUC, PR-AUC, Precision, Recall.

- Genera matriz de confusión y reportes de clasificación.

- Guarda los mejores modelos en ./artifacts/.

### 📊 Resultados esperados

- cnn_best.keras → mejor CNN en escala de grises.

- eff_b0_best.keras y pneumonia_effnet.keras → mejores checkpoints de EfficientNetB0.

- Gráficas de historia de entrenamiento y métricas impresas en consola.

###  Interpretabilidad (Grad-CAM)
🔍 Interpretabilidad (Grad-CAM)

Incluye utilidades para generar mapas de calor Grad-CAM, que resaltan las zonas de la radiografía que influyeron en la decisión del modelo.

Ejemplo de uso en un script de inferencia:

    from utils import show_prediction_with_cam
    import tensorflow as tf

    model = tf.keras.models.load_model("./artifacts/pneumonia_effnet.keras", compile=False)
    show_prediction_with_cam("ejemplo.jpg", model, model_name="EffNetB0")


Esto genera:

- Imagen original

- Heatmap Grad-CAM

- Overlay con predicción y probabilidad

### 📌 Notas

**El entrenamiento puede ser intensivo → se recomienda usar GPU.**

class_weight se calcula automáticamente para balancear el dataset.

Para reproducibilidad se fijan semillas y (opcional) determinismo en GPU.