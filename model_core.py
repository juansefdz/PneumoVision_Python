import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
from utils import encode_heatmap_to_base64

class AIModelManager:
    def __init__(self):
        self.models = {}
        self.class_names = ['Normal', 'Neumonía'] 

        self.layer_names = {'model_a': 'top_activation'} 

    def load_models(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path_a = os.path.join(base_dir, 'artifacts', 'effnet_best.keras')
        if not os.path.exists(path_a):
            path_a = os.path.join(base_dir, 'artifacts', 'eff_b0_best.keras')
            
        path_b_graph = os.path.join(base_dir, 'artifacts', 'custom_graph') 
        path_b_keras = os.path.join(base_dir, 'artifacts', 'custom_best.keras')
    
        if os.path.exists(path_a):
            try:
                self.models['model_a'] = tf.keras.models.load_model(path_a, compile=False)
                print("(EfficientNet) cargado.")
                
                layer_found = False
                for layer in self.models['model_a'].layers:
                    if layer.name == 'top_activation':
                        layer_found = True
                        break
                
                if not layer_found:
                    print("top_activation no encontrado. Buscando capa alternativa...")
                    for layer in reversed(self.models['model_a'].layers):
                        if 'conv' in layer.name or 'activation' in layer.name:
                            self.layer_names['model_a'] = layer.name
                            print(f"capa GradCAM detectada: {layer.name}")
                            break
            except Exception as e:
                print(f"Error cargando Modelo A: {e}")

        # 2. Modelo B: Custom 
        if os.path.exists(path_b_keras):
            try:
                self.models['model_b'] = tf.keras.models.load_model(path_b_keras, compile=False)
                print("Modelo B (Custom Keras) cargado.")
            except Exception as e:
                print(f"Error cargando Modelo B Keras: {e}")
                try:
                    import keras
                    self.models['model_b'] = keras.models.load_model(path_b_keras, compile=False)
                    print("Modelo B cargado via Keras nativo.")
                except Exception as e2:
                    print(f"Error secundario cargando Modelo B: {e2}")
        elif os.path.exists(path_b_graph):
            try:
                self.models['model_b'] = tf.saved_model.load(path_b_graph)
                print("Modelo B (Custom SavedModel) cargado.")
            except Exception as e:
                print(f"Error cargando Modelo B SavedModel: {e}")

    def make_gradcam_heatmap(self, model, img_array, last_conv_layer_name):
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs], 
                [model.get_layer(last_conv_layer_name).output, model.output]
            )
        except Exception as e:
            print(f"Error creando modelo GradCAM: {e}")
            return np.zeros((224, 224))

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            last_conv_layer_output, preds = grad_model(img_tensor)
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0]
            else:
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        if grads is None:
            return np.zeros((224, 224))
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = tf.zeros_like(heatmap)

        return heatmap.numpy()

    def analyze(self, model_key, img_array):
        if model_key not in self.models:
            return {"error": f"Modelo {model_key} no cargado"}

        model = self.models[model_key]
        heatmap_b64 = None
        
        try:
            input_tensor = tf.cast(img_array, tf.float32)
            
            # Normalización y conversión de color según modelo
            if hasattr(model, 'serve'): 
                if input_tensor.shape[-1] == 3:
                    input_tensor = tf.image.rgb_to_grayscale(input_tensor)
                if tf.reduce_max(input_tensor) > 1.0:
                    input_tensor = input_tensor / 255.0
                
                raw_pred = model.serve(input_tensor)
                preds = raw_pred.numpy()
            else: 
                # Si el modelo espera 1 canal (Custom ResNet) o si es model_b
                if model_key == 'model_b' or (hasattr(model, 'input_shape') and model.input_shape[-1] == 1):
                    if input_tensor.shape[-1] == 3:
                        input_tensor = tf.image.rgb_to_grayscale(input_tensor)
                    if tf.reduce_max(input_tensor) > 1.0:
                        input_tensor = input_tensor / 255.0
                
                preds = model.predict(input_tensor, verbose=0)
                
                try:
                    layer = self.layer_names.get(model_key)
                    if layer:
                        heatmap = self.make_gradcam_heatmap(model, input_tensor, layer)
                        if np.max(heatmap) > 0: 
                            heatmap_b64 = encode_heatmap_to_base64(heatmap)
                except Exception as e:
                    print(f"Error GradCAM: {e}")

            # Interpretación de resultados
            if preds.shape[-1] == 1:
                score = float(preds[0][0])
                if score > 0.5:
                    label = self.class_names[1] # Neumonía
                    confidence = score
                else:
                    label = self.class_names[0] # Normal
                    confidence = 1.0 - score
            else:
                top_idx = np.argmax(preds[0])
                label = self.class_names[top_idx]
                confidence = float(np.max(preds[0]))

            return {
                "model_name": model_key,
                "prediction": label,
                "confidence": round(confidence, 4),
                "heatmap": heatmap_b64
            }
            
        except Exception as e:
            print(f"Error fatal analizando {model_key}: {e}")
            return {"error": str(e)}