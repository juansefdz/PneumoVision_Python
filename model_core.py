import os
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
        path_a = os.path.join(base_dir, 'artifacts', 'eff_b0_best.keras')
        path_b = os.path.join(base_dir, 'artifacts', 'custom_graph') 
    
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
                print(f"Error A: {e}")

        # 2. Modelo B: Custom 
        if os.path.exists(path_b):
            try:
                self.models['model_b'] = tf.saved_model.load(path_b)
                print("Modelo B (Custom) cargado.")
            except Exception as e:
                print(f"Error cargando Modelo B: {e}")

    def make_gradcam_heatmap(self, model, img_array, last_conv_layer_name):
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs], 
                [model.get_layer(last_conv_layer_name).output, model.output]
            )
        except Exception as e:
            print(f"Error creando modelo GradCAM: {e}")
            return np.zeros((224, 224))

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def analyze(self, model_key, img_array):
        if model_key not in self.models:
            return {"error": f"Modelo {model_key} no cargado"}

        model = self.models[model_key]
        heatmap_b64 = None
        
        try:
          
            if hasattr(model, 'serve'): 
                input_tensor = img_array
                if img_array.shape[-1] == 3:
                    input_tensor = tf.image.rgb_to_grayscale(img_array)
                
                raw_pred = model.serve(input_tensor)
                preds = raw_pred.numpy()
            else: 
                preds = model.predict(img_array)
                
              
                try:
                    layer = self.layer_names.get(model_key)
                    if layer:
                        heatmap = self.make_gradcam_heatmap(model, img_array, layer)
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