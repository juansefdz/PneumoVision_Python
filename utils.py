import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf

def prepare_image(image_bytes, target_size=(224, 224)):
  
    # Convertir bytes a objeto de imagen PIL
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # Redimensionar (Resize)
    img = img.resize(target_size)
    
    # Convertir a array de Numpy
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    

    return img_array

def encode_heatmap_to_base64(heatmap_array, original_img_size=(224, 224)):
   
    heatmap = np.uint8(255 * heatmap_array)
    
    # Aplicar mapa de color (JET)
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Redimensionar
    jet_heatmap = cv2.resize(jet_heatmap, original_img_size)
    
    # BGR a RGB
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    
    # Guardar en buffer
    pil_img = Image.fromarray(jet_heatmap)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    return "data:image/png;base64," + new_image_string