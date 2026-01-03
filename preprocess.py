# preprocess.py
import os
import shutil
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import config

# Importamos kagglehub para la descarga automática
import kagglehub

def download_dataset_if_needed(target_dir):
    """
    Descarga el dataset usando KaggleHub si no existe localmente.
    """
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        # Verificación rápida: si tiene subcarpetas 'train', asumimos que está bien
        if os.path.exists(os.path.join(target_dir, "train")):
            return

    print("Dataset no encontrado. Iniciando descarga automática con KaggleHub...")
    print("(Esto puede tardar unos minutos dependiendo de tu conexión)")
    
    try:
        # Descarga el dataset oficial compatible con este proyecto
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"Descarga completada en caché: {path}")
        
        # El dataset suele venir anidado como: .../cache/chest_xray/chest_xray/...
        # Buscamos dónde está la carpeta 'train' real
        source_data_dir = None
        for root, dirs, files in os.walk(path):
            if "train" in dirs and "test" in dirs:
                source_data_dir = root
                break
        
        if source_data_dir is None:
            raise ValueError("No se encontró la estructura train/test esperada en la descarga.")

        print(f"Moviendo archivos desde {source_data_dir} a {target_dir}...")
        
        # Mover (o copiar) los contenidos a nuestra carpeta de proyecto
        shutil.copytree(source_data_dir, target_dir, dirs_exist_ok=True)
        print("Dataset colocado correctamente en el proyecto.")

    except Exception as e:
        print(f"Error en la descarga automática: {e}")
        print("Intenta descargar manualmente o revisa tu conexión.")
        raise e

def resize_letterbox(img, size=(224, 224), color=(0, 0, 0)):
    """Redimensiona manteniendo aspect ratio (Letterbox)."""
    h, w = img.shape[:2]
    target_h, target_w = size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    return canvas

def process_file(args):
    """Procesa un solo archivo."""
    src_path, dst_path, size = args
    if os.path.exists(dst_path):
        return
    try:
        img = cv2.imread(src_path)
        if img is None: return
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        img_processed = resize_letterbox(img, size=size)
        cv2.imwrite(dst_path, img_processed)
    except Exception as e:
        print(f"Error procesando {src_path}: {e}")

def prepare_dataset():
    """Flujo principal: Descarga (si hace falta) -> Procesa."""
    raw_dir = os.path.join(config.BASE_DIR, "chest_xray")
    processed_dir = config.DATASET_BASE 
    
    # 1. PASO NUEVO: Descarga automática
    download_dataset_if_needed(raw_dir)

    print(f"Verificando/Procesando imágenes en: {processed_dir}")
    
    tasks = []
    for split in ["train", "val", "test"]:
        for category in ["NORMAL", "PNEUMONIA"]:
            src_folder = os.path.join(raw_dir, split, category)
            dst_folder = os.path.join(processed_dir, split, category)
            
            os.makedirs(dst_folder, exist_ok=True)
            
            if not os.path.exists(src_folder):
                continue
                
            files = os.listdir(src_folder)
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src = os.path.join(src_folder, fname)
                    dst = os.path.join(dst_folder, fname)
                    tasks.append((src, dst, (config.IMG_SIZE, config.IMG_SIZE)))
    
    if tasks:
        print(f"    Procesando {len(tasks)} imágenes...")
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_file, tasks), total=len(tasks), unit="img"))
    else:
        print("No hay imágenes nuevas para procesar.")
        
    print("Dataset listo para entrenamiento.")
    return True