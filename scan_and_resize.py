
"""
Escaneo de calidad (oscura/blanca/varianza/tamaño) y reescalado con letterbox
de las imágenes del dataset original hacia ./chest_xray_resized (224x224).

Basado en tu notebook y las mejoras acordadas.
"""

import os, glob
from collections import Counter
from PIL import Image, ImageStat, UnidentifiedImageError

from config import DATASET_ORIGINAL, RESIZED_BASE

# Umbrales de control de calidad
BLACK_THRESHOLD_MEAN = 5
WHITE_THRESHOLD_MEAN = 240
LOW_VARIANCE_THRESH  = 5
MIN_W, MIN_H = 100, 200
TARGET = (224, 224)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def scan_and_flag(folder):
    recs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            path = os.path.join(root, fname)
            info = {
                "path": path,
                "class": os.path.basename(root),
                "ok": True,
                "reason": [],
                "w": None, "h": None,
                "mean": None, "var": None,
            }
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    info["w"], info["h"] = w, h
                    g = im.convert("L")
                    st = ImageStat.Stat(g)
                    info["mean"] = st.mean[0]
                    info["var"]  = st.var[0]

                    if info["mean"] <= BLACK_THRESHOLD_MEAN:
                        info["reason"].append("casi_negra")
                    if info["mean"] >= WHITE_THRESHOLD_MEAN:
                        info["reason"].append("casi_blanca")
                    if info["var"]  <= LOW_VARIANCE_THRESH:
                        info["reason"].append("baja_varianza")
                    if (w < MIN_W) or (h < MIN_H):
                        info["reason"].append("muy_pequena")

            except (UnidentifiedImageError, OSError) as e:
                info["ok"] = False
                info["reason"].append(f"ilegible:{e}")

            recs.append(info)
    return recs

def count_flags(recs):
    c = Counter()
    for r in recs:
        for tag in r["reason"]:
            c[tag] += 1
    return c

def resize_letterbox(im, size=(224, 224)):
    im = im.convert("L")
    w, h = im.size
    scale = min(size[0]/w, size[1]/h)
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    im_res = im.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("L", size, color=0)
    off_x, off_y = (size[0]-nw)//2, (size[1]-nh)//2
    canvas.paste(im_res, (off_x, off_y))
    return canvas

def process_split(split):
    src_split = os.path.join(DATASET_ORIGINAL, split)
    dst_split = os.path.join(RESIZED_BASE, split)
    ensure_dir(dst_split)

    for cls in ["NORMAL", "PNEUMONIA"]:
        src_cls = os.path.join(src_split, cls)
        if not os.path.isdir(src_cls):
            continue
        dst_cls = os.path.join(dst_split, cls)
        ensure_dir(dst_cls)

        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for p in glob.glob(os.path.join(src_cls, ext)):
                try:
                    with Image.open(p) as im:
                        out = resize_letterbox(im, TARGET)
                        out.save(os.path.join(dst_cls, os.path.basename(p)))
                except Exception as e:
                    print("Error:", p, "->", e)

def main():
    if not os.path.isdir(DATASET_ORIGINAL):
        raise SystemExit(f"No existe la carpeta original: {DATASET_ORIGINAL}")

    print("== Escaneando calidad ==")
    for split in ["train", "val", "test"]:
        recs = scan_and_flag(os.path.join(DATASET_ORIGINAL, split))
        print(f"{split.upper()} flags:", count_flags(recs))

    print("\n== Redimensionando a letterbox 224x224 ==")
    for split in ["train", "val", "test"]:
        print("Procesando", split)
        process_split(split)
    print("Listo. Carpeta destino:", RESIZED_BASE)

if __name__ == "__main__":
    main()
