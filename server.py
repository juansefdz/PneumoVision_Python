import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from model_core import AIModelManager
from utils import prepare_image

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

ai_manager = AIModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando servidor y cargando modelos de IA...")
    ai_manager.load_models()
    yield
    print("Servidor apagándose. Liberando recursos...")

app = FastAPI(title="PneumoVision API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
def favicon():
    return Response(status_code=204)

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "IA Trainer System Online", "service": "PneumoVision API"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {
        "status": "healthy",
        "loaded_models": list(ai_manager.models.keys()),
        "service": "PneumoVision API"
    }

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...), 
    mode: str = Form(...) 
):
    try:
        # 1. Validar tipo de contenido si está presente
        if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400, 
                detail="Formato de archivo no soportado. Suba una imagen JPG, PNG o WEBP."
            )
            
        # 2. Leer bytes y verificar tamaño
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="El archivo subido está vacío.")
            
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"El archivo excede el tamaño máximo permitido de {MAX_FILE_SIZE // (1024*1024)}MB."
            )

        img_array = prepare_image(image_bytes)
        response = {}
        
        if mode == 'model_a':
            response['left'] = ai_manager.analyze('model_a', img_array)
            response['right'] = None
            
        elif mode == 'model_b':
            response['left'] = None
            response['right'] = ai_manager.analyze('model_b', img_array)
            
        elif mode == 'comparison':
            response['left'] = ai_manager.analyze('model_a', img_array)
            response['right'] = ai_manager.analyze('model_b', img_array)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Modo no válido. Use: model_a, model_b, o comparison"
            )
            
        return response

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Error crítico en endpoint /predict: {e}")
        raise HTTPException(status_code=500, detail="Error interno procesando la imagen.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)