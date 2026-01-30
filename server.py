import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_core import AIModelManager
from utils import prepare_image

os.environ["TF_USE_LEGACY_KERAS"] = "1"


ai_manager = AIModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando servidor y cargando modelos de IA...")
    ai_manager.load_models()
    yield
    print("Servidor apagándose. Liberando recursos...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "IA Trainer System Online", "service": "PneumoVision API"}

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...), 
    mode: str = Form(...) 
):
    try:
    
        image_bytes = await file.read()
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
            return {"error": "Modo no válido. Use: model_a, model_b, o comparison"}
            
        return response

    except Exception as e:
        print(f"Error crítico en endpoint: {e}")
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)