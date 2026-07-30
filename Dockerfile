FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y compilar librerías
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente del proyecto y los modelos (.keras)
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Ejecutar Uvicorn en producción
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]