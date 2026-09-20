FROM python:3.11-slim

# Instalar Tesseract OCR
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*
# Directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY tactiver/cod/requirements.txt /app/requirements.txt

# Instalar dependencias Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar todo el proyecto
COPY . /app

# Puerto de Render
EXPOSE 10000

# Ejecutar Flask mediante Gunicorn
CMD ["gunicorn", "--chdir", "tactiver/cod", "--bind", "0.0.0.0:10000", "api:app"]
