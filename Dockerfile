FROM python:3.11-slim

# Instalar Tesseract OCR, con español (ñ, tildes: "España", "Perú"; sin él
# salían "Espafia", "Pert"). segmentador._idioma_ocr usa spa+eng si están.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-spa \
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

# Ejecutar Flask mediante Gunicorn.
# --timeout 300: generar un STL (varias series, texturas, texto en Braille)
# puede tardar bien por encima de los 30s por defecto de gunicorn; sin este
# valor, el worker se mata a mitad de la generación y la API devuelve 502.
CMD ["gunicorn", "--chdir", "tactiver/cod", "--bind", "0.0.0.0:10000", "--timeout", "300", "api:app"]
