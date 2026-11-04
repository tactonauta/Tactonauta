# Modelos

## figuras_docling_v2.onnx

DocumentFigureClassifier v2.0, de Docling (IBM):
https://huggingface.co/docling-project/DocumentFigureClassifier-v2.0

- Licencia: Apache 2.0.
- EfficientNet-B0 (4 M de parámetros) entrenada para clasificar figuras de
  documentos en 26 tipos (gráfico de líneas, de barras, tabla, foto,
  diagrama de flujo, captura de pantalla...).
- Es el `model.onnx` publicado en esa página, sin cambios (16 MB).
- Lo usa `clasificador_figuras.py`, con onnxruntime.
