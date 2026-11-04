"""Qué tipo de figura es cada imagen extraída del PDF (gráfico de líneas,
de barras, tabla, foto...), con el modelo DocumentFigureClassifier v2.0 de
Docling (IBM): https://huggingface.co/docling-project/DocumentFigureClassifier-v2.0
(licencia Apache 2.0). Es una EfficientNet-B0 entrenada con figuras de
documentos, en formato ONNX (modelos/figuras_docling_v2.onnx, 16 MB): corre
con onnxruntime, sin PyTorch.

Antes se decidía con una regla geométrica (¿la forma más grande está hecha
de líneas finas?), y las tablas, los recuadros de texto, los radares o los
histogramas pasaban como "gráfico de líneas". Con los 15 PDFs de prueba
(188 figuras revisadas a mano): la regla encontraba 16 de 20 gráficos de
líneas con 46 falsos; este modelo, 20 de 20 con 9 falsos (más 9 páginas o
recuadros que sí tenían un gráfico de líneas adentro).

Memoria: ~100 MB mientras se clasifica. El modelo se carga al empezar y se
libera al terminar (clasificar()), no queda ocupando memoria entre pedidos:
el servidor tiene 512 MB.
"""
import ctypes
import gc
import os

import numpy as np

MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos", "figuras_docling_v2.onnx")

# En el orden de salida del modelo (tarjeta del modelo en Hugging Face)
ETIQUETAS = ["logo", "photograph", "icon", "engineering_drawing", "line_chart", "bar_chart", "other", "table",
             "flow_chart", "screenshot_from_computer", "signature", "screenshot_from_manual", "geographical_map",
             "pie_chart", "page_thumbnail", "stamp", "music", "calendar", "qr_code", "bar_code", "full_page_image",
             "scatter_plot", "chemistry_structure", "topographical_map", "crossword_puzzle", "box_plot"]
_MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
_DESVIO = np.array([0.47853944, 0.4732864, 0.47434163], np.float32)

# Lo que el resto del sistema entiende por "tipo"
TIPOS = {"line_chart": "linea", "bar_chart": "barra_o_torta", "pie_chart": "barra_o_torta",
         "scatter_plot": "dispersion", "box_plot": "otro", "table": "tabla"}


def disponible():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return os.path.isfile(MODELO)


def _entrada(ruta):
    from PIL import Image
    with Image.open(ruta) as img:
        img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0 - _MEDIA) / _DESVIO
    return x.transpose(2, 0, 1)[None]


def _devolver_memoria():
    """Devuelve al sistema la memoria ya liberada (lo que dejó la extracción
    de las figuras): en Linux, malloc no la devuelve sola y el modelo se
    sumaría encima. En otros sistemas no hace nada."""
    gc.collect()
    try:
        import fitz
        fitz.TOOLS.store_shrink(100)
    except Exception:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def clasificar(rutas):
    """[(etiqueta, probabilidad)] de cada imagen, o None si el modelo no
    está disponible (y entonces se usa la regla geométrica)."""
    if not rutas or not disponible():
        return None
    _devolver_memoria()
    import onnxruntime as ort
    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = 1
    opciones.inter_op_num_threads = 1
    opciones.enable_cpu_mem_arena = False      # menos memoria retenida entre imágenes
    opciones.enable_mem_pattern = False
    sesion = ort.InferenceSession(MODELO, sess_options=opciones, providers=["CPUExecutionProvider"])
    entrada = sesion.get_inputs()[0].name
    resultados = []
    try:
        for ruta in rutas:
            try:
                logits = sesion.run(None, {entrada: _entrada(ruta)})[0][0]
            except Exception:
                resultados.append(("other", 0.0))
                continue
            p = np.exp(logits - logits.max())
            p /= p.sum()
            i = int(p.argmax())
            resultados.append((ETIQUETAS[i], float(p[i])))
    finally:
        del sesion
        _devolver_memoria()
    return resultados
