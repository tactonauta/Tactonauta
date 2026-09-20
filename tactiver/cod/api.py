# -*- coding: utf-8 -*-
"""
api.py
======
API HTTP mínima para conectar el pipeline automático de segmentación
de gráficas con una interfaz web YA EXISTENTE (cualquier stack: React,
Vue, PHP, HTML plano, etc.). No trae interfaz propia: solo expone
endpoints que tu frontend puede llamar directamente.

Ejecutar:
    pip install -r requirements.txt
    python api.py
    (queda escuchando en http://localhost:5000)

------------------------------------------------------------------
FLUJO COMPLETO (PDF -> gráficas detectadas -> segmentación -> CSV)
------------------------------------------------------------------

1) POST /api/clasificar
   Subes el PDF completo. Corre pipeline_rapido.py (sin DePlot: solo
   extracción + filtros baratos + clasificación geométrica) y devuelve
   la lista de figuras encontradas, cada una con su tipo (línea, barra
   o torta, compuesta, indeterminado) y una URL de vista previa.

2) POST /api/segmentar/<id>
   Con el "id" que te devolvió /api/clasificar para una figura de tipo
   línea, corre el segmentador (segmentador.py) sobre ESA imagen
   puntual (ejes, curva, textos, calibración) y devuelve el mismo
   resumen + CSV + overlay que /api/procesar.

3) POST /api/procesar
   Igual que el paso 2, pero recibiendo la imagen directo (sin pasar
   primero por /api/clasificar) — útil si ya tienes la imagen de un
   solo gráfico y no un PDF completo.

------------------------------------------------------------------
ENDPOINTS
------------------------------------------------------------------

POST /api/clasificar
    Recibe un PDF (multipart/form-data, campo "pdf") y devuelve la
    lista de figuras detectadas.

    curl -F "pdf=@paper.pdf" http://localhost:5000/api/clasificar

    Respuesta:
    {
      "ok": true,
      "lote_id": "a1b2c3d4",
      "graficos": [
        {
          "id": "a1b2c3d4_page2_img1.png",
          "pagina": 2,
          "ancho": 420, "alto": 260,
          "tipo": "linea",
          "confianza": 0.81,
          "es_lineal": true,
          "preview_url": "/api/resultados/a1b2c3d4_page2_img1.png"
        },
        ...
      ]
    }

POST /api/segmentar/<id>
    <id> es el campo "id" que devolvió /api/clasificar para una figura
    con "es_lineal": true. No hace falta volver a subir el archivo.

    curl -X POST http://localhost:5000/api/segmentar/a1b2c3d4_page2_img1.png

    Respuesta: igual formato que /api/procesar (ver abajo).

POST /api/procesar
    Recibe una imagen (multipart/form-data, campo "imagen") y devuelve
    un JSON con el resumen de lo detectado + las URLs para descargar
    el CSV y el overlay de verificación.

    curl -F "imagen=@grafica.png" http://localhost:5000/api/procesar

    Respuesta:
    {
      "ok": true,
      "resumen": { ... },
      "csv_url": "/api/resultados/descripcion_xxxx.csv",
      "csv_download_url": "/api/resultados/descripcion_xxxx.csv/descargar",
      "overlay_url": "/api/resultados/overlay_xxxx.png"
    }

POST /api/procesar?formato=csv
    Igual que arriba, pero devuelve DIRECTAMENTE el archivo CSV como
    descarga (útil si tu interfaz solo necesita el archivo, sin JSON
    intermedio).

GET /api/resultados/<nombre_archivo>
    Sirve un archivo ya generado (CSV, overlay PNG o figura extraída
    del PDF) para visualizarlo inline (por ejemplo, en un <img>).

GET /api/resultados/<nombre_archivo>/descargar
    Igual, pero forzando la descarga (Content-Disposition: attachment).

GET /api/salud
    Chequeo simple de que la API está viva: {"ok": true}
------------------------------------------------------------------
"""

import os
import shutil
import uuid

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from segmentador import procesar_imagen
import pipeline_rapido
from generador_stl import generar_modelo_bana, generar_modelo_desde_recta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
STL_DIR = os.path.join(RESULTADOS_DIR, "stl")
CLASIFICACION_DIR = os.path.join(BASE_DIR, "resultados_rapido")
EXTENSIONES_VALIDAS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(STL_DIR, exist_ok=True)
os.makedirs(CLASIFICACION_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB máx (PDFs pesan más que una imagen suelta)


@app.after_request
def habilitar_cors(response):
    # CORS abierto para que tu interfaz (en otro dominio/puerto) pueda
    # llamar a esta API sin problemas. Restringe el origen si lo necesitas,
    # p. ej. response.headers["Access-Control-Allow-Origin"] = "https://tu-web.com"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def extension_valida(nombre_archivo):
    return "." in nombre_archivo and \
        nombre_archivo.rsplit(".", 1)[1].lower() in EXTENSIONES_VALIDAS


@app.route("/", methods=["GET"])
def index():
    return """
    <!doctype html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <title>Tactonauta API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 760px; margin: 48px auto; padding: 0 20px; color: #1f2937; }
            code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
            .card { border: 1px solid #d1d5db; border-radius: 12px; padding: 20px; background: #fff; }
            a { color: #2563eb; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Tactonauta</h1>
            <p>La API está funcionando en este servidor.</p>
            <p>Ruta de salud: <a href="/api/salud"><code>/api/salud</code></a></p>
            <p>Para abrir la interfaz visual de segmentación en Chrome:</p>
            <ol>
                <li>lanzá el servidor Flask en el puerto 5000</li>
                <li>abre la interfaz desde el archivo <code>tactiver/interfaz/interfaz_segmentador.html</code></li>
                <li>o servila localmente con <code>python -m http.server 8000</code> desde la carpeta <code>tactiver/interfaz</code></li>
            </ol>
            <p>Si quieres probar la API directamente, usa <code>http://127.0.0.1:5000/api/salud</code>.</p>
        </div>
    </body>
    </html>
    """


@app.route("/api/salud", methods=["GET"])
def salud():
    return jsonify({"ok": True})


def crear_stl_desde_imagen(ruta_imagen, prefijo="grafica"):
    """Segmenta una gráfica lineal y convierte sus extremos a una placa STL."""
    resultado = procesar_imagen(ruta_imagen, RESULTADOS_DIR)
    nombre = f"{prefijo}_{uuid.uuid4().hex[:8]}.stl"
    generar_modelo_desde_recta(
        resultado["puntos_curva"],
        archivo_salida=os.path.join(STL_DIR, nombre),
    )
    return resultado, nombre


@app.route("/api/generar-stl", methods=["POST"])
def generar_stl():
    """Genera la placa táctil de Fase 1 a partir de dos puntos en mm.

    JSON esperado: {"p1": [x, y], "p2": [x, y], "ancho": 210,
    "alto": 148, "intervalo_ticks": 25}. Los últimos tres campos son
    opcionales y las dimensiones se expresan en milímetros.
    """
    datos = request.get_json(silent=True)
    if not isinstance(datos, dict):
        return jsonify({"ok": False, "error": "Envía un JSON con p1 y p2."}), 400
    try:
        p1, p2 = datos["p1"], datos["p2"]
        if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in (p1, p2)):
            raise ValueError("p1 y p2 deben tener dos coordenadas.")
        p1 = (float(p1[0]), float(p1[1]))
        p2 = (float(p2[0]), float(p2[1]))
        ancho = float(datos.get("ancho", 210))
        alto = float(datos.get("alto", 148))
        intervalo = int(datos.get("intervalo_ticks", 25))
        nombre = f"grafica_tactil_{uuid.uuid4().hex[:8]}.stl"
        generar_modelo_bana(ancho, alto, p1, p2, intervalo, os.path.join(STL_DIR, nombre))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"Parámetros inválidos: {e}"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"No se pudo generar el STL: {e}"}), 500

    return jsonify({
        "ok": True,
        "stl_url": f"/api/resultados/stl/{nombre}",
        "stl_download_url": f"/api/resultados/stl/{nombre}/descargar",
    }), 201


@app.route("/api/generar-stl/<path:id_grafico>", methods=["POST"])
def generar_stl_desde_grafico(id_grafico):
    """Convierte una gráfica extraída previamente por /api/clasificar a STL."""
    nombre_seguro = secure_filename(id_grafico)
    ruta_imagen = os.path.join(RESULTADOS_DIR, nombre_seguro)
    if not os.path.isfile(ruta_imagen):
        return jsonify({"ok": False, "error": "No se encontró esa gráfica clasificada."}), 404
    try:
        resultado, nombre = crear_stl_desde_imagen(ruta_imagen, "grafica_tactil")
    except Exception as e:
        return jsonify({"ok": False, "error": f"No se pudo generar el STL: {e}"}), 500
    return jsonify({
        "ok": True,
        "resumen": resultado["resumen"],
        "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
        "csv_download_url": f"/api/resultados/{resultado['nombre_csv']}/descargar",
        "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
        "stl_url": f"/api/resultados/stl/{nombre}",
        "stl_download_url": f"/api/resultados/stl/{nombre}/descargar",
    }), 201


@app.route("/api/pdf-a-stl", methods=["POST"])
def pdf_a_stl():
    """Flujo completo: PDF -> rectas detectadas -> archivos STL."""
    archivo = request.files.get("pdf")
    if archivo is None or archivo.filename == "":
        return jsonify({"ok": False, "error": "No se envió ningún PDF (campo 'pdf')."}), 400
    if not archivo.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "El archivo debe ser un PDF."}), 400

    lote_id = uuid.uuid4().hex[:8]
    ruta_pdf = os.path.join(UPLOAD_DIR, f"{lote_id}_{secure_filename(archivo.filename)}")
    archivo.save(ruta_pdf)
    try:
        detectados = pipeline_rapido.detectar_graficos(
            ruta_pdf, os.path.join(CLASIFICACION_DIR, f"extraidas_{lote_id}")
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error al analizar el PDF: {e}"}), 500

    stls = []
    for grafico in detectados:
        if not grafico.get("es_lineal"):
            continue
        try:
            resultado, nombre = crear_stl_desde_imagen(grafico["file_path"], "grafica_tactil")
            stls.append({
                "pagina": grafico["page"], "tipo": grafico["tipo"],
                "resumen": resultado["resumen"],
                "stl_url": f"/api/resultados/stl/{nombre}",
                "stl_download_url": f"/api/resultados/stl/{nombre}/descargar",
            })
        except Exception as e:
            stls.append({
                "pagina": grafico["page"], "tipo": grafico["tipo"], "error": str(e),
            })
    return jsonify({"ok": True, "lote_id": lote_id, "n_detectados": len(detectados), "stls": stls})


# ------------------------------------------------------------------
# Paso 1: subir el PDF completo y detectar qué tipo de gráficas hay
# ------------------------------------------------------------------
@app.route("/api/clasificar", methods=["POST"])
def clasificar():
    archivo = request.files.get("pdf")

    if archivo is None or archivo.filename == "":
        return jsonify({"ok": False, "error": "No se envió ningún PDF (campo 'pdf')."}), 400

    if not archivo.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "El archivo debe ser un PDF."}), 400

    nombre_seguro = secure_filename(archivo.filename)
    lote_id = uuid.uuid4().hex[:8]
    ruta_pdf = os.path.join(UPLOAD_DIR, f"{lote_id}_{nombre_seguro}")
    archivo.save(ruta_pdf)

    output_dir = os.path.join(CLASIFICACION_DIR, f"extraidas_{lote_id}")

    try:
        detectados = pipeline_rapido.detectar_graficos(ruta_pdf, output_dir)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error al analizar el PDF: {e}"}), 500

    graficos = []
    for d in detectados:
        nombre_original = os.path.basename(d["file_path"])
        nombre_servido = f"{lote_id}_{nombre_original}"
        ruta_servida = os.path.join(RESULTADOS_DIR, nombre_servido)
        shutil.copy(d["file_path"], ruta_servida)
        graficos.append({
            "id": nombre_servido,
            "pagina": d["page"],
            "ancho": round(d["width"]),
            "alto": round(d["height"]),
            "tipo": d["tipo"],
            "confianza": d["confianza"],
            "es_lineal": d["es_lineal"],
            "preview_url": f"/api/resultados/{nombre_servido}",
        })

    return jsonify({"ok": True, "lote_id": lote_id, "graficos": graficos})


# ------------------------------------------------------------------
# Paso 2: segmentar UNA figura ya clasificada (por su id), sin volver
# a subir el archivo — el segmentador la toma directo de "resultados/"
# ------------------------------------------------------------------
@app.route("/api/segmentar/<path:id_grafico>", methods=["POST"])
def segmentar_extraido(id_grafico):
    nombre_seguro = secure_filename(id_grafico)
    ruta_imagen = os.path.join(RESULTADOS_DIR, nombre_seguro)

    if not os.path.isfile(ruta_imagen):
        return jsonify({"ok": False, "error": "No se encontró esa figura clasificada. Vuelve a subir el PDF."}), 404

    try:
        resultado = procesar_imagen(ruta_imagen, RESULTADOS_DIR)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error al segmentar la imagen: {e}"}), 500

    return jsonify({
        "ok": True,
        "resumen": resultado["resumen"],
        "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
        "csv_download_url": f"/api/resultados/{resultado['nombre_csv']}/descargar",
        "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
    })


# ------------------------------------------------------------------
# Segmentar una imagen suelta (sin pasar por /api/clasificar)
# ------------------------------------------------------------------
@app.route("/api/procesar", methods=["POST"])
def procesar():
    archivo = request.files.get("imagen")

    if archivo is None or archivo.filename == "":
        return jsonify({"ok": False, "error": "No se envió ninguna imagen (campo 'imagen')."}), 400

    if not extension_valida(archivo.filename):
        return jsonify({"ok": False, "error": "Formato no soportado. Usa PNG, JPG, JPEG, BMP o TIFF."}), 400

    nombre_seguro = secure_filename(archivo.filename)
    nombre_unico = f"{uuid.uuid4().hex[:8]}_{nombre_seguro}"
    ruta_subida = os.path.join(UPLOAD_DIR, nombre_unico)
    archivo.save(ruta_subida)

    try:
        resultado = procesar_imagen(ruta_subida, RESULTADOS_DIR)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error al procesar la imagen: {e}"}), 500

    # Si el frontend solo quiere el archivo CSV directo (sin JSON):
    if request.args.get("formato") == "csv":
        return send_from_directory(RESULTADOS_DIR, resultado["nombre_csv"], as_attachment=True)

    return jsonify({
        "ok": True,
        "resumen": resultado["resumen"],
        "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
        "csv_download_url": f"/api/resultados/{resultado['nombre_csv']}/descargar",
        "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
    })


@app.route("/api/resultados/<path:nombre_archivo>", methods=["GET"])
def servir_resultado(nombre_archivo):
    return send_from_directory(RESULTADOS_DIR, nombre_archivo, as_attachment=False)


@app.route("/api/resultados/<path:nombre_archivo>/descargar", methods=["GET"])
def descargar_resultado(nombre_archivo):
    return send_from_directory(RESULTADOS_DIR, nombre_archivo, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
