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

POST /api/segmentar/<id>?stl=1
    Igual, pero en la misma llamada genera también la lámina táctil y
    agrega "stl_url" / "stl_download_url" a la respuesta (o "stl_error"
    si la gráfica no daba para una lámina). Sin el parámetro no se
    genera nada: el STL tarda bastante más que la segmentación.

POST /api/generar-stl/<id>
    Segmenta esa figura y devuelve directamente la lámina STL.
    Cuerpo JSON opcional: {"ancho": 210, "alto": 148} (milímetros;
    mínimo 130 x 90).

POST /api/procesar
    Recibe una imagen (multipart/form-data, campo "imagen") y devuelve
    un JSON con el resumen de lo detectado + las URLs para descargar
    el CSV y el overlay de verificación.

    curl -F "imagen=@grafica.png" http://localhost:5000/api/procesar

    Respuesta:
    {
      "ok": true,
      "resumen": { ... },
      "advertencias": ["..."],
      "csv_url": "/api/resultados/descripcion_xxxx.csv",
      "csv_download_url": "/api/resultados/descripcion_xxxx.csv/descargar",
      "json_url": "/api/resultados/grafica_xxxx.json",
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
import json
import shutil
import uuid
import traceback
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
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024*1024  # 40 MB máx (PDFs pesan más que una imagen suelta)


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
    return send_from_directory(
        "../interfaz",
        "tactiverso (6).html"
    )


@app.route("/api/salud", methods=["GET"])
def salud():
    return jsonify({"ok": True})


# ------------------------------------------------------------------
# Puente segmentador -> generador de STL
# ------------------------------------------------------------------
# `procesar_imagen()` devuelve series, puntos y resumen, pero la calibración
# de los ejes (la recta píxel -> valor real) solo queda escrita en el JSON de
# resultados. `generar_modelo_desde_recta()` la busca en la clave "ejes", así
# que aquí se vuelve a juntar todo antes de llamarlo. Si el JSON no se puede
# leer, el generador reconstruye la escala a partir de los propios puntos.

def _leer_json_extra(resultado):
    """Devuelve (ejes, textos) del JSON del segmentador.

    Ninguno de los dos bloques viene en lo que procesar_imagen() regresa
    directamente: la calibración de los ejes y los títulos/etiquetas leídos
    por OCR solo quedan escritos en el archivo JSON de resultados. El
    generador de STL los necesita para dibujar números fieles a lo que
    realmente decía la gráfica (no una escala inventada) y para grabar el
    título y los nombres de los ejes en Braille.
    """
    ruta_json = resultado.get("ruta_json")
    if not ruta_json or not os.path.isfile(ruta_json):
        return {}, {}
    try:
        with open(ruta_json, encoding="utf-8") as f:
            datos = json.load(f)
    except (OSError, ValueError):
        return {}, {}
    return datos.get("ejes") or {}, datos.get("textos") or {}


def _payload_stl(resultado):
    """Arma la entrada completa que espera el generador de STL."""
    ejes, textos = _leer_json_extra(resultado)
    return {
        "series": resultado.get("series") or [],
        "puntos_curva": resultado.get("puntos_curva") or [],
        "resumen": resultado.get("resumen") or {},
        "ejes": ejes,
        "textos": textos,
    }


def crear_stl_desde_resultado(resultado, prefijo="grafica", dim_x=210.0, dim_y=148.0):
    """Convierte una segmentación ya hecha en una placa táctil STL.

    Devuelve (nombre_stl, advertencias). Lanza ValueError con un mensaje
    entendible si la gráfica no da para una lámina.
    """
    payload = _payload_stl(resultado)
    if not payload["series"] and not payload["puntos_curva"]:
        raise ValueError(
            "El segmentador no encontró ninguna curva en la imagen: no hay nada "
            "que llevar a la lámina táctil."
        )

    advertencias = list(resultado.get("advertencias") or [])
    if not (resultado.get("resumen") or {}).get("listo_para_stl"):
        advertencias.append(
            "La lámina se generó SIN calibración completa de los ejes: la forma de "
            "la curva es correcta, pero la escala y los números en Braille pueden "
            "faltar o no corresponder a los valores reales. Revisar el overlay "
            "antes de imprimir."
        )

    nombre = f"{prefijo}_{uuid.uuid4().hex[:8]}.stl"
    generar_modelo_desde_recta(
        payload,
        dim_x=dim_x,
        dim_y=dim_y,
        archivo_salida=os.path.join(STL_DIR, nombre),
    )
    return nombre, advertencias


def crear_stl_desde_imagen(ruta_imagen, prefijo="grafica", dim_x=210.0, dim_y=148.0):
    """Segmenta una gráfica lineal y genera su placa táctil STL.

    Devuelve (resultado_segmentador, nombre_stl, advertencias).
    """
    resultado = procesar_imagen(ruta_imagen, RESULTADOS_DIR)
    nombre, advertencias = crear_stl_desde_resultado(resultado, prefijo, dim_x, dim_y)
    return resultado, nombre, advertencias


def _dimensiones_placa(datos):
    """Lee ancho/alto (mm) del cuerpo de la petición, con los valores A5 por defecto."""
    try:
        return float(datos.get("ancho", 210)), float(datos.get("alto", 148))
    except (TypeError, ValueError):
        raise ValueError("'ancho' y 'alto' deben ser números en milímetros.")


def _respuesta_segmentacion(resultado):
    """Campos comunes que toda respuesta del segmentador devuelve al frontend."""
    return {
        "ok": True,
        "resumen": resultado["resumen"],
        "advertencias": resultado.get("advertencias", []),
        "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
        "csv_download_url": f"/api/resultados/{resultado['nombre_csv']}/descargar",
        "json_url": f"/api/resultados/{resultado['nombre_json']}",
        "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
    }


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
    """Convierte una gráfica extraída previamente por /api/clasificar a STL.

    Cuerpo JSON opcional: {"ancho": 210, "alto": 148} en milímetros.
    """
    nombre_seguro = secure_filename(id_grafico)
    ruta_imagen = os.path.join(RESULTADOS_DIR, nombre_seguro)
    if not os.path.isfile(ruta_imagen):
        return jsonify({"ok": False, "error": "No se encontró esa gráfica clasificada."}), 404

    try:
        dim_x, dim_y = _dimensiones_placa(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    try:
        resultado, nombre, advertencias = crear_stl_desde_imagen(
            ruta_imagen, "grafica_tactil", dim_x, dim_y
        )
    except ValueError as e:
        # La imagen se segmentó, pero no da para una lámina (sin curva, placa
        # demasiado chica...). Es un problema del contenido, no del servidor.
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception as e:
        print("ERROR GENERANDO STL:", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({
            "ok": False,
            "error": f"No se pudo generar el STL: {e}"
        }), 500

    respuesta = _respuesta_segmentacion(resultado)
    respuesta.update({
        "advertencias": advertencias,
        "stl_url": f"/api/resultados/stl/{nombre}",
        "stl_download_url": f"/api/resultados/stl/{nombre}/descargar",
    })
    return jsonify(respuesta), 201


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
            resultado, nombre, advertencias = crear_stl_desde_imagen(
                grafico["file_path"], "grafica_tactil"
            )
            stls.append({
                "pagina": grafico["page"], "tipo": grafico["tipo"],
                "resumen": resultado["resumen"],
                "advertencias": advertencias,
                "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
                "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
                "stl_url": f"/api/resultados/stl/{nombre}",
                "stl_download_url": f"/api/resultados/stl/{nombre}/descargar",
            })
        except Exception as e:
            # Una gráfica que falla no debe tumbar el lote entero.
            print(f"ERROR EN {grafico['file_path']}:", flush=True)
            print(traceback.format_exc(), flush=True)
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

    respuesta = _respuesta_segmentacion(resultado)

    # Con ?stl=1 la misma llamada devuelve también la lámina, sin volver a
    # segmentar la imagen. Sin el parámetro el comportamiento no cambia: la
    # generación del STL es lenta y no todas las pantallas la necesitan.
    if request.args.get("stl") in ("1", "true", "si", "sí"):
        try:
            nombre, advertencias = crear_stl_desde_resultado(resultado, "grafica_tactil")
        except ValueError as e:
            respuesta["stl_error"] = str(e)
        except Exception as e:
            print("ERROR GENERANDO STL:", flush=True)
            print(traceback.format_exc(), flush=True)
            respuesta["stl_error"] = f"No se pudo generar el STL: {e}"
        else:
            respuesta.update({
                "advertencias": advertencias,
                "stl_url": f"/api/resultados/stl/{nombre}",
                "stl_download_url": f"/api/resultados/stl/{nombre}/descargar",
            })

    return jsonify(respuesta)


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

    return jsonify(_respuesta_segmentacion(resultado))


@app.route("/api/resultados/<path:nombre_archivo>", methods=["GET"])
def servir_resultado(nombre_archivo):
    return send_from_directory(RESULTADOS_DIR, nombre_archivo, as_attachment=False)


@app.route("/api/resultados/<path:nombre_archivo>/descargar", methods=["GET"])
def descargar_resultado(nombre_archivo):
    return send_from_directory(RESULTADOS_DIR, nombre_archivo, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
