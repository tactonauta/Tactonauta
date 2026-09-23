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
import secrets
import shutil
import uuid
import traceback
from flask import Flask, request, jsonify, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import db
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

# Clave para firmar la cookie de sesión (login de Puntito). En Render se debe
# definir la variable de entorno SECRET_KEY con un valor fijo y secreto: sin
# eso, cada reinicio del contenedor generaría una clave nueva y cerraría la
# sesión de todo el mundo. El valor de respaldo es solo para correr en local.
_SECRET_KEY = os.environ.get("SECRET_KEY")
if not _SECRET_KEY:
    print(
        "[AVISO] SECRET_KEY no está definida: usando una clave temporal solo "
        "para desarrollo local. En Render, definila como variable de entorno "
        "o las sesiones se van a cerrar solas en cada reinicio.",
        flush=True,
    )
    _SECRET_KEY = secrets.token_hex(32)
app.secret_key = _SECRET_KEY


@app.after_request
def habilitar_cors(response):
    # CORS abierto para que tu interfaz (en otro dominio/puerto) pueda
    # llamar a esta API sin problemas. Con login por cookie de sesión, un
    # Access-Control-Allow-Origin fijo en "*" no alcanza: el navegador
    # bloquea las cookies en pedidos con credenciales salvo que el origen se
    # devuelva reflejado (no "*") y se declare Allow-Credentials. Si no viene
    # cabecera Origin (p. ej. curl), se deja "*" como antes.
    origen = request.headers.get("Origin")
    if origen:
        response.headers["Access-Control-Allow-Origin"] = origen
        response.headers["Access-Control-Allow-Credentials"] = "true"
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def extension_valida(nombre_archivo):
    return "." in nombre_archivo and \
        nombre_archivo.rsplit(".", 1)[1].lower() in EXTENSIONES_VALIDAS


# ====================================================================
# AUTENTICACIÓN — Fase 1 de Puntito (tactiverso (10).html)
# ====================================================================
# Cuentas reales (antes vivían en el localStorage del navegador, así que un
# supervisor en otra computadora nunca veía las solicitudes de su
# estudiante). Contraseñas con werkzeug.security (ya era dependencia del
# proyecto), sesión con la cookie firmada de Flask. Los datos van a
# db.py: SQLite local mientras no haya credenciales de Turso configuradas,
# Turso cuando las haya — mismo código en los dos casos.

ROLES_VALIDOS = ("estudiante", "supervisor", "imprenta")

_CAMPOS_REGISTRO = {
    "estudiante": ("nombre", "correo", "codigo", "facultad", "carrera", "clave"),
    "supervisor": ("nombre", "correo", "entidad", "clave"),
    "imprenta": ("nombre", "correo", "usuario", "area", "clave"),
}


def _vacio(v):
    return v is None or not str(v).strip()


def _iniciar_sesion(usuario):
    session.clear()
    session["usuario_id"] = usuario["id"]
    session["rol"] = usuario["rol"]


def _conectar_imprenta_automatica(supervisor):
    """Igual que el prototipo: en cuanto haya una imprenta registrada, el
    supervisor se conecta a ella sin tener que hacer nada."""
    if supervisor.get("imprenta_predeterminada_id"):
        return supervisor
    imprenta = db.primera_imprenta()
    if not imprenta:
        return supervisor
    return db.actualizar_usuario(supervisor["id"], {"imprenta_predeterminada_id": imprenta["id"]})


def _enriquecer_estudiante(usuario):
    """Si es un estudiante con supervisor conectado, agrega el código de ese
    supervisor (para prellenar "Cambiar código de supervisor" en el
    frontend, sin que tenga que resolverlo por su cuenta)."""
    if usuario and usuario.get("rol") == "estudiante" and usuario.get("supervisor_predeterminado_id"):
        supervisor = db.buscar_por_id(usuario["supervisor_predeterminado_id"])
        usuario["supervisor_codigo"] = supervisor["usuario"] if supervisor else None
    return usuario


@app.route("/api/auth/registro/<rol>", methods=["POST"])
def auth_registro(rol):
    if rol not in ROLES_VALIDOS:
        return jsonify({"ok": False, "error": "Rol inválido."}), 400

    datos = request.get_json(silent=True) or {}
    campos = _CAMPOS_REGISTRO[rol]
    if any(_vacio(datos.get(c)) for c in campos):
        return jsonify({"ok": False, "error": "Completa todos los campos para registrarte."}), 400

    correo = str(datos["correo"]).strip()
    if db.buscar_por_correo(rol, correo):
        return jsonify({
            "ok": False,
            "error": f"Ya existe una cuenta de {rol} con ese correo. Inicia sesión.",
        }), 409

    if rol == "imprenta":
        nombre_usuario = str(datos["usuario"]).strip()
        if db.buscar_por_usuario("imprenta", nombre_usuario):
            return jsonify({"ok": False, "error": "Ese nombre de usuario ya está en uso. Elige otro."}), 409

    clave_hash = generate_password_hash(str(datos["clave"]))
    usuario = db.crear_usuario(rol, {**datos, "correo": correo}, clave_hash)

    if rol == "supervisor":
        usuario = _conectar_imprenta_automatica(usuario)

    _iniciar_sesion(usuario)
    return jsonify({"ok": True, "usuario": _enriquecer_estudiante(usuario)}), 201


@app.route("/api/auth/login/<rol>", methods=["POST"])
def auth_login(rol):
    if rol not in ROLES_VALIDOS:
        return jsonify({"ok": False, "error": "Rol inválido."}), 400

    datos = request.get_json(silent=True) or {}
    correo = str(datos.get("correo") or "").strip()
    clave = str(datos.get("clave") or "")
    if _vacio(correo) or _vacio(clave):
        return jsonify({"ok": False, "error": "Completa correo y contraseña."}), 400

    usuario = db.buscar_por_correo(rol, correo, incluir_clave=True)
    if not usuario or not check_password_hash(usuario["clave_hash"], clave):
        return jsonify({"ok": False, "error": "Correo o contraseña incorrectos."}), 401

    usuario.pop("clave_hash", None)
    if rol == "supervisor":
        usuario = _conectar_imprenta_automatica(usuario)

    _iniciar_sesion(usuario)
    return jsonify({"ok": True, "usuario": _enriquecer_estudiante(usuario)})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/yo", methods=["GET"])
def auth_yo():
    usuario_id = session.get("usuario_id")
    if not usuario_id:
        return jsonify({"ok": False})
    usuario = db.buscar_por_id(usuario_id)
    if not usuario:
        # La cuenta ya no existe (borrada a mano en la base, por ejemplo):
        # limpiamos la cookie para no quedar en un estado inconsistente.
        session.clear()
        return jsonify({"ok": False})
    return jsonify({"ok": True, "usuario": _enriquecer_estudiante(usuario)})


@app.route("/api/auth/conectar-supervisor", methods=["POST"])
def auth_conectar_supervisor():
    if session.get("rol") != "estudiante":
        return jsonify({"ok": False, "error": "Iniciá sesión como estudiante primero."}), 401

    datos = request.get_json(silent=True) or {}
    codigo = str(datos.get("codigo") or "").strip()
    if _vacio(codigo):
        return jsonify({"ok": False, "error": "Escribe el código que te dio tu supervisor."}), 400

    supervisor = db.buscar_por_usuario("supervisor", codigo)
    if not supervisor:
        return jsonify({"ok": False, "error": "Ese código no corresponde a ningún supervisor registrado."}), 404

    usuario = db.actualizar_usuario(session["usuario_id"], {"supervisor_predeterminado_id": supervisor["id"]})
    return jsonify({"ok": True, "usuario": _enriquecer_estudiante(usuario)})


@app.route("/api/auth/reintentar-imprenta", methods=["POST"])
def auth_reintentar_imprenta():
    """El supervisor se registró antes de que existiera ninguna imprenta.
    Vuelve a intentar la conexión automática con la que haya ahora."""
    if session.get("rol") != "supervisor":
        return jsonify({"ok": False, "error": "Iniciá sesión como supervisor primero."}), 401
    usuario = db.buscar_por_id(session["usuario_id"])
    usuario = _conectar_imprenta_automatica(usuario)
    return jsonify({"ok": True, "usuario": usuario})


# ====================================================================
# SOLICITUDES — Fase 2 de Puntito
# ====================================================================
# El estudiante ya clasificó su PDF con /api/clasificar (existente, más
# abajo) y eligió qué gráficos de línea quiere convertir; acá se guarda esa
# elección como una solicitud real, visible para su supervisor. Todavía no
# dispara la segmentación/STL de verdad — eso es la fase siguiente
# ("Autorizar" en la pantalla de supervisor).

@app.route("/api/solicitudes", methods=["POST"])
def crear_solicitud():
    if session.get("rol") != "estudiante":
        return jsonify({"ok": False, "error": "Iniciá sesión como estudiante primero."}), 401

    estudiante = db.buscar_por_id(session["usuario_id"])
    if not estudiante or not estudiante.get("supervisor_predeterminado_id"):
        return jsonify({
            "ok": False,
            "error": "Conectate con un supervisor antes de enviar una solicitud.",
        }), 400

    datos = request.get_json(silent=True) or {}
    figuras = datos.get("figuras")
    if not isinstance(figuras, list) or not figuras:
        return jsonify({"ok": False, "error": "Elegí al menos un gráfico para enviar."}), 400

    solicitud = db.crear_solicitud(
        estudiante_id=estudiante["id"],
        supervisor_id=estudiante["supervisor_predeterminado_id"],
        lote_id=str(datos.get("lote_id") or ""),
        archivo_nombre=str(datos.get("archivo_nombre") or ""),
        figuras=figuras,
    )
    return jsonify({"ok": True, "solicitud": solicitud}), 201


@app.route("/api/solicitudes/mias", methods=["GET"])
def mis_solicitudes():
    if session.get("rol") != "estudiante":
        return jsonify({"ok": False, "error": "Iniciá sesión como estudiante primero."}), 401
    return jsonify({"ok": True, "solicitudes": db.solicitudes_de_estudiante(session["usuario_id"])})


def _con_estudiante(solicitud):
    """Agrega los datos del estudiante (nombre, correo, código, facultad,
    carrera) a una solicitud, para que el supervisor los vea sin tener que
    resolverlos aparte."""
    if not solicitud:
        return solicitud
    estudiante = db.buscar_por_id(solicitud["estudiante_id"]) or {}
    solicitud["estudiante"] = {
        "nombre": estudiante.get("nombre"),
        "correo": estudiante.get("correo"),
        "codigo": estudiante.get("codigo"),
        "facultad": estudiante.get("facultad"),
        "carrera": estudiante.get("carrera"),
    }
    return solicitud


@app.route("/api/solicitudes/supervisor", methods=["GET"])
def solicitudes_supervisor():
    if session.get("rol") != "supervisor":
        return jsonify({"ok": False, "error": "Iniciá sesión como supervisor primero."}), 401
    solicitudes = [_con_estudiante(s) for s in db.solicitudes_de_supervisor(session["usuario_id"])]
    return jsonify({"ok": True, "solicitudes": solicitudes})


# ====================================================================
# FASE 3 — Autorizar (segmentación + STL real) y flujo de imprenta
# ====================================================================
# "Autorizar" reusa exactamente el mismo puente segmentador->STL que ya
# usan /api/generar-stl/<id> y /api/pdf-a-stl (crear_stl_desde_imagen, más
# abajo) — no hay una ruta nueva de generación, solo se la conecta acá.
#
# Nota sobre almacenamiento: la imagen original de cada figura vive en
# RESULTADOS_DIR desde que se clasificó el PDF (paso 3 del estudiante). Ese
# disco es efímero en el plan gratuito de Render — si el contenedor se
# reinició entre que el estudiante subió el PDF y el supervisor autoriza,
# esa imagen ya no está. Por eso cada figura se procesa en un try/except
# propio: una que falle (imagen perdida, o cualquier otro error) no tira
# abajo el resto de la solicitud, y queda marcada con su propio "error" en
# vez de romper todo silenciosamente. Subir esos archivos a un storage
# persistente (Cloudflare R2) es el siguiente paso natural, pendiente.

@app.route("/api/solicitudes/<int:id_solicitud>/autorizar", methods=["POST"])
def autorizar_solicitud(id_solicitud):
    if session.get("rol") != "supervisor":
        return jsonify({"ok": False, "error": "Iniciá sesión como supervisor primero."}), 401

    solicitud = db.buscar_solicitud(id_solicitud)
    if not solicitud or solicitud["supervisor_id"] != session["usuario_id"]:
        return jsonify({"ok": False, "error": "No se encontró esa solicitud."}), 404
    if solicitud["estado_supervisor"] != "en_espera":
        return jsonify({"ok": False, "error": "Esta solicitud ya fue autorizada."}), 400

    figuras_actualizadas = []
    algun_exito = False

    for figura in solicitud["figuras"]:
        figura_actualizada = dict(figura)
        nombre_seguro = secure_filename(figura.get("id") or "")
        ruta_imagen = os.path.join(RESULTADOS_DIR, nombre_seguro)

        if not nombre_seguro or not os.path.isfile(ruta_imagen):
            figura_actualizada["error"] = (
                "La imagen original ya no está disponible en el servidor "
                "(pudo reiniciarse desde que se subió el PDF). Hay que volver "
                "a subir el documento para esta gráfica."
            )
            figuras_actualizadas.append(figura_actualizada)
            continue

        try:
            resultado, nombre_stl, advertencias = crear_stl_desde_imagen(
                ruta_imagen, "grafica_tactil"
            )
            figura_actualizada.update({
                "resumen": resultado["resumen"],
                "advertencias": advertencias,
                "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
                "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
                "stl_url": f"/api/resultados/stl/{nombre_stl}",
                "stl_download_url": f"/api/resultados/stl/{nombre_stl}/descargar",
            })
            algun_exito = True
        except Exception as e:
            print(f"ERROR AUTORIZANDO figura {figura.get('id')} de la solicitud {id_solicitud}:", flush=True)
            print(traceback.format_exc(), flush=True)
            figura_actualizada["error"] = str(e)

        figuras_actualizadas.append(figura_actualizada)

    solicitud = db.actualizar_solicitud(id_solicitud, {
        "estado_supervisor": "aprobada",
        "stl_generado": algun_exito,
        "figuras": figuras_actualizadas,
    })
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/<int:id_solicitud>/enviar-imprenta", methods=["POST"])
def enviar_a_imprenta(id_solicitud):
    if session.get("rol") != "supervisor":
        return jsonify({"ok": False, "error": "Iniciá sesión como supervisor primero."}), 401

    solicitud = db.buscar_solicitud(id_solicitud)
    if not solicitud or solicitud["supervisor_id"] != session["usuario_id"]:
        return jsonify({"ok": False, "error": "No se encontró esa solicitud."}), 404
    if solicitud["estado_supervisor"] != "aprobada":
        return jsonify({"ok": False, "error": "Primero hay que autorizar la solicitud."}), 400
    if solicitud["estado_imprenta"] != "no_enviada":
        return jsonify({"ok": False, "error": "Ya fue enviada a la imprenta."}), 400

    supervisor = db.buscar_por_id(session["usuario_id"])
    if not supervisor or not supervisor.get("imprenta_predeterminada_id"):
        return jsonify({"ok": False, "error": "No tenés una imprenta conectada."}), 400

    solicitud = db.actualizar_solicitud(id_solicitud, {
        "imprenta_id": supervisor["imprenta_predeterminada_id"],
        "estado_imprenta": "pendiente",
        "orden": f"TV-{id_solicitud:05d}",
    })
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/imprenta", methods=["GET"])
def solicitudes_imprenta():
    if session.get("rol") != "imprenta":
        return jsonify({"ok": False, "error": "Iniciá sesión como imprenta primero."}), 401
    solicitudes = [_con_estudiante(s) for s in db.solicitudes_de_imprenta(session["usuario_id"])]
    return jsonify({"ok": True, "solicitudes": solicitudes})


_TRANSICIONES_IMPRENTA = {"pendiente": "imprimiendo", "imprimiendo": "impreso"}


@app.route("/api/solicitudes/<int:id_solicitud>/estado-imprenta", methods=["POST"])
def avanzar_estado_imprenta(id_solicitud):
    """Avanza un paso en la cola (pendiente -> imprimiendo -> impreso). No
    recibe el estado destino del frontend: siempre avanza uno solo, así no
    hay forma de saltarse un paso mandando el JSON equivocado."""
    if session.get("rol") != "imprenta":
        return jsonify({"ok": False, "error": "Iniciá sesión como imprenta primero."}), 401

    solicitud = db.buscar_solicitud(id_solicitud)
    if not solicitud or solicitud["imprenta_id"] != session["usuario_id"]:
        return jsonify({"ok": False, "error": "No se encontró esa solicitud."}), 404

    siguiente = _TRANSICIONES_IMPRENTA.get(solicitud["estado_imprenta"])
    if not siguiente:
        return jsonify({"ok": False, "error": "Esta solicitud ya está impresa."}), 400

    solicitud = db.actualizar_solicitud(id_solicitud, {"estado_imprenta": siguiente})
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/", methods=["GET"])
def index():
    # Puntito (tactiverso 10): flujo de 3 roles (estudiante/supervisor/
    # imprenta) con cuentas reales. Reemplaza a tactiverso (6).html, que
    # servía acá antes (flujo más simple, sin cuentas ni aprobación).
    return send_from_directory(
        "../interfaz",
        "tactiverso (10).html"
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


def crear_stl_desde_resultado(
    resultado, prefijo="grafica", dim_x=210.0, dim_y=148.0, incluir_etiquetas=False
):
    """Convierte una segmentación ya hecha en una placa táctil STL.

    `incluir_etiquetas=False` (el valor por defecto, por ahora): la lámina
    sale sin números ni títulos en Braille, para revisar primero la forma de
    ejes y curvas. Los márgenes que ya les reserva espacio no se tocan, así
    que agregarlos más adelante (poner `incluir_etiquetas=True`) no requiere
    volver a acomodar la placa.

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
    if not incluir_etiquetas:
        advertencias.append(
            "Lámina sin números ni títulos en Braille todavía (pendiente de "
            "activar): los márgenes ya quedan reservados para agregarlos."
        )

    nombre = f"{prefijo}_{uuid.uuid4().hex[:8]}.stl"
    generar_modelo_desde_recta(
        payload,
        dim_x=dim_x,
        dim_y=dim_y,
        archivo_salida=os.path.join(STL_DIR, nombre),
        incluir_etiquetas=incluir_etiquetas,
    )
    return nombre, advertencias


def crear_stl_desde_imagen(
    ruta_imagen, prefijo="grafica", dim_x=210.0, dim_y=148.0, incluir_etiquetas=False
):
    """Segmenta una gráfica lineal y genera su placa táctil STL.

    Devuelve (resultado_segmentador, nombre_stl, advertencias).
    """
    resultado = procesar_imagen(ruta_imagen, RESULTADOS_DIR)
    nombre, advertencias = crear_stl_desde_resultado(
        resultado, prefijo, dim_x, dim_y, incluir_etiquetas
    )
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
