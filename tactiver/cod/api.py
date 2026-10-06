# -*- coding: utf-8 -*-
"""
api.py
======
Servidor Flask de Tactonauta: sirve la interfaz (tactiver/interfaz/
tactiverso (10).html) y la API que usa.

Ejecutar:
    pip install -r requirements.txt
    python api.py
    (queda escuchando en http://localhost:5000)

------------------------------------------------------------------
FLUJO
------------------------------------------------------------------
1. Estudiante: sube el PDF (/api/clasificar), elige las gráficas de
   líneas y las envía a su supervisor (/api/solicitudes).
2. Supervisor: "Revisar" (/api/solicitudes/<id>/revisar) segmenta cada
   gráfica; la revisa al lado de la original y de la vista previa de la
   lámina (/api/lamina/vista); la corrige en la pizarra si hace falta
   (/api/corregir para recalcular, .../figuras/<i>/corregir para guardar
   la lectura y los ajustes de la lámina); da el visto bueno y "Autoriza"
   (/api/solicitudes/<id>/autorizar): recién ahí se genera el STL, la
   narración JSON y el CSV de Hand_Tracking de cada gráfica.
3. Supervisor: "Enviar a imprenta"; la imprenta las pasa por su tablero
   (/api/solicitudes/imprenta, .../estado-imprenta).

La placa es siempre de 22 x 22 cm (ver generador_stl.PLACA_ANCHO).

------------------------------------------------------------------
ENDPOINTS
------------------------------------------------------------------
Sesión:     /api/auth/registro/<rol>, /api/auth/login/<rol>, /api/auth/logout,
            /api/auth/yo, /api/auth/conectar-supervisor,
            /api/auth/reintentar-imprenta
Estudiante: POST /api/clasificar (PDF, campo "pdf") -> figuras detectadas
            POST /api/solicitudes, GET /api/solicitudes/mias
Supervisor: GET /api/solicitudes/supervisor
            POST /api/solicitudes/<id>/revisar | autorizar | rechazar |
                 enviar-imprenta
            POST /api/solicitudes/<id>/figuras/<i>/visto-bueno | corregir |
                 quitar | restaurar | reemplazar
            POST /api/solicitudes/<id>/figuras/agregar
            GET  /api/solicitudes/<id>/pdf
            POST /api/lamina/vista       vista previa de la lámina (sin STL)
            POST /api/corregir/<correccion_id>   recalcula una lectura
Imprenta:   GET /api/solicitudes/imprenta
            POST /api/solicitudes/<id>/estado-imprenta
Archivos:   GET /api/resultados/<archivo>[/descargar]
QR:         GET /q/<token>  el CSV de Hand_Tracking de una lámina (lo abre
            el QR impreso; ver _url_qr)
Salud:      GET /api/salud -> {"ok": true}
------------------------------------------------------------------
"""

import glob
import hashlib
import os
import json
import math
import re
import secrets
import shutil
import time
import uuid
import traceback

import cv2
from flask import Flask, has_request_context, request, jsonify, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import numpy as np

import db
from segmentador import aplicar_correcciones, procesar_imagen
import narracion
import pipeline_rapido
from generador_stl import (
    _RE_AJUSTE, PLACA_ALTO, PLACA_ANCHO, aplicar_ajustes, generar_modelo_desde_recta,
)

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


LARGO_MAX_CAMPO = 200
MAX_FIGURAS_POR_SOLICITUD = 50


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

    # Solo los campos propios del rol: antes se guardaba el JSON entero, así
    # que un supervisor podía mandar su propio "usuario" (código de
    # conexión) y, si ya existía, el alta fallaba con error 500.
    limpios = {c: str(datos[c]).strip()[:LARGO_MAX_CAMPO] for c in campos if c != "clave"}
    correo = limpios["correo"]
    if db.buscar_por_correo(rol, correo):
        return jsonify({
            "ok": False,
            "error": f"Ya existe una cuenta de {rol} con ese correo. Inicia sesión.",
        }), 409

    if rol == "imprenta":
        nombre_usuario = limpios["usuario"]
        if db.buscar_por_usuario("imprenta", nombre_usuario):
            return jsonify({"ok": False, "error": "Ese nombre de usuario ya está en uso. Elige otro."}), 409

    clave_hash = generate_password_hash(str(datos["clave"]))
    usuario = db.crear_usuario(rol, limpios, clave_hash)

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

    figuras = _figuras_de_solicitud(figuras)
    if not figuras:
        return jsonify({
            "ok": False,
            "error": "Ninguno de los gráficos enviados está en el servidor. Vuelve a subir el PDF.",
        }), 400

    solicitud = db.crear_solicitud(
        estudiante_id=estudiante["id"],
        supervisor_id=estudiante["supervisor_predeterminado_id"],
        lote_id=str(datos.get("lote_id") or ""),
        archivo_nombre=str(datos.get("archivo_nombre") or ""),
        figuras=figuras,
    )
    return jsonify({"ok": True, "solicitud": solicitud}), 201


def _figuras_de_solicitud(figuras):
    """Arma las figuras de una solicitud en el SERVIDOR, a partir de lo que
    /api/clasificar realmente extrajo, en vez de guardar tal cual lo que
    manda el navegador. Antes se podía enviar, por ejemplo, un
    "stl_download_url" con "javascript:..." que el supervisor veía como
    botón "Descargar STL", o apuntar "imagen_url" a cualquier lado. Solo se
    aceptan figuras cuyo archivo existe en resultados/."""
    limpias = []
    for f in figuras[:MAX_FIGURAS_POR_SOLICITUD]:
        if not isinstance(f, dict):
            continue
        nombre = secure_filename(str(f.get("id") or ""))
        ruta = os.path.join(RESULTADOS_DIR, nombre)
        if not nombre or not os.path.isfile(ruta):
            continue
        ext = _extraccion(ruta)
        try:
            pagina = int(ext.get("pagina") or f.get("pagina") or 0)
        except (TypeError, ValueError):
            pagina = 0
        pie = ext.get("pie_figura") or ""
        limpias.append({
            "id": nombre,
            "titulo": "Gráfico de línea",
            "caption": pie or f"Gráfico de líneas sin pie de figura en la página {pagina}",
            "pagina": pagina,
            "imagen_url": f"/api/resultados/{nombre}",
            "pie_figura": pie,
        })
    return limpias


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
# Nota sobre almacenamiento: la imagen original de cada figura vive en
# RESULTADOS_DIR desde que se clasificó el PDF (paso 3 del estudiante). Ese
# disco es efímero en el plan gratuito de Render — si el contenedor se
# reinició entre que el estudiante subió el PDF y el supervisor autoriza,
# esa imagen ya no está. Por eso cada figura se procesa en un try/except
# propio: una que falle (imagen perdida, o cualquier otro error) no tira
# abajo el resto de la solicitud, y queda marcada con su propio "error" en
# vez de romper todo silenciosamente. Subir esos archivos a un storage
# persistente (Cloudflare R2) es el siguiente paso natural, pendiente.

def _solicitud_del_supervisor(id_solicitud):
    """(solicitud, None) si es del supervisor con sesión, o (None, respuesta de error)."""
    if session.get("rol") != "supervisor":
        return None, (jsonify({"ok": False, "error": "Iniciá sesión como supervisor primero."}), 401)
    solicitud = db.buscar_solicitud(id_solicitud)
    if not solicitud or solicitud["supervisor_id"] != session["usuario_id"]:
        return None, (jsonify({"ok": False, "error": "No se encontró esa solicitud."}), 404)
    return solicitud, None


def _resultado_guardado(nombre_json):
    """El resultado del segmentador de una figura, reconstruido desde su JSON
    (grafica_<uid>.json; el CSV y el overlay comparten el uid). Así la
    autorización genera el STL con la segmentación que el supervisor revisó
    y corrigió, en vez de volver a segmentar la imagen y perder las
    correcciones. None si el archivo ya no está (disco efímero)."""
    nombre = secure_filename(nombre_json or "")
    ruta = os.path.join(RESULTADOS_DIR, nombre)
    if not nombre.startswith("grafica_") or not nombre.endswith(".json") or not os.path.isfile(ruta):
        return None
    try:
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
    except (OSError, ValueError):
        return None
    uid = nombre[len("grafica_"):-len(".json")]
    return {
        "series": datos.get("series") or [],
        "puntos_curva": [],
        "resumen": datos.get("resumen") or {},
        "advertencias": datos.get("advertencias") or [],
        "ruta_json": ruta,
        "nombre_json": nombre,
        "nombre_csv": f"descripcion_{uid}.csv",
        "nombre_overlay": f"overlay_{uid}.png",
    }


def _campos_revision(resultado, pie_figura):
    """Lo que se guarda en una figura tras segmentarla (paso "Revisar" o una
    corrección de la pizarra), antes de generar su STL."""
    return {
        "resumen": resultado["resumen"],
        "advertencias": list(resultado.get("advertencias") or []),
        "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
        "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
        # json_url + correccion_id: lo que la pizarra necesita para abrir y
        # recalcular esta figura (ver /api/corregir y .../corregir más abajo).
        "json_url": f"/api/resultados/{resultado['nombre_json']}",
        "correccion_id": resultado["nombre_json"],
        "descripcion": narracion.describir_grafica(_payload_stl(resultado), pie_figura),
    }


def _campos_figura(resultado, nombre_stl, advertencias, narracion_info):
    """Lo que se guarda en una figura al generar su STL (al autorizar, o al
    corregir una ya autorizada)."""
    return {
        **_campos_revision(resultado, None),
        **_urls_lamina(nombre_stl, narracion_info),
        "advertencias": advertencias,
    }


# Flujo del supervisor (en este orden):
#   1. "Revisar"      -> /revisar: segmenta cada gráfica (sin STL todavía) y
#                        la muestra con su descripción y la vista previa de
#                        la lámina (/api/lamina/vista); se puede corregir en
#                        la pizarra (/figuras/<i>/corregir), tanto la lectura
#                        como lo que se imprime ("ajustes_lamina").
#   2. Visto bueno    -> /figuras/<i>/visto-bueno, o guardar una corrección.
#   3. "Autorizar"    -> /autorizar: solo con todas las gráficas revisadas;
#                        genera el STL y la narración de cada una a partir de
#                        la segmentación revisada.
#   4. "Enviar a imprenta".
# El estado "en revisión" vive en las figuras (correccion_id / "revisada"),
# no en una columna nueva: no hace falta migrar la base (SQLite ni Turso).

@app.route("/api/solicitudes/<int:id_solicitud>/revisar", methods=["POST"])
def revisar_solicitud(id_solicitud):
    solicitud, error = _solicitud_del_supervisor(id_solicitud)
    if error:
        return error
    if solicitud["estado_supervisor"] != "en_espera":
        return jsonify({"ok": False, "error": "Esta solicitud ya fue autorizada."}), 400

    figuras = []
    for figura in solicitud["figuras"]:
        figura = dict(figura)
        # Quitada por el supervisor, o ya segmentada (y quizá corregida): no se toca.
        if figura.get("quitada") or (figura.get("correccion_id") and _resultado_guardado(figura["correccion_id"])):
            figuras.append(figura)
            continue
        ruta_imagen = os.path.join(RESULTADOS_DIR, secure_filename(figura.get("id") or ""))
        if not figura.get("id") or not os.path.isfile(ruta_imagen):
            figura["error"] = (
                "La imagen original ya no está disponible en el servidor (pudo "
                "reiniciarse desde que se subió el PDF). Hay que volver a subir el documento."
            )
            figuras.append(figura)
            continue
        try:
            resultado = segmentar_figura(ruta_imagen)
            figura.pop("error", None)
            figura.update(_campos_revision(resultado, figura.get("pie_figura")))
            figura["revisada"] = False
        except Exception as e:
            print(f"ERROR REVISANDO figura {figura.get('id')} de la solicitud {id_solicitud}:", flush=True)
            print(traceback.format_exc(), flush=True)
            figura["error"] = f"No se pudo segmentar esta gráfica: {e}"
        figuras.append(figura)

    solicitud = db.actualizar_solicitud(id_solicitud, {"figuras": figuras})
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/<int:id_solicitud>/figuras/<int:indice>/visto-bueno", methods=["POST"])
def visto_bueno_figura(id_solicitud, indice):
    solicitud, error = _solicitud_del_supervisor(id_solicitud)
    if error:
        return error
    if solicitud["estado_supervisor"] != "en_espera":
        return jsonify({"ok": False, "error": "Esta solicitud ya fue autorizada."}), 400
    figuras = list(solicitud["figuras"])
    if not 0 <= indice < len(figuras):
        return jsonify({"ok": False, "error": "No se encontró esa figura."}), 404
    if not figuras[indice].get("correccion_id") or figuras[indice].get("error"):
        return jsonify({"ok": False, "error": "Esta gráfica todavía no se pudo revisar."}), 400
    datos = request.get_json(silent=True) or {}
    figuras[indice] = {**figuras[indice], "revisada": bool(datos.get("revisada", True))}
    solicitud = db.actualizar_solicitud(id_solicitud, {"figuras": figuras})
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/<int:id_solicitud>/autorizar", methods=["POST"])
def autorizar_solicitud(id_solicitud):
    solicitud, error = _solicitud_del_supervisor(id_solicitud)
    if error:
        return error
    if solicitud["estado_supervisor"] != "en_espera":
        return jsonify({"ok": False, "error": "Esta solicitud ya fue autorizada."}), 400

    utilizables = [f for f in solicitud["figuras"] if not f.get("error") and not f.get("quitada")]
    if not utilizables:
        return jsonify({"ok": False, "error": "No queda ninguna gráfica para autorizar: agrega o restaura una, o rechaza la solicitud."}), 400
    if not any(f.get("correccion_id") for f in utilizables):
        return jsonify({"ok": False, "error": "Primero pulsa \"Revisar\" para ver las gráficas."}), 400
    pendientes = [f for f in utilizables if not f.get("revisada")]
    if pendientes:
        return jsonify({
            "ok": False,
            "error": f"Falta dar el visto bueno a {len(pendientes)} gráfica(s) antes de autorizar.",
        }), 400

    figuras_actualizadas = []
    algun_exito = False
    for figura in solicitud["figuras"]:
        figura_actualizada = dict(figura)
        if figura.get("error") or figura.get("quitada"):
            figuras_actualizadas.append(figura_actualizada)
            continue
        resultado = _resultado_guardado(figura.get("correccion_id"))
        if resultado is None:
            figura_actualizada["error"] = (
                "Los archivos de esta gráfica ya no están en el servidor (pudo "
                "reiniciarse). Hay que volver a subir el documento."
            )
            figuras_actualizadas.append(figura_actualizada)
            continue
        try:
            nombre_stl, advertencias, narracion_info = crear_stl_desde_resultado(
                resultado, "grafica_tactil", pie_figura=figura.get("pie_figura"),
                ajustes=figura.get("ajustes_lamina"))
            figura_actualizada.update(
                _campos_figura(resultado, nombre_stl, advertencias, narracion_info))
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


# Corrección de UNA figura en la pizarra:
#   - en revisión (sin autorizar): se recalcula la segmentación y la figura
#     queda con visto bueno (el supervisor ya la miró y la arregló);
#   - autorizada y todavía no enviada a imprenta: además se regenera su STL.
# Después de enviada a la imprenta no se puede: la imprenta ya podría estar
# usando el STL anterior.
@app.route("/api/solicitudes/<int:id_solicitud>/figuras/<int:indice>/corregir", methods=["POST"])
def corregir_figura_solicitud(id_solicitud, indice):
    solicitud, error = _solicitud_del_supervisor(id_solicitud)
    if error:
        return error
    en_revision = solicitud["estado_supervisor"] == "en_espera"
    if not en_revision and solicitud["estado_imprenta"] != "no_enviada":
        return jsonify({"ok": False, "error": "Ya fue enviada a la imprenta: no se puede corregir."}), 400

    figuras = solicitud["figuras"]
    if not 0 <= indice < len(figuras):
        return jsonify({"ok": False, "error": "No se encontró esa figura."}), 404
    figura = figuras[indice]

    correcciones = request.get_json(silent=True)
    if not isinstance(correcciones, dict):
        return jsonify({"ok": False, "error": "Se esperaba un JSON con las correcciones."}), 400
    # "ajustes_lamina": lo que el supervisor cambió en la vista previa de la
    # lámina (quitar, mover, cambiar textos). El resto, si viene, corrige la
    # lectura de la gráfica y obliga a volver a calcularla.
    hay_ajustes = "ajustes_lamina" in correcciones
    ajustes = _ajustes_validos(correcciones.pop("ajustes_lamina", None)) if hay_ajustes \
        else figura.get("ajustes_lamina")

    ruta_imagen = os.path.join(RESULTADOS_DIR, secure_filename(figura.get("id") or ""))
    ruta_json = os.path.join(RESULTADOS_DIR, secure_filename(figura.get("correccion_id") or ""))
    if not figura.get("correccion_id") or not os.path.isfile(ruta_imagen) or not os.path.isfile(ruta_json):
        return jsonify({
            "ok": False,
            "error": "Los archivos de esta figura ya no están en el servidor (pudo "
                     "reiniciarse). Hay que volver a subir el documento.",
        }), 404

    try:
        if correcciones:
            resultado = aplicar_correcciones(ruta_imagen, RESULTADOS_DIR, correcciones,
                                             ruta_json_original=ruta_json)
        else:
            # solo cambió la lámina: la lectura guardada sigue valiendo
            resultado = _resultado_guardado(figura["correccion_id"])
            if resultado is None:
                raise ValueError("Los archivos de esta figura ya no están en el servidor.")
        if en_revision:
            campos = {**_campos_revision(resultado, figura.get("pie_figura")), "revisada": True}
        else:
            nombre_stl, advertencias, narracion_info = crear_stl_desde_resultado(
                resultado, "grafica_tactil", pie_figura=figura.get("pie_figura"), ajustes=ajustes)
            campos = _campos_figura(resultado, nombre_stl, advertencias, narracion_info)
        if hay_ajustes:
            campos["ajustes_lamina"] = ajustes
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        print(f"ERROR CORRIGIENDO figura {indice} de la solicitud {id_solicitud}:", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"ok": False, "error": f"No se pudo aplicar la corrección: {e}"}), 500

    figura = dict(figura)
    figura.pop("error", None)
    figura.update(campos)
    figuras = list(figuras)
    figuras[indice] = figura

    cambios = {"figuras": figuras}
    if not en_revision:
        cambios["stl_generado"] = any(f.get("stl_url") for f in figuras)
    solicitud = db.actualizar_solicitud(id_solicitud, cambios)
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


# ====================================================================
# Revisión del supervisor: rechazar, quitar/restaurar, reemplazar y agregar
# gráficas, y descargar el PDF original.
# ====================================================================
LARGO_MAX_MOTIVO = 300


def _figura_en_revision(id_solicitud, indice):
    """(solicitud, figuras, None) si la solicitud es del supervisor, sigue en
    espera y el índice existe; si no, (None, None, respuesta de error)."""
    solicitud, error = _solicitud_del_supervisor(id_solicitud)
    if error:
        return None, None, error
    if solicitud["estado_supervisor"] != "en_espera":
        return None, None, (jsonify({"ok": False, "error": "Solo se pueden cambiar las gráficas de una solicitud en espera."}), 400)
    figuras = list(solicitud["figuras"])
    if indice is not None and not 0 <= indice < len(figuras):
        return None, None, (jsonify({"ok": False, "error": "No se encontró esa gráfica."}), 404)
    return solicitud, figuras, None


def _guardar_imagen_subida(solicitud, sufijo):
    """Guarda como PNG la imagen del campo "imagen" del formulario, junto a
    las demás figuras del lote. Devuelve (nombre, None) o (None, error)."""
    archivo = request.files.get("imagen")
    if archivo is None or archivo.filename == "":
        return None, "Elige una imagen (PNG o JPG)."
    if not extension_valida(archivo.filename):
        return None, "Formato no soportado. Usa PNG, JPG, JPEG, BMP o TIFF."
    datos = np.frombuffer(archivo.read(), dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None
    if imagen is None:
        return None, "No se pudo leer esa imagen. Prueba con otro archivo PNG o JPG."
    lote = secure_filename(solicitud.get("lote_id") or "") or "sup"
    nombre = f"{lote}_{sufijo}_{uuid.uuid4().hex[:8]}.png"
    cv2.imwrite(os.path.join(RESULTADOS_DIR, nombre), imagen)
    return nombre, None


def _segmentar_para_revision(figura):
    """Segmenta una figura nueva (reemplazada o agregada) para que entre a
    la revisión como las demás; si falla, queda con su "error"."""
    try:
        resultado = segmentar_figura(os.path.join(RESULTADOS_DIR, figura["id"]))
        figura.update(_campos_revision(resultado, figura.get("pie_figura")))
        figura["revisada"] = False
    except Exception as e:
        print(f"ERROR SEGMENTANDO figura {figura.get('id')}:", flush=True)
        print(traceback.format_exc(), flush=True)
        figura["error"] = f"No se pudo leer esta gráfica: {e}"
    return figura


_CAMPOS_SEGMENTACION = ("ajustes_lamina", "resumen", "advertencias", "csv_url", "overlay_url", "json_url",
                        "correccion_id", "descripcion", "revisada", "error")


@app.route("/api/solicitudes/<int:id_solicitud>/rechazar", methods=["POST"])
def rechazar_solicitud(id_solicitud):
    solicitud, error = _solicitud_del_supervisor(id_solicitud)
    if error:
        return error
    if solicitud["estado_supervisor"] != "en_espera":
        return jsonify({"ok": False, "error": "Solo se puede rechazar una solicitud en espera."}), 400
    motivo = str((request.get_json(silent=True) or {}).get("motivo") or "").strip()[:LARGO_MAX_MOTIVO]
    solicitud = db.actualizar_solicitud(id_solicitud, {
        "rechazo": motivo,              # "" = rechazada sin motivo
        "rechazado_en": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/<int:id_solicitud>/figuras/<int:indice>/quitar", methods=["POST"])
def quitar_figura(id_solicitud, indice):
    """La gráfica sale de la solicitud (no se imprime), pero se guarda para
    poder restaurarla mientras la solicitud siga en espera."""
    solicitud, figuras, error = _figura_en_revision(id_solicitud, indice)
    if error:
        return error
    figuras[indice] = {**figuras[indice], "quitada": True, "revisada": False}
    solicitud = db.actualizar_solicitud(id_solicitud, {"figuras": figuras})
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/<int:id_solicitud>/figuras/<int:indice>/restaurar", methods=["POST"])
def restaurar_figura(id_solicitud, indice):
    solicitud, figuras, error = _figura_en_revision(id_solicitud, indice)
    if error:
        return error
    figura = dict(figuras[indice])
    figura.pop("quitada", None)
    # Si se quitó antes de "Revisar" y la revisión ya empezó, nunca se leyó:
    # se lee ahora para que entre a la revisión como las demás.
    revision_iniciada = any(f.get("correccion_id") for f in figuras)
    if revision_iniciada and not figura.get("correccion_id") and not figura.get("error"):
        figura = _segmentar_para_revision(figura)
    figuras[indice] = figura
    solicitud = db.actualizar_solicitud(id_solicitud, {"figuras": figuras})
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/<int:id_solicitud>/figuras/<int:indice>/reemplazar", methods=["POST"])
def reemplazar_figura(id_solicitud, indice):
    """Se copió la figura equivocada: el supervisor sube la correcta (por
    ejemplo, un recorte del PDF). Se segmenta de nuevo y vuelve a revisión."""
    solicitud, figuras, error = _figura_en_revision(id_solicitud, indice)
    if error:
        return error
    nombre, problema = _guardar_imagen_subida(solicitud, "reemplazo")
    if problema:
        return jsonify({"ok": False, "error": problema}), 400
    figura = {k: v for k, v in figuras[indice].items() if k not in _CAMPOS_SEGMENTACION}
    figura.update({"id": nombre, "imagen_url": f"/api/resultados/{nombre}", "reemplazada": True})
    figuras[indice] = _segmentar_para_revision(figura)
    solicitud = db.actualizar_solicitud(id_solicitud, {"figuras": figuras})
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/<int:id_solicitud>/figuras/agregar", methods=["POST"])
def agregar_figura(id_solicitud):
    """Una figura que el sistema no detectó: el supervisor la sube, con un
    texto opcional que hace de pie de figura (abre la descripción narrada)."""
    solicitud, figuras, error = _figura_en_revision(id_solicitud, None)
    if error:
        return error
    if len(figuras) >= MAX_FIGURAS_POR_SOLICITUD:
        return jsonify({"ok": False, "error": "La solicitud ya tiene el máximo de gráficas."}), 400
    nombre, problema = _guardar_imagen_subida(solicitud, "agregada")
    if problema:
        return jsonify({"ok": False, "error": problema}), 400
    texto = str(request.form.get("texto") or "").strip()[:LARGO_MAX_CAMPO * 3]
    figura = {
        "id": nombre, "titulo": "Gráfico de línea", "pagina": None,
        "caption": texto or "Imagen agregada por el supervisor.",
        "imagen_url": f"/api/resultados/{nombre}", "pie_figura": texto, "agregada": True,
    }
    figuras.append(_segmentar_para_revision(figura))
    solicitud = db.actualizar_solicitud(id_solicitud, {"figuras": figuras})
    return jsonify({"ok": True, "solicitud": _con_estudiante(solicitud)})


@app.route("/api/solicitudes/<int:id_solicitud>/pdf", methods=["GET"])
def pdf_de_solicitud(id_solicitud):
    """El PDF que subió el estudiante, para compararlo con las gráficas.
    Lo pueden bajar el estudiante, su supervisor y la imprenta asignada."""
    solicitud = db.buscar_solicitud(id_solicitud)
    usuario_id, rol = session.get("usuario_id"), session.get("rol")
    dueno = solicitud and (
        (rol == "estudiante" and solicitud["estudiante_id"] == usuario_id)
        or (rol == "supervisor" and solicitud["supervisor_id"] == usuario_id)
        or (rol == "imprenta" and solicitud.get("imprenta_id") == usuario_id))
    if not dueno:
        return jsonify({"ok": False, "error": "No se encontró esa solicitud."}), 404
    lote = secure_filename(solicitud.get("lote_id") or "")
    candidatos = sorted(glob.glob(os.path.join(UPLOAD_DIR, f"{lote}_*.pdf"))) if lote else []
    if not candidatos:
        return jsonify({
            "ok": False,
            "error": "El PDF ya no está en el servidor (se borra al reiniciarse). "
                     "Pídele al estudiante que lo vuelva a enviar si lo necesitas.",
        }), 404
    return send_from_directory(UPLOAD_DIR, os.path.basename(candidatos[0]), as_attachment=True,
                               download_name=solicitud.get("archivo_nombre") or "documento.pdf",
                               mimetype="application/pdf")


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
    # La interfaz de Puntito: estudiante, supervisor e imprenta.
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


def crear_stl_desde_resultado(resultado, prefijo="grafica", pie_figura=None, ajustes=None):
    """Convierte una segmentación ya hecha en una placa táctil STL, y escribe
    al lado su narración para Hand_Tracking (ver narracion.py).

    La lámina lleva en Braille los números de los ejes, el nombre de cada
    eje, el título del gráfico y los valores que el gráfico original tenía
    anotados junto a la curva. Placa fija de 22 x 22 cm: el gráfico va arriba y la leyenda abajo (la textura
    de cada curva, con 2 o más, y el texto completo de lo que no entraba en
    su lugar y se escribió como "A", "B"...).

    `pie_figura`: el "Figura N. ..." que /api/clasificar encontró junto a la
    figura en el PDF; abre la descripción narrada.

    `ajustes`: lo que el supervisor cambió en la vista previa de la lámina
    (ver generador_stl.aplicar_ajustes); la narración usa los mismos textos.

    Devuelve (nombre_stl, advertencias, narracion), con narracion =
    {"descripcion": texto, "nombre": archivo JSON para Hand_Tracking}.
    Lanza ValueError con un mensaje entendible si la gráfica no da para una
    lámina.
    """
    payload = _payload_stl(resultado)
    if not payload["series"] and not payload["puntos_curva"]:
        raise ValueError(
            "El segmentador no encontró ninguna curva en la imagen: no hay nada "
            "que llevar a la lámina táctil."
        )
    payload, _ = aplicar_ajustes(payload, ajustes)

    advertencias = list(resultado.get("advertencias") or [])
    if not (resultado.get("resumen") or {}).get("listo_para_stl"):
        advertencias.append(
            "La lámina se generó SIN calibración completa de los ejes: la forma de "
            "la curva es correcta, pero la escala y los números en Braille pueden "
            "faltar o no corresponder a los valores reales. Revisar el overlay "
            "antes de imprimir."
        )

    base = f"{prefijo}_{uuid.uuid4().hex[:8]}"
    nombre = f"{base}.stl"
    token = _token_qr(resultado)
    url_qr = _url_qr(token)
    diseno = {}
    generar_modelo_desde_recta(
        payload,
        archivo_salida=os.path.join(STL_DIR, nombre),
        incluir_etiquetas=True,
        diseno=diseno,
        ajustes=ajustes,
        url_qr=url_qr,
    )
    advertencias.extend(diseno.get("avisos") or [])

    descripcion = narracion.describir_grafica(payload, pie_figura, diseno)
    exportacion = narracion.exportar_hand_tracking(payload, diseno, descripcion)
    nombre_narracion = f"{base}_narracion.json"
    with open(os.path.join(STL_DIR, nombre_narracion), "w", encoding="utf-8") as f:
        json.dump(exportacion, f, ensure_ascii=False, indent=2)
    # Lo mismo en CSV: coordenadas de todo lo que hay en la placa + textos a
    # narrar, para Hand_Tracking (lo carga igual que el JSON).
    nombre_csv = f"{base}_handtracking.csv"
    ruta_csv = os.path.join(STL_DIR, nombre_csv)
    narracion.exportar_csv_hand_tracking(exportacion, ruta_csv)
    # La copia que sirve el QR (/q/<token>): se reemplaza de una vez, así
    # quien la esté descargando nunca recibe un archivo a medio escribir.
    temporal = f"{_ruta_csv_qr(token)}.{uuid.uuid4().hex[:6]}.tmp"
    shutil.copyfile(ruta_csv, temporal)
    os.replace(temporal, _ruta_csv_qr(token))
    return nombre, advertencias, {"descripcion": descripcion, "nombre": nombre_narracion,
                                  "csv": nombre_csv, "qr": url_qr}


# ------------------------------------------------------------------
# QR de la lámina
# ------------------------------------------------------------------
# El QR en relieve lleva el enlace con el que el programa de Hand_Tracking
# (que corre en la computadora del estudiante) descarga el CSV de la lámina.
# Ese enlace tiene que ser:
#   - absoluto: el programa no sabe en qué servidor está esta API;
#   - corto: menos módulos en el QR, y cada uno sale más grande en relieve;
#   - el mismo en la vista previa, al autorizar y al corregir después: el
#     supervisor ve en la pizarra exactamente el QR que se va a imprimir.
# Por eso el token sale de la imagen original de la figura (no cambia al
# corregir la lectura; sí al reemplazar la gráfica, que es otra) y /q/<token>
# sirve siempre el último CSV generado para ella.
_RE_TOKEN_QR = re.compile(r"[0-9a-f]{10}")


def _base_publica():
    """Dirección pública del servidor: PUBLIC_BASE_URL si se definió, la que
    Render pone sola (RENDER_EXTERNAL_URL), o la del pedido en curso."""
    base = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    if not base and has_request_context():
        base = request.host_url
    return (base or "http://localhost:5000").rstrip("/")


def _token_qr(resultado):
    """Token del QR de una figura (ver arriba), a partir de su lectura."""
    ruta_imagen = None
    try:
        with open(resultado["ruta_json"], encoding="utf-8") as f:
            ruta_imagen = json.load(f).get("_ruta_imagen_original")
    except (KeyError, TypeError, OSError, ValueError):
        pass
    origen = os.path.basename(ruta_imagen or "") or resultado.get("nombre_json") or ""
    return hashlib.sha1(origen.encode("utf-8")).hexdigest()[:10]


def _url_qr(token):
    return f"{_base_publica()}/q/{token}"


def _ruta_csv_qr(token):
    return os.path.join(STL_DIR, f"qr_{token}.csv")


@app.route("/q/<token>", methods=["GET"])
def csv_por_qr(token):
    """Lo que abre el QR impreso: el CSV de Hand_Tracking más reciente de esa
    lámina. Sin caché, para que una corrección posterior llegue siempre."""
    if not _RE_TOKEN_QR.fullmatch(token) or not os.path.isfile(_ruta_csv_qr(token)):
        return ("Esta lámina todavía no tiene datos publicados (falta que el supervisor "
                "la autorice) o el enlace no es válido.", 404,
                {"Content-Type": "text/plain; charset=utf-8"})
    respuesta = send_from_directory(STL_DIR, f"qr_{token}.csv", mimetype="text/csv",
                                    as_attachment=True, download_name=f"lamina_{token}.csv")
    respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


def _urls_lamina(nombre_stl, narracion_info):
    """Campos de una lámina ya generada que se devuelven al frontend."""
    return {
        "qr_url": narracion_info["qr"],
        "stl_url": f"/api/resultados/stl/{nombre_stl}",
        "stl_download_url": f"/api/resultados/stl/{nombre_stl}/descargar",
        "descripcion": narracion_info["descripcion"],
        "narracion_url": f"/api/resultados/stl/{narracion_info['nombre']}",
        "narracion_download_url": f"/api/resultados/stl/{narracion_info['nombre']}/descargar",
        "handtracking_csv_url": f"/api/resultados/stl/{narracion_info['csv']}",
        "handtracking_csv_download_url": f"/api/resultados/stl/{narracion_info['csv']}/descargar",
    }


def _ruta_extraccion(ruta_imagen):
    return ruta_imagen + ".extraccion.json"


def _guardar_extraccion(ruta_imagen, detectado):
    """Todo lo que /api/clasificar sacó del PDF para esta figura, al lado de
    su imagen: página, pie de figura y el texto real del PDF dentro de la
    figura (el segmentador lo usa en vez del OCR). Así la segmentación, al
    autorizar, tiene la misma información sin volver a abrir el PDF."""
    with open(_ruta_extraccion(ruta_imagen), "w", encoding="utf-8") as f:
        json.dump({
            "pagina": detectado.get("page"),
            "origen": detectado.get("origen"),
            "pie_figura": detectado.get("pie_figura", ""),
            "palabras": detectado.get("palabras") or [],
        }, f, ensure_ascii=False)


def _extraccion(ruta_imagen):
    """Lo guardado por _guardar_extraccion ({} si no hay: imagen suelta)."""
    try:
        with open(_ruta_extraccion(ruta_imagen), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def segmentar_figura(ruta_imagen, palabras=None):
    """procesar_imagen() con el texto del PDF de esa figura, si se extrajo."""
    if palabras is None:
        palabras = _extraccion(ruta_imagen).get("palabras")
    return procesar_imagen(ruta_imagen, RESULTADOS_DIR, palabras=palabras)


def _respuesta_segmentacion(resultado):
    """Campos comunes que toda respuesta del segmentador devuelve al frontend."""
    return {
        # Descripción narrable (sin pie de figura: acá no se sabe de qué PDF
        # vino). Con STL se reemplaza por la completa, ver _urls_lamina.
        "descripcion": narracion.describir_grafica(_payload_stl(resultado)),
        "ok": True,
        "resumen": resultado["resumen"],
        "advertencias": resultado.get("advertencias", []),
        "csv_url": f"/api/resultados/{resultado['nombre_csv']}",
        "csv_download_url": f"/api/resultados/{resultado['nombre_csv']}/descargar",
        "json_url": f"/api/resultados/{resultado['nombre_json']}",
        "overlay_url": f"/api/resultados/{resultado['nombre_overlay']}",
        # La pizarra de la interfaz manda este id a /api/corregir/<id> para
        # recalcular a partir de este resultado (el JSON guarda la imagen original).
        "correccion_id": resultado["nombre_json"],
    }


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
        _guardar_extraccion(ruta_servida, d)
        graficos.append({
            "id": nombre_servido,
            "pagina": d["page"],
            "ancho": round(d["width"]),
            "alto": round(d["height"]),
            "tipo": d["tipo"],
            "confianza": d["confianza"],
            "es_lineal": d["es_lineal"],
            # "Figura N. ..." junto a la figura en el PDF ("" si no hay):
            # el frontend lo guarda en la solicitud y abre la descripción narrada
            "pie_figura": d.get("pie_figura", ""),
            "preview_url": f"/api/resultados/{nombre_servido}",
        })

    return jsonify({"ok": True, "lote_id": lote_id, "graficos": graficos})


# ------------------------------------------------------------------
# Pizarra: la interfaz manda lo que el supervisor corrigió a mano
# (ejes, etiquetas, puntos de la curva) y se recalcula CSV/JSON/overlay.
# El id es el nombre del JSON de un resultado anterior (`correccion_id`).
# ------------------------------------------------------------------
LARGO_MAX_TEXTO_LAMINA = 300


def _ajustes_validos(ajustes):
    """Ajustes de la lámina que manda el navegador, limpios: solo claves de
    elementos conocidos, textos acotados y desplazamientos dentro de la placa."""
    if not isinstance(ajustes, dict):
        return {}
    limpio = {"textos": {}, "ocultar": [], "mover": {}}
    for clave, valor in (ajustes.get("textos") or {}).items() if isinstance(ajustes.get("textos"), dict) else ():
        if _RE_AJUSTE.fullmatch(str(clave)) and isinstance(valor, (str, int, float)):
            limpio["textos"][str(clave)] = str(valor).strip()[:LARGO_MAX_TEXTO_LAMINA]
    for clave in ajustes.get("ocultar") or [] if isinstance(ajustes.get("ocultar"), list) else ():
        if _RE_AJUSTE.fullmatch(str(clave)) and str(clave) not in limpio["ocultar"]:
            limpio["ocultar"].append(str(clave))
    for clave, d in (ajustes.get("mover") or {}).items() if isinstance(ajustes.get("mover"), dict) else ():
        try:
            dx, dy = float(d[0]), float(d[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if _RE_AJUSTE.fullmatch(str(clave)) and math.isfinite(dx) and math.isfinite(dy) and (dx or dy):
            limite = max(PLACA_ANCHO, PLACA_ALTO)
            limpio["mover"][str(clave)] = [max(-limite, min(limite, dx)), max(-limite, min(limite, dy))]
    return limpio


# Vista previa de la lámina ANTES de generar el STL: la misma distribución
# que tendrá la impresión (generador_stl en modo vista previa, sin armar las
# mallas), con los ajustes del supervisor. `correccion_id` es el JSON de la
# lectura: la guardada en la figura o la recién recalculada en la pizarra.
@app.route("/api/lamina/vista", methods=["POST"])
def vista_lamina():
    if session.get("rol") != "supervisor":
        return jsonify({"ok": False, "error": "Iniciá sesión como supervisor primero."}), 401
    datos = request.get_json(silent=True) or {}
    resultado = _resultado_guardado(datos.get("correccion_id"))
    if resultado is None:
        return jsonify({"ok": False, "error": "No se encontró la lectura de esta gráfica (pudo "
                                              "reiniciarse el servidor). Vuelve a revisarla."}), 404
    ajustes = _ajustes_validos(datos.get("ajustes"))
    payload, series_ocultas = aplicar_ajustes(_payload_stl(resultado), ajustes)
    diseno = {}
    # con el mismo QR que llevará el STL: ocupa su lugar y se ve en la pizarra
    url_qr = _url_qr(_token_qr(resultado))
    try:
        generar_modelo_desde_recta(payload, archivo_salida=None, incluir_etiquetas=True,
                                   diseno=diseno, ajustes=ajustes, url_qr=url_qr)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({
        "ok": True,
        "qr_url": url_qr,
        "placa": diseno["placa"],
        "zonas": diseno["zonas"],
        "area": diseno["area"],
        "elementos": diseno["elementos"],
        "ocultos": series_ocultas + diseno["ocultos"],
        "avisos": diseno["avisos"],
        "ajustes": ajustes,
    })


@app.route("/api/corregir/<path:correccion_id>", methods=["POST"])
def corregir(correccion_id):
    correcciones = request.get_json(silent=True)
    if not isinstance(correcciones, dict):
        return jsonify({"ok": False, "error": "Se esperaba un JSON con las correcciones."}), 400

    ruta_json = os.path.join(RESULTADOS_DIR, secure_filename(correccion_id))
    if not os.path.isfile(ruta_json):
        return jsonify({"ok": False, "error": "No se encontró ese resultado. Vuelve a segmentar la gráfica."}), 404

    try:
        with open(ruta_json, encoding="utf-8") as f:
            ruta_imagen = json.load(f).get("_ruta_imagen_original")
    except (OSError, ValueError):
        ruta_imagen = None

    # La ruta viene de un JSON escrito por el segmentador; aun así solo se
    # aceptan imágenes dentro de las carpetas del propio servidor.
    carpetas = (os.path.abspath(RESULTADOS_DIR), os.path.abspath(UPLOAD_DIR))
    if (not ruta_imagen or not os.path.isfile(ruta_imagen)
            or os.path.dirname(os.path.abspath(ruta_imagen)) not in carpetas):
        return jsonify({"ok": False, "error": "No se encontró la imagen original de ese resultado. Vuelve a segmentar la gráfica."}), 404

    try:
        resultado = aplicar_correcciones(ruta_imagen, RESULTADOS_DIR, correcciones,
                                         ruta_json_original=ruta_json)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"Correcciones no válidas: {e}"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Error al recalcular: {e}"}), 500

    return jsonify(_respuesta_segmentacion(resultado))


@app.route("/api/resultados/<path:nombre_archivo>", methods=["GET"])
def servir_resultado(nombre_archivo):
    return send_from_directory(RESULTADOS_DIR, nombre_archivo, as_attachment=False)


@app.route("/api/resultados/<path:nombre_archivo>/descargar", methods=["GET"])
def descargar_resultado(nombre_archivo):
    return send_from_directory(RESULTADOS_DIR, nombre_archivo, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
