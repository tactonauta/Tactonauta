"""db.py
Acceso a datos de Puntito: cuentas de usuario y solicitudes (fase 2 — todavía
sin la generación real de STL, eso queda para la fase siguiente).

Habla SQL parametrizado estándar contra uno de dos backends, elegido por
variables de entorno:

  - Turso (si están definidas TURSO_DATABASE_URL y TURSO_AUTH_TOKEN): base de
    datos compartida y persistente — pensada para producción (Render no da
    disco persistente en su plan gratuito).
  - SQLite local (tactiver/cod/data/puntito.db, vía la librería estándar): para
    desarrollo y pruebas sin necesitar todavía una cuenta de Turso creada.

Turso habla el protocolo de SQLite, así que el esquema y las consultas son
IDÉNTICOS en los dos caminos; lo único que cambia es cómo se abre la conexión
(ver _Conexion más abajo). Pasar de uno a otro en producción es nada más
definir las 2 variables de entorno y agregar `libsql-client` a
requirements.txt — no hace falta tocar el resto de este archivo ni api.py.
"""
import json
import os
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_LOCAL_PATH = os.path.join(DATA_DIR, "puntito.db")

TURSO_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")
USANDO_TURSO = bool(TURSO_URL and TURSO_TOKEN)

_ESQUEMA = [
    """
    CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rol TEXT NOT NULL CHECK(rol IN ('estudiante','supervisor','imprenta')),
        nombre TEXT NOT NULL,
        correo TEXT NOT NULL,
        clave_hash TEXT NOT NULL,
        usuario TEXT,
        codigo TEXT,
        facultad TEXT,
        carrera TEXT,
        entidad TEXT,
        area TEXT,
        supervisor_predeterminado_id INTEGER REFERENCES usuarios(id),
        imprenta_predeterminada_id INTEGER REFERENCES usuarios(id),
        creado_en TEXT NOT NULL,
        UNIQUE(rol, correo)
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_usuarios_usuario
        ON usuarios(rol, usuario) WHERE usuario IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS solicitudes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        estudiante_id INTEGER NOT NULL REFERENCES usuarios(id),
        supervisor_id INTEGER REFERENCES usuarios(id),
        imprenta_id INTEGER REFERENCES usuarios(id),
        lote_id TEXT,
        archivo_nombre TEXT,
        figuras_json TEXT NOT NULL,
        estado_supervisor TEXT NOT NULL DEFAULT 'en_espera'
            CHECK(estado_supervisor IN ('en_espera','aprobada')),
        estado_imprenta TEXT NOT NULL DEFAULT 'no_enviada'
            CHECK(estado_imprenta IN ('no_enviada','pendiente','imprimiendo','impreso')),
        stl_generado INTEGER NOT NULL DEFAULT 0,
        orden TEXT,
        laminas_json TEXT,
        creado_en TEXT NOT NULL
    )
    """,
]


# ============================================================
# Conexión: SQLite local o Turso, misma interfaz hacia afuera
# ============================================================

class _ConexionSQLite:
    """Camino de desarrollo: un archivo SQLite local."""

    def __init__(self, ruta):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        self._con = sqlite3.connect(ruta, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA foreign_keys = ON")

    def ejecutar(self, sql, parametros=()):
        """SELECT: devuelve una lista de dicts."""
        cur = self._con.execute(sql, parametros)
        filas = cur.fetchall()
        return [dict(f) for f in filas]

    def ejecutar_escritura(self, sql, parametros=()):
        """INSERT/UPDATE: hace commit y devuelve el id de la fila insertada
        (o None si no aplica)."""
        cur = self._con.execute(sql, parametros)
        self._con.commit()
        return cur.lastrowid


class _ConexionTurso:
    """Camino de producción: Turso por HTTP puro (protocolo Hrana sobre
    HTTP, endpoint `/v2/pipeline`), con `urllib` de la librería estándar —
    sin ninguna dependencia nueva.

    Se armó así en vez de usar el paquete `libsql-client` porque ese cliente
    usa WebSocket por defecto, y esa conexión falló (probado: la misma URL y
    token funcionan perfecto por HTTP normal, pero la negociación de
    WebSocket se cuelga/falla según la red desde donde se corra). HTTP puro
    es además más simple y no depende de asyncio.
    """

    def __init__(self, url, token):
        # Turso muestra la URL como "libsql://...."; acá se habla por HTTPS.
        if url.startswith("libsql://"):
            url = "https://" + url[len("libsql://"):]
        self._endpoint = url.rstrip("/") + "/v2/pipeline"
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _tipar(valor):
        """Python -> valor tipado del protocolo Hrana."""
        if valor is None:
            return {"type": "null"}
        if isinstance(valor, bool):
            return {"type": "integer", "value": str(int(valor))}
        if isinstance(valor, int):
            return {"type": "integer", "value": str(valor)}
        if isinstance(valor, float):
            return {"type": "float", "value": valor}
        return {"type": "text", "value": str(valor)}

    @staticmethod
    def _destipar(celda):
        """Valor tipado del protocolo Hrana -> Python."""
        tipo = celda.get("type")
        if tipo == "null":
            return None
        if tipo == "integer":
            return int(celda["value"])
        if tipo == "float":
            return float(celda["value"])
        if tipo == "blob":
            return celda.get("base64")
        return celda.get("value")  # "text" y cualquier otro caso

    def _ejecutar_pipeline(self, sql, parametros):
        import urllib.error
        import urllib.request

        cuerpo = json.dumps({
            "requests": [
                {"type": "execute", "stmt": {
                    "sql": sql,
                    "args": [self._tipar(p) for p in parametros],
                }},
                {"type": "close"},
            ]
        }).encode("utf-8")
        peticion = urllib.request.Request(
            self._endpoint, data=cuerpo, headers=self._headers, method="POST"
        )
        try:
            with urllib.request.urlopen(peticion, timeout=20) as resp:
                datos = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detalle = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"Turso respondió HTTP {e.code}: {detalle}")

        primero = datos["results"][0]
        if primero.get("type") == "error":
            raise RuntimeError(primero.get("error", {}).get("message", "Error desconocido de Turso"))
        return primero["response"]["result"]

    def ejecutar(self, sql, parametros=()):
        resultado = self._ejecutar_pipeline(sql, parametros)
        columnas = [c["name"] for c in resultado.get("cols", [])]
        filas = resultado.get("rows", [])
        return [dict(zip(columnas, (self._destipar(c) for c in fila))) for fila in filas]

    def ejecutar_escritura(self, sql, parametros=()):
        resultado = self._ejecutar_pipeline(sql, parametros)
        rowid = resultado.get("last_insert_rowid")
        return int(rowid) if rowid is not None else None


_conexion = None


def _obtener_conexion():
    global _conexion
    if _conexion is None:
        if USANDO_TURSO:
            _conexion = _ConexionTurso(TURSO_URL, TURSO_TOKEN)
        else:
            _conexion = _ConexionSQLite(DB_LOCAL_PATH)
        for sentencia in _ESQUEMA:
            _conexion.ejecutar_escritura(sentencia)
    return _conexion


# ============================================================
# Usuarios
# ============================================================

_CAMPOS_USUARIO = (
    "id", "rol", "nombre", "correo", "usuario", "codigo", "facultad",
    "carrera", "entidad", "area", "supervisor_predeterminado_id",
    "imprenta_predeterminada_id", "creado_en",
)  # sin clave_hash: nunca sale del backend


def _sin_clave(fila):
    if fila is None:
        return None
    return {k: fila.get(k) for k in _CAMPOS_USUARIO}


def buscar_por_correo(rol, correo, incluir_clave=False):
    con = _obtener_conexion()
    filas = con.ejecutar(
        "SELECT * FROM usuarios WHERE rol = ? AND lower(correo) = lower(?)",
        (rol, correo),
    )
    if not filas:
        return None
    return filas[0] if incluir_clave else _sin_clave(filas[0])


def buscar_por_usuario(rol, codigo_usuario):
    """Busca un supervisor/imprenta por su código de conexión (`usuario`)."""
    con = _obtener_conexion()
    filas = con.ejecutar(
        "SELECT * FROM usuarios WHERE rol = ? AND lower(usuario) = lower(?)",
        (rol, codigo_usuario),
    )
    return _sin_clave(filas[0]) if filas else None


def buscar_por_id(id_usuario):
    con = _obtener_conexion()
    filas = con.ejecutar("SELECT * FROM usuarios WHERE id = ?", (id_usuario,))
    return _sin_clave(filas[0]) if filas else None


def generar_codigo_unico(rol):
    """Código corto para que estudiantes se conecten a un supervisor, o
    supervisores a una imprenta."""
    import secrets
    import string

    con = _obtener_conexion()
    alfabeto = string.ascii_uppercase + string.digits
    while True:
        codigo = "".join(secrets.choice(alfabeto) for _ in range(6))
        existe = con.ejecutar(
            "SELECT 1 FROM usuarios WHERE rol = ? AND upper(usuario) = ?",
            (rol, codigo),
        )
        if not existe:
            return codigo


def crear_usuario(rol, datos, clave_hash):
    """`datos` trae los campos propios del rol (nombre, correo, y los
    específicos: código/facultad/carrera para estudiante, entidad para
    supervisor, usuario/area para imprenta). Devuelve el usuario creado (sin
    clave_hash)."""
    con = _obtener_conexion()
    usuario_codigo = datos.get("usuario") or (
        generar_codigo_unico(rol) if rol in ("supervisor",) else None
    )
    id_creado = con.ejecutar_escritura(
        """
        INSERT INTO usuarios (
            rol, nombre, correo, clave_hash, usuario, codigo, facultad,
            carrera, entidad, area, creado_en
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rol, datos.get("nombre"), datos.get("correo"), clave_hash,
            usuario_codigo, datos.get("codigo"), datos.get("facultad"),
            datos.get("carrera"), datos.get("entidad"), datos.get("area"),
            time.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )
    return buscar_por_id(id_creado)


def actualizar_usuario(id_usuario, cambios):
    """`cambios` es un dict {columna: valor}. Solo toca columnas conocidas."""
    con = _obtener_conexion()
    columnas = [c for c in cambios if c in _CAMPOS_USUARIO and c != "id"]
    if not columnas:
        return buscar_por_id(id_usuario)
    set_sql = ", ".join(f"{c} = ?" for c in columnas)
    valores = [cambios[c] for c in columnas] + [id_usuario]
    con.ejecutar_escritura(f"UPDATE usuarios SET {set_sql} WHERE id = ?", valores)
    return buscar_por_id(id_usuario)


def primera_imprenta():
    """La primera imprenta registrada (mismo criterio simple que tenía el
    prototipo: "en cuanto se registre una imprenta, te conectás sola")."""
    con = _obtener_conexion()
    filas = con.ejecutar(
        "SELECT * FROM usuarios WHERE rol = 'imprenta' ORDER BY id ASC LIMIT 1"
    )
    return _sin_clave(filas[0]) if filas else None


# ============================================================
# Solicitudes
# ============================================================
# `figuras` (y, más adelante, `laminas`) se guardan como JSON en una sola
# columna en vez de una tabla aparte: para esta fase alcanza con un snapshot
# de lo que el estudiante eligió, sin necesitar más JOINs. Si en el futuro
# hace falta consultarlas por separado (por página, por tipo...), ahí sí
# conviene una tabla propia.

def _fila_a_solicitud(fila):
    if fila is None:
        return None
    figuras_json = fila.get("figuras_json")
    laminas_json = fila.get("laminas_json")
    return {
        "id": fila.get("id"),
        "estudiante_id": fila.get("estudiante_id"),
        "supervisor_id": fila.get("supervisor_id"),
        "imprenta_id": fila.get("imprenta_id"),
        "lote_id": fila.get("lote_id"),
        "archivo_nombre": fila.get("archivo_nombre"),
        "figuras": json.loads(figuras_json) if figuras_json else [],
        "estado_supervisor": fila.get("estado_supervisor"),
        "estado_imprenta": fila.get("estado_imprenta"),
        "stl_generado": bool(fila.get("stl_generado")),
        "orden": fila.get("orden"),
        "laminas": json.loads(laminas_json) if laminas_json else None,
        "creado_en": fila.get("creado_en"),
    }


def crear_solicitud(estudiante_id, supervisor_id, lote_id, archivo_nombre, figuras):
    con = _obtener_conexion()
    id_creado = con.ejecutar_escritura(
        """
        INSERT INTO solicitudes (
            estudiante_id, supervisor_id, lote_id, archivo_nombre, figuras_json,
            creado_en
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            estudiante_id, supervisor_id, lote_id, archivo_nombre,
            json.dumps(figuras, ensure_ascii=False),
            time.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )
    return buscar_solicitud(id_creado)


def buscar_solicitud(id_solicitud):
    con = _obtener_conexion()
    filas = con.ejecutar("SELECT * FROM solicitudes WHERE id = ?", (id_solicitud,))
    return _fila_a_solicitud(filas[0]) if filas else None


def solicitudes_de_estudiante(estudiante_id):
    con = _obtener_conexion()
    filas = con.ejecutar(
        "SELECT * FROM solicitudes WHERE estudiante_id = ? ORDER BY id DESC",
        (estudiante_id,),
    )
    return [_fila_a_solicitud(f) for f in filas]


def solicitudes_de_supervisor(supervisor_id):
    con = _obtener_conexion()
    filas = con.ejecutar(
        "SELECT * FROM solicitudes WHERE supervisor_id = ? ORDER BY id DESC",
        (supervisor_id,),
    )
    return [_fila_a_solicitud(f) for f in filas]


def solicitudes_de_imprenta(imprenta_id):
    con = _obtener_conexion()
    filas = con.ejecutar(
        "SELECT * FROM solicitudes WHERE imprenta_id = ? ORDER BY id DESC",
        (imprenta_id,),
    )
    return [_fila_a_solicitud(f) for f in filas]


_CAMPOS_SOLICITUD_DIRECTOS = (
    "supervisor_id", "imprenta_id", "estado_supervisor", "estado_imprenta",
    "stl_generado", "orden",
)


def actualizar_solicitud(id_solicitud, cambios):
    """`cambios` puede traer columnas directas (estado_supervisor,
    estado_imprenta, imprenta_id, orden, stl_generado) y/o `figuras`/
    `laminas` como listas de Python — estas últimas se serializan solas a
    JSON antes de guardarse."""
    con = _obtener_conexion()
    columnas, valores = [], []
    for c in _CAMPOS_SOLICITUD_DIRECTOS:
        if c in cambios:
            columnas.append(c)
            valores.append(cambios[c])
    if "figuras" in cambios:
        columnas.append("figuras_json")
        valores.append(json.dumps(cambios["figuras"], ensure_ascii=False))
    if "laminas" in cambios:
        columnas.append("laminas_json")
        valor = cambios["laminas"]
        valores.append(json.dumps(valor, ensure_ascii=False) if valor is not None else None)
    if not columnas:
        return buscar_solicitud(id_solicitud)
    set_sql = ", ".join(f"{c} = ?" for c in columnas)
    valores.append(id_solicitud)
    con.ejecutar_escritura(f"UPDATE solicitudes SET {set_sql} WHERE id = ?", valores)
    return buscar_solicitud(id_solicitud)
