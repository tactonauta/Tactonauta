"""db.py
Acceso a datos de Puntito (cuentas de usuario por ahora; solicitudes quedan
para la próxima fase).

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
    """Camino de producción: Turso, vía el cliente síncrono de libsql-client
    (`create_client_sync` / `ClientSync.execute`) — mismo SQL que el camino
    local, sin necesitar asyncio en el resto de api.py.

    Import diferido: si no hay credenciales de Turso configuradas, nunca se
    intenta importar `libsql_client` (no hace falta tenerlo instalado para
    correr en modo local).
    """

    def __init__(self, url, token):
        import libsql_client  # import diferido: ver docstring de la clase
        self._cliente = libsql_client.create_client_sync(url=url, auth_token=token)

    def ejecutar(self, sql, parametros=()):
        rs = self._cliente.execute(sql, list(parametros))
        return [dict(zip(rs.columns, fila)) for fila in rs.rows]

    def ejecutar_escritura(self, sql, parametros=()):
        rs = self._cliente.execute(sql, list(parametros))
        return getattr(rs, "last_insert_rowid", None)


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
