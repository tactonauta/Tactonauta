"""Copia de los archivos del servidor en Cloudflare R2.

En el plan gratuito de Render el disco se borra cada vez que el servidor se
reinicia: se perdían los PDF subidos, las imágenes de cada figura, sus
lecturas (JSON/CSV/overlay) y los STL, y el supervisor veía "hay que volver a
subir el documento". Con R2 configurado:

  - después de cada pedido que pudo crear archivos, sincronizar() busca los
    nuevos o cambiados en las carpetas vigiladas (uploads/ y resultados/) y
    los sube a R2 en segundo plano (la respuesta no espera la subida);
  - antes de leer un archivo, asegurar(ruta) lo baja de R2 si ya no está en
    el disco.

En R2 cada archivo se guarda con su ruta relativa a esta carpeta
("resultados/stl/grafica_tactil_1a2b3c4d.stl"), así que vuelve al mismo lugar.

Configuración: R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID y
R2_SECRET_ACCESS_KEY (variables de entorno en Render; en una computadora,
también en el archivo .env de esta carpeta, que no se sube a GitHub). Si
falta alguna, este módulo no hace nada y la app funciona como antes, solo
con el disco.

No usa boto3: suma ~50 MB de memoria y el servidor tiene 512 MB en total.
R2 habla el mismo protocolo que Amazon S3, y para subir, bajar y listar
alcanza con firmar cada pedido (AWS Signature V4) con la librería estándar.
"""
import datetime
import hashlib
import hmac
import os
import queue
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_VARIABLES = ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
SUBIDAS_EN_PARALELO = 3
ESPERA_MAX = 30          # s por pedido a R2
_BLOQUE = 1024 * 1024

ACTIVO = False
_base = None             # carpeta de la app: las claves en R2 son relativas a ella
_vigiladas = ()          # carpetas cuyos archivos se copian en R2
_conf = {}
_conocidos = {}          # ruta -> (mtime_ns, tamaño) ya subido (o bajado) así
_candado = threading.Lock()
_cola = queue.Queue()


# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
def _leer_env_local(ruta):
    """Las variables R2_* del archivo .env, sin pisar las ya definidas."""
    try:
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#") or "=" not in linea:
                    continue
                clave, valor = (p.strip() for p in linea.split("=", 1))
                if clave in _VARIABLES and valor and not os.environ.get(clave):
                    os.environ[clave] = valor.strip("\"'")
    except OSError:
        pass


def iniciar(base, carpetas):
    """Activa la copia en R2 si están las credenciales. `carpetas`: las que
    se vigilan (dentro de `base`). Los archivos que ya están en el disco al
    arrancar se dan por conocidos: no se suben de nuevo."""
    global ACTIVO, _base, _vigiladas
    _base = os.path.abspath(base)
    _vigiladas = tuple(os.path.abspath(c) for c in carpetas)
    _leer_env_local(os.path.join(_base, ".env"))
    faltan = [v for v in _VARIABLES if not os.environ.get(v)]
    if faltan:
        print(f"[AVISO] Cloudflare R2 no configurado (falta {', '.join(faltan)}): los archivos "
              "solo quedan en el disco del servidor y se pierden si se reinicia.", flush=True)
        return False
    _conf.update(
        cuenta=os.environ["R2_ACCOUNT_ID"].strip(),
        bucket=os.environ["R2_BUCKET"].strip(),
        clave=os.environ["R2_ACCESS_KEY_ID"].strip(),
        secreto=os.environ["R2_SECRET_ACCESS_KEY"].strip(),
    )
    _conf["host"] = f"{_conf['cuenta']}.r2.cloudflarestorage.com"
    for ruta, firma in _recorrer():
        _conocidos[ruta] = firma
    for _ in range(SUBIDAS_EN_PARALELO):
        threading.Thread(target=_subir_en_cola, daemon=True).start()
    ACTIVO = True
    print(f"[R2] Copia de archivos activa en el bucket '{_conf['bucket']}'.", flush=True)
    return True


# ------------------------------------------------------------------
# Pedidos firmados a R2 (AWS Signature V4)
# ------------------------------------------------------------------
def _hmac(clave, texto):
    return hmac.new(clave, texto.encode("utf-8"), hashlib.sha256).digest()


def _pedido(metodo, clave="", consulta=None, cuerpo=None, largo=0, hash_cuerpo=None):
    """Hace un pedido a R2 y devuelve la respuesta abierta (hay que cerrarla).
    Lanza urllib.error.HTTPError si R2 responde con error."""
    ruta = "/" + urllib.parse.quote(f"{_conf['bucket']}/{clave}" if clave else _conf["bucket"], safe="/-_.~")
    consulta = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(str(v), safe='-_.~')}"
        for k, v in sorted((consulta or {}).items()))
    ahora = datetime.datetime.now(datetime.timezone.utc)
    fecha_hora, fecha = ahora.strftime("%Y%m%dT%H%M%SZ"), ahora.strftime("%Y%m%d")
    hash_cuerpo = hash_cuerpo or hashlib.sha256(b"").hexdigest()
    cabeceras = {"host": _conf["host"], "x-amz-content-sha256": hash_cuerpo, "x-amz-date": fecha_hora}
    firmadas = ";".join(sorted(cabeceras))
    canonico = "\n".join([
        metodo, ruta, consulta,
        "".join(f"{k}:{cabeceras[k]}\n" for k in sorted(cabeceras)),
        firmadas, hash_cuerpo,
    ])
    alcance = f"{fecha}/auto/s3/aws4_request"
    a_firmar = "\n".join(["AWS4-HMAC-SHA256", fecha_hora, alcance,
                          hashlib.sha256(canonico.encode("utf-8")).hexdigest()])
    llave = ("AWS4" + _conf["secreto"]).encode("utf-8")
    for parte in (fecha, "auto", "s3", "aws4_request"):
        llave = _hmac(llave, parte)
    firma = hmac.new(llave, a_firmar.encode("utf-8"), hashlib.sha256).hexdigest()
    cabeceras["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={_conf['clave']}/{alcance}, "
                                  f"SignedHeaders={firmadas}, Signature={firma}")
    if cuerpo is not None:
        cabeceras["Content-Length"] = str(largo)
    url = f"https://{_conf['host']}{ruta}" + (f"?{consulta}" if consulta else "")
    req = urllib.request.Request(url, data=cuerpo, method=metodo, headers=cabeceras)
    return urllib.request.urlopen(req, timeout=ESPERA_MAX)


def _clave(ruta):
    """Clave en R2 de un archivo local, o None si no está en una carpeta vigilada."""
    ruta = os.path.abspath(ruta)
    if not any(ruta.startswith(c + os.sep) for c in _vigiladas):
        return None
    return os.path.relpath(ruta, _base).replace(os.sep, "/")


def _firma_local(ruta):
    st = os.stat(ruta)
    return st.st_mtime_ns, st.st_size


def subir(ruta):
    """Sube un archivo local a R2 (lo reemplaza si ya estaba)."""
    clave = _clave(ruta)
    if not ACTIVO or clave is None:
        return
    firma = _firma_local(ruta)
    sha = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(_BLOQUE), b""):
            sha.update(bloque)
        f.seek(0)
        _pedido("PUT", clave, cuerpo=f, largo=firma[1], hash_cuerpo=sha.hexdigest()).close()
    with _candado:
        _conocidos[ruta] = firma


def asegurar(ruta):
    """True si el archivo está en el disco, bajándolo de R2 si hace falta
    (por ejemplo, después de un reinicio del servidor)."""
    if os.path.isfile(ruta):
        return True
    clave = _clave(ruta) if ACTIVO else None
    if clave is None:
        return False
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    temporal = None
    try:
        with _pedido("GET", clave) as respuesta:
            fd, temporal = tempfile.mkstemp(dir=os.path.dirname(ruta), suffix=".r2tmp")
            with os.fdopen(fd, "wb") as f:
                for bloque in iter(lambda: respuesta.read(_BLOQUE), b""):
                    f.write(bloque)
        os.replace(temporal, ruta)
        temporal = None
        with _candado:
            _conocidos[os.path.abspath(ruta)] = _firma_local(ruta)
        return True
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"[R2] No se pudo bajar {clave}: HTTP {e.code}", flush=True)
        return False
    except Exception as e:
        print(f"[R2] No se pudo bajar {clave}: {e}", flush=True)
        return False
    finally:
        if temporal and os.path.exists(temporal):
            os.remove(temporal)


def buscar(prefijo_ruta):
    """Rutas locales de los archivos en R2 cuya ruta empieza con
    `prefijo_ruta` (estén o no en el disco). [] sin R2."""
    prefijo = _clave(prefijo_ruta) if ACTIVO else None
    if prefijo is None:
        return []
    rutas, seguir = [], None
    try:
        while True:
            consulta = {"list-type": "2", "prefix": prefijo}
            if seguir:
                consulta["continuation-token"] = seguir
            with _pedido("GET", consulta=consulta) as respuesta:
                raiz = ET.fromstring(respuesta.read())
            ns = {"s3": raiz.tag.split("}")[0].strip("{")} if raiz.tag.startswith("{") else {}
            p = "s3:" if ns else ""
            rutas += [os.path.join(_base, *e.text.split("/")) for e in raiz.iterfind(f"{p}Contents/{p}Key", ns)]
            seguir = raiz.findtext(f"{p}NextContinuationToken", namespaces=ns)
            if not seguir:
                return rutas
    except Exception as e:
        print(f"[R2] No se pudo listar {prefijo}: {e}", flush=True)
        return rutas


# ------------------------------------------------------------------
# Sincronización en segundo plano
# ------------------------------------------------------------------
def _recorrer():
    for carpeta in _vigiladas:
        for raiz, _, archivos in os.walk(carpeta):
            for nombre in archivos:
                if nombre.endswith(".r2tmp"):
                    continue
                ruta = os.path.join(raiz, nombre)
                try:
                    yield ruta, _firma_local(ruta)
                except OSError:
                    pass


def sincronizar():
    """Pone en cola de subida los archivos nuevos o cambiados desde la
    última vez. Rápido: solo mira fechas y tamaños; la subida va aparte."""
    if not ACTIVO:
        return
    with _candado:
        for ruta, firma in _recorrer():
            if _conocidos.get(ruta) != firma:
                _conocidos[ruta] = firma   # en cola: no se vuelve a encolar
                _cola.put(ruta)


def _subir_en_cola():
    while True:
        ruta = _cola.get()
        try:
            if os.path.isfile(ruta):
                subir(ruta)
        except Exception as e:
            print(f"[R2] No se pudo subir {os.path.basename(ruta)}: {e}", flush=True)
            with _candado:
                _conocidos.pop(ruta, None)   # se reintenta en la próxima sincronización
        finally:
            _cola.task_done()


def esperar_subidas():
    """Espera a que termine todo lo que está en cola (para pruebas)."""
    _cola.join()
