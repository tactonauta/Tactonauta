#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLORADOR DE GRÁFICAS IMPRESAS CON TEXTURA  ·  GUI accesible (tkinter)
=======================================================================

Pensado para personas ciegas o de baja visión. Se ejecuta en local:

    python explorador_graficas_gui.py

Flujo
-----
1) Pestaña "Bienvenida": el usuario elige cómo vincular la gráfica.
     · Opción 1 (F2): pegar un enlace (p. ej. un CSV en Drive). La interfaz
       lo verifica, lo descarga y avisa si los datos son válidos. Si es válido
       se habilita el botón "Empezar", que abre la cámara.
     · Opción 2 (F3): se abre la cámara directamente y se busca el QR impreso
       en la gráfica. El QR contiene el mismo enlace de la opción 1. Al leerlo
       se descarga el archivo y se avisa que la gráfica está cargada.
2) Pestaña "Cámara": exploración con el gesto de pinza (dedo medio + índice
   juntos = audio activo) sobre los puntos y tramos de la gráfica impresa.

Formato del archivo de datos (CSV, coma o punto y coma)
-------------------------------------------------------
    # titulo: Temperatura media mensual
    # magnitud: Temperatura
    # unidad: grados Celsius
    # descripcion: Gráfica de línea con un punto por mes.
    # ancho_referencia: 600        (opcional)
    # alto_referencia: 400         (opcional)
    etiqueta,x,y,valor,info
    Enero,80,350,11,Mes más frío del año
    Febrero,120,330,13,

 · "x" e "y" son coordenadas sobre el lienzo de referencia (como en tu script).
 · "info" (opcional) es texto explicativo que se lee después del valor.
 · Ejes (opcionales): "# eje_x: Meses del año" y "# eje_y: Temperatura (°C)" son
   regiones que se describen al pasar el dedo sobre ellos. Su posición se da con
   "# origen_x", "# origen_y" (cruce de los ejes) y "# fin_x", "# fin_y" (extremos),
   en coordenadas del mismo lienzo; si faltan, se estiman a partir de los puntos.

Láminas de tactiverso: el QR impreso en la placa abre el CSV de Hand_Tracking
que genera el servidor (columnas tipo,id,serie,x1_mm,y1_mm,x2_mm,y2_mm,...; ver
tactiver/cod/narracion.py). Se reconoce solo: la placa es el lienzo de
referencia y cada punto, tramo, eje, texto Braille y entrada de la leyenda se
narra con el texto que ya trae el archivo.

Dependencias
------------
    pip install opencv-python numpy mediapipe pyttsx3
    pip install pygrabber        (opcional, Windows: elegir cámara por nombre)
    pip install pyzbar           (opcional: lectura de QR más robusta)

Teclas (con la ventana en primer plano)
---------------------------------------
    F1  repetir instrucciones          F2  pegar enlace      F3  escanear QR
    F9  activar/desactivar la voz de la interfaz
    F4  leer los botones de la pestaña y decir cuál está seleccionado
    ← → (pestaña Cámara) moverse entre los botones inferiores
    Esc retrocede un nivel (cancela calibración → vuelve al inicio → en la
        bienvenida, dos veces seguidas cierra el programa)
    En la pestaña Cámara:  R repetir descripción · C calibrar hoja
                           A autodetectar hoja
                           + / -  agrandar / reducir el video
"""

import base64
import csv
import hashlib
import io
import json
import math
import os
import queue
import re
import sys
import threading
import time
import subprocess
import traceback
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import tkinter as tk
from tkinter import ttk

import cv2

# ============================================================
# CONFIGURACIÓN
# ============================================================
def _carpeta_base():
    """Junto a este script; si no se puede escribir ahí, en ~/ExploradorGraficas."""
    aqui = os.path.dirname(os.path.abspath(__file__))
    if os.access(aqui, os.W_OK):
        return aqui
    alt = os.path.join(os.path.expanduser("~"), "ExploradorGraficas")
    os.makedirs(alt, exist_ok=True)
    return alt


CARPETA = _carpeta_base()
CARPETA_DATOS = os.path.join(CARPETA, "datos_graficas")
ARCHIVO_CONFIG = os.path.join(CARPETA, "config_exploracion.json")

CONFIG_DEFECTO = {
    "nombre_camara": "WN CAM-K211L",   # nombre tal cual aparece en la app Cámara
    "indice_camara": None,             # número fijo (anula la búsqueda por nombre)
    "voltear": "auto",                 # auto | ninguno | horizontal | 180
    "hoja": [104, 6, 436, 306],        # x, y, ancho, alto de la hoja en el frame
    "velocidad_voz": 200,
    "voz_interfaz": True,              # F9 la activa/desactiva
    "distancia_pinza": 35,             # px entre punta de medio e índice
    "radio_punto": 20,                 # px para "estar sobre" un punto
    "radio_linea": 15,                 # px para "estar sobre" un tramo
    "radio_eje": 20,                   # px para "estar sobre" un eje X/Y
    "vista_fraccion": 0.5,             # ancho del video = fracción del ancho de la pestaña
    "resolucion": [640, 480],          # se pide a la cámara; 'hoja' y los radios dependen de esto
    "mjpg": True,                      # MJPG suele dar 30 fps (YUY2 a veces baja a 5-15 fps)
    "complejidad_mano": 0,             # 0 = modelo ligero (rápido) · 1 = más preciso y más lento
    "depurar_voz": True,               # imprime en la consola cada frase que se dice
}

VISTA_DEFECTO = (800, 600)             # caja inicial del video (ancho, alto) en px
TIMEOUT_DESCARGA = 15                  # segundos
MAX_BYTES = 2_000_000                  # tamaño máximo del archivo de datos


def cargar_config():
    cfg = dict(CONFIG_DEFECTO)
    try:
        with open(ARCHIVO_CONFIG, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    return cfg


def guardar_config(cfg):
    try:
        with open(ARCHIVO_CONFIG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print("Aviso: no se pudo guardar la configuración:", e)


# ============================================================
# UTILIDADES DE TEXTO
# ============================================================
def sin_acentos(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def normalizar_clave(s):
    return sin_acentos(s).strip().lower().replace(" ", "_")


def texto_cv(s):
    """Las fuentes de OpenCV no dibujan acentos ni flechas: versión ASCII."""
    return sin_acentos(s).replace("→", "->").replace("°", " ").encode("ascii", "replace").decode()


def fmt_num(v):
    """Número para voz y pantalla: sin ceros sobrantes y con coma decimal."""
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",")


# ============================================================
# VOZ (hilo propio; reutiliza el esquema de tu script)
# ============================================================
class Voz:
    """Síntesis de voz en segundo plano con pyttsx3.

    - decir(): encola texto sin bloquear.
    - detener(): corta el audio en curso y vacía la cola.
    - Los mensajes de la interfaz (esencial=False) se pueden silenciar con F9
      para quien ya usa un lector de pantalla; la narración de la gráfica y lo
      que se pide explícitamente (F1, F4) es esencial=True y nunca se silencia.
    - Con depurar=True imprime en la consola cada frase que dice: sirve para
      distinguir "el programa no la envió" de "el motor de voz no la sonó".
    """

    def __init__(self, velocidad=200, depurar=True):
        self.cola = queue.Queue()
        self.hablando = threading.Event()
        self._detener = threading.Event()
        self._terminada = threading.Event()
        self.interfaz_activa = True
        self.disponible = False
        self.velocidad = velocidad
        self.depurar = depurar
        try:
            import pyttsx3  # noqa: F401
            self.disponible = True
        except ImportError:
            print("Aviso: falta 'pyttsx3' (pip install pyttsx3). Sin voz.")
        if self.disponible:
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        import pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.velocidad)
            engine.setProperty("volume", 1.0)
            engine.connect("finished-utterance", lambda n, c: self._terminada.set())
        except Exception as e:
            print("Aviso: no se pudo iniciar el motor de voz:", e)
            self.disponible = False
            return

        while True:
            texto = self.cola.get()
            if texto is None:
                break
            self._detener.clear()
            self._terminada.clear()
            try:
                engine.say(texto)
                engine.startLoop(False)
                inicio, t_stop = time.time(), None
                limite = 10 + len(texto) * 0.08         # un texto largo necesita más de 20 s
                while not self._terminada.is_set():
                    if self._detener.is_set():
                        if t_stop is None:
                            engine.stop()
                            t_stop = time.time()
                        elif time.time() - t_stop > 1.5:
                            break                       # el motor no avisó el corte: seguimos
                    engine.iterate()
                    time.sleep(0.01)
                    if time.time() - inicio > limite:   # salvaguarda anti-cuelgue
                        break
                engine.endLoop()
            except Exception as e:
                print("Aviso: error en el motor de audio:", e)
            finally:
                self.hablando.clear()
                self.cola.task_done()

    def decir(self, texto, interrumpir=False, esencial=False):
        if not texto:
            return
        if not esencial and not self.interfaz_activa:
            if self.depurar:
                print("[voz silenciada con F9]", texto[:90])
            return
        if not self.disponible:
            if self.depurar:
                print("[sin motor de voz]", texto[:90])
            return
        if self.depurar:
            print("[voz]", ("(interrumpe) " if interrumpir else "") + texto[:90])
        if interrumpir:
            self.detener()
        self.hablando.set()
        self.cola.put(texto)

    def detener(self):
        while True:
            try:
                self.cola.get_nowait()
                self.cola.task_done()
            except queue.Empty:
                break
        self._detener.set()
        self.hablando.clear()

    def cerrar(self):
        self.detener()
        self.cola.put(None)


# ============================================================
# MODELO DE DATOS DE LA GRÁFICA
# ============================================================
class ErrorDatos(Exception):
    """Mensaje pensado para ser leído en voz alta al usuario."""


class DatosGrafica:
    def __init__(self):
        self.titulo = "Gráfica sin título"
        self.magnitud = "Valor"
        self.unidad = ""
        self.descripcion = ""
        self.ancho_ref = 600.0
        self.alto_ref = 400.0
        self.eje_x = ""             # descripción del eje horizontal
        self.eje_y = ""             # descripción del eje vertical
        self.origen_x = self.origen_y = self.fin_x = self.fin_y = 0.0
        self.puntos = {}            # nombre -> {"x","y","valor","info"} (+ "texto" si viene narrado)
        # Solo con el CSV de una lámina de tactiverso: tramos de curva, ejes,
        # textos Braille, leyenda y QR, cada uno con su texto ya narrado
        # ({"tipo","id","x1","y1","x2","y2","texto","corto"}). Si hay, reemplazan
        # a los tramos y ejes que se arman a partir de los puntos.
        self.regiones = []
        self.origen = ""            # "descarga" | "copia local" | "archivo local"
        self.url = ""
        self.ruta_local = ""

    def valor_texto(self, v):
        return f"{fmt_num(v)} {self.unidad}".strip()

    def texto_eje(self, cual):
        """Descripción hablada del eje 'x' (horizontal) o 'y' (vertical)."""
        if cual == "x":
            nombres = list(self.puntos)
            t = "Eje horizontal." + (f" {self.eje_x}." if self.eje_x else "")
            return f"{t} Va de {nombres[0]} a {nombres[-1]}."
        vals = [p["valor"] for p in self.puntos.values()]
        t = "Eje vertical." + (f" {self.eje_y}." if self.eje_y else f" {self.magnitud}.")
        return f"{t} Los valores van de {self.valor_texto(min(vals))} a {self.valor_texto(max(vals))}."

    def resumen(self):
        """Texto para leer al empezar o al pedir 'repetir descripción'."""
        if self.regiones:           # lámina de tactiverso: su descripción ya lo cuenta todo
            return self.descripcion or f"{self.titulo}."
        partes = [self.titulo + "."]
        if self.descripcion:
            partes.append(self.descripcion)
        partes.append(f"Tiene {len(self.puntos)} puntos.")
        if self.eje_x:
            partes.append(f"Eje horizontal: {self.eje_x}.")
        if self.eje_y:
            partes.append(f"Eje vertical: {self.eje_y}.")
        mayor = max(self.puntos.items(), key=lambda kv: kv[1]["valor"])
        menor = min(self.puntos.items(), key=lambda kv: kv[1]["valor"])
        partes.append(f"El valor más alto es {self.valor_texto(mayor[1]['valor'])} en {mayor[0]}, "
                      f"y el más bajo es {self.valor_texto(menor[1]['valor'])} en {menor[0]}.")
        return " ".join(partes)


_RE_META = re.compile(r"^#\s*([^:=]+?)\s*[:=]\s*(.*)$")


def _num(s):
    return float(s.strip().replace(",", "."))


def _es_csv_tactiverso(texto):
    primera = next((l for l in texto.splitlines() if l.strip()), "")
    encabezados = [normalizar_clave(c) for c in primera.split(",")]
    return "tipo" in encabezados and "x1_mm" in encabezados


def interpretar_csv_tactiverso(texto):
    """El CSV que tactiverso genera junto a cada l\u00e1mina STL (el que descarga
    el QR impreso; ver exportar_csv_hand_tracking en tactiver/cod/narracion.py).
    Coordenadas en mm sobre la placa, con el origen arriba a la izquierda: la
    placa es el lienzo de referencia. Cada punto, tramo de curva, eje, texto
    Braille, entrada de la leyenda y el QR ya traen el texto a narrar."""
    d = DatosGrafica()
    d.titulo = "L\u00e1mina t\u00e1ctil"
    n = 0
    try:
        for n, fila in enumerate(csv.DictReader(io.StringIO(texto)), start=1):
            def num(columna):
                v = (fila.get(columna) or "").strip()
                return _num(v) if v else None

            tipo, ident = (fila.get("tipo") or "").strip(), (fila.get("id") or "").strip()
            narrado = (fila.get("texto") or "").strip()
            if tipo == "placa":
                d.ancho_ref, d.alto_ref = num("x2_mm"), num("y2_mm")
            elif tipo == "descripcion":
                d.descripcion = narrado
            elif tipo == "punto":
                x, y = num("x1_mm"), num("y1_mm")
                if x is None or y is None:
                    raise ValueError
                d.puntos[ident] = {"x": x, "y": y, "valor": num("valor_y"), "info": "",
                                   "texto": narrado}
            elif tipo in ("curva", "eje", "texto", "leyenda"):
                coords = [num(c) for c in ("x1_mm", "y1_mm", "x2_mm", "y2_mm")]
                if None in coords:
                    raise ValueError
                if tipo == "eje":
                    corto = "Eje horizontal" if ident == "eje_x" else "Eje vertical"
                elif tipo == "curva":
                    corto = f"Curva: {(fila.get('tendencia') or '').strip() or 'tramo'}"
                else:
                    corto = narrado.split(":")[0][:40]
                if tipo == "texto" and narrado.startswith("T\u00edtulo:"):
                    d.titulo = narrado[len("T\u00edtulo:"):].strip().rstrip(".") or d.titulo
                d.regiones.append({"tipo": tipo, "id": ident, "texto": narrado, "corto": corto,
                                   **dict(zip(("x1", "y1", "x2", "y2"), coords))})
    except (KeyError, TypeError, ValueError, csv.Error):
        raise ErrorDatos(f"La fila {n} del archivo de la l\u00e1mina no es v\u00e1lida.")

    if not d.ancho_ref or not d.alto_ref or d.ancho_ref <= 0 or d.alto_ref <= 0:
        raise ErrorDatos("El archivo de la l\u00e1mina no dice el tama\u00f1o de la placa.")
    if not d.puntos and not any(r["tipo"] == "curva" for r in d.regiones):
        raise ErrorDatos("El archivo de la l\u00e1mina no tiene ninguna curva para explorar.")
    return d


def interpretar_csv(texto):
    """Valida y convierte el texto del archivo. Lanza ErrorDatos si no sirve."""
    texto = texto.lstrip("\ufeff")
    if _es_csv_tactiverso(texto):
        return interpretar_csv_tactiverso(texto)
    meta, lineas = {}, []
    for linea in texto.splitlines():
        s = linea.strip()
        if not s:
            continue
        if s.startswith("#"):
            m = _RE_META.match(s)
            if m:
                meta[normalizar_clave(m.group(1))] = m.group(2).strip()
            continue
        lineas.append(linea)

    if len(lineas) < 3:
        raise ErrorDatos("El archivo no tiene suficientes filas. Necesito una fila de "
                         "encabezados y al menos dos puntos.")

    delim = ";" if lineas[0].count(";") > lineas[0].count(",") else ","
    filas = list(csv.reader(lineas, delimiter=delim))
    enc = [normalizar_clave(c) for c in filas[0]]

    def col(*alias):
        for a in alias:
            if a in enc:
                return enc.index(a)
        return None

    i_nom = col("etiqueta", "nombre", "punto", "mes", "label", "categoria")
    i_x, i_y = col("x"), col("y")
    i_val = col("valor", "value", "v")
    i_info = col("info", "informacion", "descripcion", "texto", "nota")

    faltan = [n for n, i in (("etiqueta", i_nom), ("x", i_x), ("y", i_y), ("valor", i_val)) if i is None]
    if faltan:
        raise ErrorDatos("Faltan estas columnas en el archivo: " + ", ".join(faltan) +
                         ". Los encabezados encontrados son: " + ", ".join(c for c in filas[0] if c.strip()) + ".")

    d = DatosGrafica()
    for n, fila in enumerate(filas[1:], start=1):
        if not any(c.strip() for c in fila):
            continue
        try:
            nombre = fila[i_nom].strip()
            x, y, valor = _num(fila[i_x]), _num(fila[i_y]), _num(fila[i_val])
        except (IndexError, ValueError):
            raise ErrorDatos(f"La fila de datos número {n} no es válida. "
                             "Revisa que x, y y valor sean números.")
        if not nombre:
            raise ErrorDatos(f"La fila de datos número {n} no tiene etiqueta.")
        if not all(math.isfinite(v) for v in (x, y, valor)):
            raise ErrorDatos(f"La fila de datos número {n} tiene valores no finitos.")
        base, k = nombre, 2
        while nombre in d.puntos:
            nombre, k = f"{base} ({k})", k + 1
        info = fila[i_info].strip() if i_info is not None and i_info < len(fila) else ""
        d.puntos[nombre] = {"x": x, "y": y, "valor": valor, "info": info}
        if len(d.puntos) > 500:
            raise ErrorDatos("El archivo tiene más de 500 puntos; es demasiado grande para explorarlo.")

    if len(d.puntos) < 2:
        raise ErrorDatos("Se necesitan al menos dos puntos para formar una gráfica.")

    d.titulo = meta.get("titulo") or d.titulo
    d.magnitud = meta.get("magnitud") or d.magnitud
    d.unidad = meta.get("unidad", "")
    d.descripcion = meta.get("descripcion", "")

    xs = [p["x"] for p in d.puntos.values()]
    ys = [p["y"] for p in d.puntos.values()]
    try:
        d.ancho_ref = float(meta["ancho_referencia"]) if "ancho_referencia" in meta else max(xs) + min(xs)
        d.alto_ref = float(meta["alto_referencia"]) if "alto_referencia" in meta else max(ys) + min(ys)
    except ValueError:
        raise ErrorDatos("ancho_referencia y alto_referencia deben ser números.")
    if d.ancho_ref <= 0 or d.alto_ref <= 0:
        raise ErrorDatos("El lienzo de referencia debe tener ancho y alto positivos.")

    # Ejes X/Y como regiones descritas. Si no se indican, se estiman alrededor de los puntos.
    d.eje_x, d.eje_y = meta.get("eje_x", ""), meta.get("eje_y", "")
    mx = max(0.08 * (max(xs) - min(xs)), 10.0)
    my = max(0.08 * (max(ys) - min(ys)), 10.0)

    def num_meta(clave, defecto):
        if clave not in meta:
            return defecto
        try:
            return float(meta[clave].replace(",", "."))
        except ValueError:
            raise ErrorDatos(f"{clave} debe ser un número.")

    d.origen_x = num_meta("origen_x", max(0.0, min(xs) - mx))
    d.origen_y = num_meta("origen_y", min(d.alto_ref, max(ys) + my))
    d.fin_x = num_meta("fin_x", min(d.ancho_ref, max(xs) + mx))
    d.fin_y = num_meta("fin_y", max(0.0, min(ys) - my))
    return d


# ============================================================
# ENLACES Y DESCARGA
# ============================================================
def normalizar_url(entrada):
    """Convierte enlaces 'de compartir' (Drive, Sheets, Dropbox, GitHub) en
    enlaces de descarga directa. Lanza ErrorDatos si no parece un enlace."""
    url = entrada.strip().strip("\"'<>")
    if not re.match(r"^https?://", url, re.I):
        raise ErrorDatos("Eso no parece un enlace web. Debe empezar con http o https.")
    u = urllib.parse.urlparse(url)
    host = u.netloc.lower()
    qs = urllib.parse.parse_qs(u.query)

    if host == "drive.google.com":
        m = re.search(r"/file/d/([\w-]+)", u.path)
        fid = m.group(1) if m else (qs.get("id") or [None])[0]
        if fid:
            return f"https://drive.google.com/uc?export=download&id={fid}"
    elif host == "docs.google.com" and "/spreadsheets/d/" in u.path:
        m = re.search(r"/spreadsheets/d/([\w-]+)", u.path)
        gid = (qs.get("gid") or [None])[0]
        if not gid:
            mg = re.search(r"gid=(\d+)", u.fragment)
            gid = mg.group(1) if mg else "0"
        if m:
            return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}"
    elif host in ("www.dropbox.com", "dropbox.com"):
        qs["dl"] = ["1"]
        return urllib.parse.urlunparse(u._replace(query=urllib.parse.urlencode(qs, doseq=True)))
    elif host == "github.com":
        m = re.match(r"^/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", u.path)
        if m:
            return "https://raw.githubusercontent.com/{}/{}/{}/{}".format(*m.groups())
    return url


def _decodificar(datos):
    for codec in ("utf-8-sig", "cp1252"):
        try:
            return datos.decode(codec)
        except UnicodeDecodeError:
            continue
    raise ErrorDatos("No pude leer el texto del archivo.")


def _parece_html(texto):
    return texto.lstrip()[:300].lower().startswith(("<!doctype", "<html", "<head", "<body"))


def cargar_desde_enlace(entrada):
    """Descarga SIEMPRE la versión actual del enlace. Si no hay conexión usa la
    última copia guardada. Devuelve DatosGrafica o lanza ErrorDatos."""
    entrada = entrada.strip().strip("\"'")
    os.makedirs(CARPETA_DATOS, exist_ok=True)

    # Archivo local (útil para pruebas sin internet)
    if os.path.isfile(entrada):
        with open(entrada, "rb") as f:
            datos = interpretar_csv(_decodificar(f.read(MAX_BYTES)))
        datos.origen, datos.ruta_local = "archivo local", entrada
        return datos

    url = normalizar_url(entrada)
    ruta = os.path.join(CARPETA_DATOS, hashlib.sha1(url.encode()).hexdigest()[:12] + ".csv")

    contenido, origen = None, "descarga"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ExploradorGraficas)"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_DESCARGA) as r:
            crudo = r.read(MAX_BYTES + 1)
        if len(crudo) > MAX_BYTES:
            raise ErrorDatos("El archivo es demasiado grande para ser una gráfica.")
        contenido = _decodificar(crudo)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            msg = ("El enlace existe pero no tengo permiso para leerlo. En Drive, comparte el "
                   "archivo con cualquier persona que tenga el enlace.")
        elif e.code == 404:
            msg = "El enlace no existe o fue eliminado."
        else:
            msg = f"El servidor respondió con el error {e.code}."
        if not os.path.isfile(ruta):
            raise ErrorDatos(msg)
    except (urllib.error.URLError, OSError):
        if not os.path.isfile(ruta):
            raise ErrorDatos("No pude descargar el archivo. Revisa tu conexión a internet y el enlace.")

    if contenido is None:                       # sin descarga: copia local previa
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()
        origen = "copia local"

    if _parece_html(contenido):
        raise ErrorDatos("El enlace lleva a una página web y no a un archivo de datos. "
                         "Si es de Drive, comprueba que el archivo sea público y que sea un CSV.")

    datos = interpretar_csv(contenido)          # puede lanzar ErrorDatos
    if origen == "descarga":
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(contenido)
    datos.origen, datos.url, datos.ruta_local = origen, url, ruta
    return datos


def crear_ejemplo():
    """Deja un CSV de ejemplo (los datos de tu script) para probar sin internet."""
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    ruta = os.path.join(CARPETA_DATOS, "ejemplo_temperatura.csv")
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                if "# eje_x" in f.read():
                    return                  # ya es la versión con ejes
        except OSError:
            return
    filas = [("Enero", 80, 350, 11), ("Febrero", 120, 330, 13), ("Marzo", 160, 330, 13),
             ("Abril", 200, 280, 17), ("Mayo", 240, 250, 20), ("Junio", 280, 220, 23),
             ("Julio", 320, 180, 26), ("Agosto", 360, 170, 27), ("Septiembre", 400, 200, 25),
             ("Octubre", 440, 240, 21), ("Noviembre", 480, 290, 17), ("Diciembre", 520, 330, 14)]
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        f.write("# titulo: Temperatura media mensual\n# magnitud: Temperatura\n"
                "# unidad: grados Celsius\n"
                "# descripcion: Gráfica de línea con un punto por mes, de enero a diciembre.\n"
                "# eje_x: Meses del año\n# eje_y: Temperatura media en grados Celsius\n"
                "# origen_x: 50\n# origen_y: 380\n# fin_x: 560\n# fin_y: 140\n"
                "# ancho_referencia: 600\n# alto_referencia: 400\n"
                "etiqueta,x,y,valor,info\n")
        for n, x, y, v in filas:
            f.write(f"{n},{x},{y},{v},\n")


# ============================================================
# GEOMETRÍA (de tu script, parametrizada)
# ============================================================
def distancia(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def distancia_segmento(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return distancia(px, py, x1, y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return distancia(px, py, x1 + t * dx, y1 + t * dy)


def transformacion(ancho_ref, alto_ref, hoja):
    """Devuelve f(x, y) que lleva coordenadas del lienzo de referencia al
    rectángulo de la hoja (un solo factor, sin deformar, centrado).
    'hoja' = (x, y, ancho, alto)."""
    hx, hy, hw, hh = hoja
    escala = min(hw / ancho_ref, hh / alto_ref)
    mx = hx + (hw - ancho_ref * escala) / 2
    my = hy + (hh - alto_ref * escala) / 2
    return lambda x, y: (int(mx + x * escala), int(my + y * escala))


def escalar_puntos(puntos, ancho_ref, alto_ref, hoja):
    f = transformacion(ancho_ref, alto_ref, hoja)
    res = {}
    for n, p in puntos.items():
        x, y = f(p["x"], p["y"])
        res[n] = {"x": x, "y": y, "valor": p["valor"], "info": p.get("info", ""),
                  "texto": p.get("texto", "")}
    return res


def escalar_regiones(regiones, ancho_ref, alto_ref, hoja):
    """Las regiones narradas de una lámina de tactiverso, en píxeles del frame."""
    f = transformacion(ancho_ref, alto_ref, hoja)
    res = []
    for r in regiones:
        (x1, y1), (x2, y2) = f(r["x1"], r["y1"]), f(r["x2"], r["y2"])
        res.append({**r, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "etiqueta": {"eje_x": "X", "eje_y": "Y"}.get(r["id"])})
    return res


def construir_ejes(d, hoja):
    """Los ejes X e Y como segmentos (en píxeles del frame) con su descripción."""
    f = transformacion(d.ancho_ref, d.alto_ref, hoja)
    ox, oy = f(d.origen_x, d.origen_y)
    fx, _ = f(d.fin_x, d.origen_y)
    _, fy = f(d.origen_x, d.fin_y)
    return [
        {"id": "x", "x1": ox, "y1": oy, "x2": fx, "y2": oy,
         "texto": d.texto_eje("x"), "corto": "Eje horizontal", "etiqueta": "X"},
        {"id": "y", "x1": ox, "y1": oy, "x2": ox, "y2": fy,
         "texto": d.texto_eje("y"), "corto": "Eje vertical", "etiqueta": "Y"},
    ]


def construir_segmentos(puntos):
    segs, nombres = [], list(puntos.keys())
    for a, b in zip(nombres, nombres[1:]):
        p1, p2 = puntos[a], puntos[b]
        if p2["valor"] > p1["valor"]:
            tend = "aumento"
        elif p2["valor"] < p1["valor"]:
            tend = "disminución"
        else:
            tend = "estable"
        segs.append({"inicio": a, "fin": b, "x1": p1["x"], "y1": p1["y"], "x2": p2["x"], "y2": p2["y"],
                     "tendencia": tend, "valor_inicial": p1["valor"], "valor_final": p2["valor"]})
    return segs


def detectar_hoja(frame):
    """Intenta hallar la hoja como el cuadrilátero convexo más grande.
    Devuelve (x, y, ancho, alto) o None. Es una ayuda, no una garantía."""
    h, w = frame.shape[:2]
    gris = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for img in (bw, 255 - bw):                  # hoja clara sobre fondo oscuro, o al revés
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contornos, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < 0.15 * h * w or area > 0.97 * h * w:
                continue
            aprox = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(aprox) == 4 and cv2.isContourConvex(aprox):
                return cv2.boundingRect(aprox)
    return None


# ============================================================
# CÁMARA
# ============================================================
def obtener_indice_por_nombre(nombre_buscado):
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return None
    try:
        dispositivos = FilterGraph().get_input_devices()
    except Exception:
        return None
    print("Cámaras detectadas:")
    for i, n in enumerate(dispositivos):
        print(f"  [{i}] {n}")
    for i, n in enumerate(dispositivos):
        if nombre_buscado.lower() in n.lower():
            return i
    return None


# ============================================================
# LECTOR DE QR (robusto a distancia, inclinación y desenfoque leve)
# ============================================================
class LectorQR:
    def __init__(self):
        self.cv = cv2.QRCodeDetector()
        self.aruco = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        try:
            from pyzbar import pyzbar
            self.zbar = pyzbar
        except Exception:
            self.zbar = None

    def _variantes(self, gris):
        """(imagen, factor) con distintos preprocesados: el QR impreso a
        distintas alturas llega con tamaños, contraste y enfoque variables."""
        yield gris, 1.0
        yield self.clahe.apply(gris), 1.0
        yield cv2.addWeighted(gris, 1.8, cv2.GaussianBlur(gris, (0, 0), 3), -0.8, 0), 1.0
        yield cv2.resize(gris, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC), 1.6
        yield cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10), 1.0

    def leer(self, frame):
        """Devuelve (texto|None, polígono|None). Si solo se localiza el QR pero
        no se decodifica, devuelve (None, polígono) para poder guiar al usuario."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        poligono = None
        for img, f in self._variantes(gris):
            if self.zbar is not None:
                try:
                    for o in self.zbar.decode(img, symbols=[self.zbar.ZBarSymbol.QRCODE]):
                        txt = o.data.decode("utf-8", "replace").strip()
                        if txt:
                            pts = [(p.x / f, p.y / f) for p in o.polygon]
                            return txt, (cv2.convexHull(self._arr(pts)) if len(pts) >= 3 else None)
                except Exception:
                    pass
            for det in (self.cv, self.aruco):
                if det is None:
                    continue
                try:
                    txt, pts, _ = det.detectAndDecode(img)
                except cv2.error:
                    continue
                if pts is not None and len(pts):
                    poligono = (pts.reshape(-1, 2) / f)
                if txt:
                    return txt.strip(), (poligono if poligono is not None else None)
        if poligono is None:                    # último recurso: solo localizar
            try:
                ok, pts = self.cv.detect(gris)
                if ok and pts is not None:
                    poligono = pts.reshape(-1, 2)
            except cv2.error:
                pass
        return None, poligono

    @staticmethod
    def _arr(pts):
        import numpy as np
        return np.array(pts, dtype="float32").reshape(-1, 1, 2)


# ============================================================
# EXPLORADOR (gesto de pinza + narración)  — lógica de tu script
# ============================================================
class Explorador:
    def __init__(self, voz, cfg):
        import mediapipe as mp                     # puede lanzar ImportError
        try:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                model_complexity=int(cfg.get("complejidad_mano", 0)))
        except TypeError:                           # mediapipe antiguo sin ese parámetro
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.voz, self.cfg = voz, cfg
        self.datos = None
        self.puntos, self.segmentos, self.ejes = {}, [], []
        self.anterior = None
        self.audio_activado = False
        self.pinza_antes = False
        self.t_mano = time.time()      # última vez que se vio la mano

    def configurar(self, datos, hoja):
        self.datos = datos
        self.puntos = escalar_puntos(datos.puntos, datos.ancho_ref, datos.alto_ref, hoja)
        if datos.regiones:
            # lámina de tactiverso: tramos de curva por un lado (se buscan antes)
            # y ejes, textos, leyenda y QR por el otro, todos con su texto
            regiones = escalar_regiones(datos.regiones, datos.ancho_ref, datos.alto_ref, hoja)
            self.segmentos = [r for r in regiones if r["tipo"] == "curva"]
            self.ejes = [r for r in regiones if r["tipo"] != "curva"]
        else:
            self.segmentos = construir_segmentos(self.puntos)
            self.ejes = construir_ejes(datos, hoja)
        self.anterior = None
        self.t_mano = time.time()

    def cerrar(self):
        try:
            self.hands.close()
        except Exception:
            pass

    def _texto_punto(self, nombre):
        d, p = self.datos, self.puntos[nombre]
        if p.get("texto"):
            return p["texto"]
        t = f"{nombre}. {d.magnitud} de {d.valor_texto(p['valor'])}."
        return t + (" " + p["info"] if p["info"] else "")

    def _texto_tramo(self, s):
        if s.get("texto"):
            return s["texto"]
        d = self.datos
        return (f"De {s['inicio']} a {s['fin']}, tendencia de {s['tendencia']}. "
                f"{d.magnitud} pasa de {fmt_num(s['valor_inicial'])} a {d.valor_texto(s['valor_final'])}.")

    def procesar(self, frame):
        """Dibuja sobre 'frame' y devuelve (frame, texto_visible|None)."""
        h, w = frame.shape[:2]
        res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        actual = None

        if res.multi_hand_landmarks is None:
            self.audio_activado = False             # mano fuera = pinza soltada
        else:
            self.t_mano = time.time()
            for hl in res.multi_hand_landmarks:
                medio, indice = hl.landmark[12], hl.landmark[8]
                mx, my = int(medio.x * w), int(medio.y * h)
                x, y = int(indice.x * w), int(indice.y * h)
                self.audio_activado = distancia(mx, my, x, y) <= self.cfg["distancia_pinza"]

                cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
                cv2.line(frame, (mx, my), (x, y), (255, 0, 0), 2)

                for nombre, p in self.puntos.items():             # 1) ¿sobre un punto?
                    if distancia(x, y, p["x"], p["y"]) <= self.cfg["radio_punto"]:
                        actual = ("punto", nombre)
                        break
                if actual is None:                                 # 2) ¿sobre un tramo?
                    for s in self.segmentos:
                        if distancia_segmento(x, y, s["x1"], s["y1"], s["x2"], s["y2"]) <= self.cfg["radio_linea"]:
                            actual = ("tendencia", s)
                            break
                if actual is None:                                 # 3) ¿sobre un eje?
                    for e in self.ejes:
                        if distancia_segmento(x, y, e["x1"], e["y1"], e["x2"], e["y2"]) <= self.cfg["radio_eje"]:
                            actual = ("eje", e)
                            break

        if actual is None:
            ident = None
        elif actual[0] == "punto":
            ident = ("punto", actual[1])
        elif actual[0] == "eje":
            ident = ("eje", actual[1]["id"])
        elif "id" in actual[1]:
            ident = ("tendencia", actual[1]["id"])
        else:
            ident = ("tendencia", actual[1]["inicio"], actual[1]["fin"])

        # Audio: solo con pinza activa, sin solapar, solo al cambiar de elemento.
        if self.audio_activado and ident != self.anterior and not self.voz.hablando.is_set():
            if actual is not None:
                if actual[0] == "punto":
                    texto = self._texto_punto(actual[1])
                elif actual[0] == "eje":
                    texto = actual[1]["texto"]
                else:
                    texto = self._texto_tramo(actual[1])
                self.voz.decir(texto, esencial=True)
                self.anterior = ident
        if not self.audio_activado:
            if self.pinza_antes:
                self.voz.detener()
            self.anterior = None
        self.pinza_antes = self.audio_activado

        for e in self.ejes:                                       # ejes en cian
            cv2.line(frame, (e["x1"], e["y1"]), (e["x2"], e["y2"]), (255, 255, 0), 2)
            if e.get("etiqueta"):
                cv2.putText(frame, e["etiqueta"], (e["x2"] + 6, e["y2"] + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        for p in self.puntos.values():
            cv2.circle(frame, (p["x"], p["y"]), 6, (0, 0, 255), -1)
        for s in self.segmentos:
            cv2.line(frame, (s["x1"], s["y1"]), (s["x2"], s["y2"]), (0, 255, 0), 2)

        cv2.putText(frame, "AUDIO ACTIVO" if self.audio_activado else "Junta medio + indice",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if self.audio_activado else (255, 255, 255), 2)

        visible = None
        if actual is not None:
            if actual[0] == "eje":
                visible = actual[1]["corto"]
            elif actual[0] == "punto":
                p = self.puntos[actual[1]]
                visible = (p["texto"] if p.get("texto") else
                           f"{actual[1]}: {fmt_num(p['valor'])} {self.datos.unidad}".strip())
            else:
                s = actual[1]
                visible = s.get("corto") or f"{s['inicio']} → {s['fin']}: {s['tendencia']}"
            cv2.putText(frame, texto_cv(visible), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, visible


class Captura(threading.Thread):
    """Lee la cámara sin parar y conserva SOLO el último frame. Así el
    procesamiento (mano, QR) nunca trabaja con imágenes viejas acumuladas en
    el búfer del driver: es la causa principal del retraso al mover la mano."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.frame, self.n, self.ok = cap, None, 0, True
        self.cond = threading.Condition()
        self._fin = threading.Event()

    def run(self):
        while not self._fin.is_set():
            ok, f = self.cap.read()
            with self.cond:
                if ok:
                    self.frame, self.n = f, self.n + 1
                else:
                    self.ok = False
                self.cond.notify_all()
            if not ok:
                break

    def esperar(self, n_visto, timeout=1.0):
        """Bloquea hasta que haya un frame más nuevo que 'n_visto'."""
        with self.cond:
            self.cond.wait_for(lambda: self.n != n_visto or not self.ok, timeout)
            if self.n == n_visto:
                return None, n_visto
            return self.frame, self.n

    def parar(self):
        self._fin.set()


# ============================================================
# HILO DE VISIÓN (cámara + QR + exploración)
# ============================================================
class MotorVision(threading.Thread):
    """Lee la cámara en un hilo propio y le pasa a la GUI:
         · self.cola_frames  → (png_base64, (ancho, alto), escala_vista)
         · cola de eventos   → ("estado"|"error_camara"|"qr"|"elemento"|"hoja", ...)
    Modos: "espera"/"vista" (solo muestra), "qr" (busca QR), "explorar"."""

    def __init__(self, cola_eventos, voz, cfg):
        super().__init__(daemon=True)
        self.cola, self.voz, self.cfg = cola_eventos, voz, cfg
        self.cola_frames = queue.Queue(maxsize=1)
        self._parar = threading.Event()
        self.modo = "vista"
        self.datos = None
        self.hoja = tuple(cfg["hoja"])
        self.vista = VISTA_DEFECTO
        self._reconfigurar = False
        self._autodetectar = False
        self._clics = []
        self.indice = 0
        self.explorador = None
        self.lector = None
        self._n = 0
        self._t_visto = time.time()
        self._t_aviso = 0.0
        self._ultimo_qr, self._t_qr = None, 0.0
        self._ultimo_elem = None
        self.formato = "ppm"                      # "ppm" (rápido) o "png" (alternativa)
        self._qr_frame, self._qr_res = None, None
        self._hilo_qr = None
        self._qr_nuevo = threading.Event()

    # ---- control desde la GUI ----
    def set_modo(self, modo):
        self.modo = modo
        self._t_visto = time.time()

    def set_datos(self, datos):
        self.datos, self._reconfigurar = datos, True

    def set_hoja(self, hoja):
        self.hoja, self._reconfigurar = tuple(int(v) for v in hoja), True

    def set_vista(self, ancho, alto):
        self.vista = (int(ancho), int(alto))

    def set_clics(self, clics):
        self._clics = list(clics)

    def pedir_autodetectar(self):
        self._autodetectar = True

    def parar(self):
        self._parar.set()

    # ---- internos ----
    def _emitir(self, *ev):
        self.cola.put(ev)

    def _abrir(self):
        idx = self.cfg.get("indice_camara")
        if idx is None:
            idx = obtener_indice_por_nombre(self.cfg["nombre_camara"])
            if idx is None:
                self._emitir("estado", "No encontré la cámara configurada; uso la cámara predeterminada.", True)
                idx = 0
        self.indice = idx
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened() and idx != 0:
            cap.release()
            self.indice = 0
            cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            return None
        try:
            if self.cfg.get("mjpg", True):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            ancho, alto = self.cfg.get("resolucion", [640, 480])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # menos cola = menos retraso
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)      # el QR llega a distintas alturas
        except Exception:
            pass
        print("Cámara abierta: %dx%d a %.0f fps" % (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
              cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)))
        return cap

    def _voltear(self, frame):
        modo = self.cfg.get("voltear", "auto")
        if modo == "auto":
            modo = "horizontal" if self.indice == 0 else "180"
        if modo == "horizontal":
            return cv2.flip(frame, 1)
        if modo == "180":
            return cv2.flip(frame, -1)
        return frame

    def _hablar_guia(self, texto, cada=4.0):
        ahora = time.time()
        if ahora - self._t_aviso >= cada and not self.voz.hablando.is_set():
            self._t_aviso = ahora
            self.voz.decir(texto, esencial=True)

    def _enviar_frame(self, frame):
        h, w = frame.shape[:2]
        bw, bh = self.vista                       # el video se ajusta a esta caja (sube o baja)
        escala = max(0.2, min(bw / w, bh / h))
        if abs(escala - 1.0) > 0.02:
            interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (int(w * escala), int(h * escala)), interpolation=interp)
        if self.formato == "ppm":                 # sin compresión: mucho más rápido para Tk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rh, rw = rgb.shape[:2]
            datos = b"P6 %d %d 255\n" % (rw, rh) + rgb.tobytes()
        else:
            ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                return
            datos = base64.b64encode(buf.tobytes())
        paquete = (datos, (w, h), escala, self.formato)
        try:
            self.cola_frames.put_nowait(paquete)
        except queue.Full:
            try:
                self.cola_frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.cola_frames.put_nowait(paquete)
            except queue.Full:
                pass

    def _trabajo_qr(self):
        """Decodifica QR en segundo plano: tarda cientos de ms y no debe frenar el video."""
        while not self._parar.is_set():
            if not self._qr_nuevo.wait(0.2):
                continue
            try:
                texto, poli = self.lector.leer(self._qr_frame)
            except Exception:
                traceback.print_exc()
                texto, poli = None, None
            self._qr_res = (texto, poli, time.time())
            self._qr_nuevo.clear()

    def _paso_qr(self, frame):
        """El QR se lee sobre la imagen SIN espejo (un QR reflejado no decodifica)."""
        if self.lector is None:
            self.lector = LectorQR()
            self._hilo_qr = threading.Thread(target=self._trabajo_qr, daemon=True)
            self._hilo_qr.start()
        h, w = frame.shape[:2]
        if not self._qr_nuevo.is_set():              # trabajador libre: dale el frame actual
            self._qr_frame = frame.copy()
            self._qr_nuevo.set()
        ahora = time.time()
        res = self._qr_res
        texto = poli = None
        if res is not None and ahora - res[2] < 0.8:
            texto, poli = res[0], res[1]
        if texto:
            if poli is not None:
                cv2.polylines(frame, [poli.astype("int32").reshape(-1, 1, 2)], True, (0, 255, 0), 4)
            self._t_visto = ahora
            if texto != self._ultimo_qr or ahora - self._t_qr > 6:
                self._ultimo_qr, self._t_qr = texto, ahora
                self._emitir("qr", texto)
        elif poli is not None:
            cv2.polylines(frame, [poli.astype("int32").reshape(-1, 1, 2)], True, (0, 255, 255), 3)
            self._t_visto = ahora
            fraccion = cv2.contourArea(poli.astype("float32").reshape(-1, 1, 2)) / float(w * h)
            if fraccion < 0.012:
                self._hablar_guia("Veo el código pero se ve muy pequeño. Acerca la gráfica a la cámara.")
            elif fraccion > 0.45:
                self._hablar_guia("El código está muy cerca. Aleja un poco la gráfica.")
            else:
                self._hablar_guia("Código detectado. Mantén la gráfica quieta un momento.")
        elif ahora - self._t_visto > 10:
            self._hablar_guia("No veo el código QR. Muévelo despacio frente a la cámara, "
                              "con el código mirando hacia la cámara.", cada=10)
        cv2.putText(frame, "Buscando codigo QR...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame

    def _paso_explorar(self, frame):
        frame = self._voltear(frame)
        if self.explorador is None:
            self._emitir("estado", "Cargando el seguimiento de la mano. Un momento.", True)
            try:
                self.explorador = Explorador(self.voz, self.cfg)
            except ImportError:
                self._emitir("error_camara", "Falta la biblioteca mediapipe. Instálala con: pip install mediapipe")
                self.modo = "vista"
                return frame
            self._reconfigurar = True
        if self._autodetectar:
            self._autodetectar = False
            rect = detectar_hoja(frame)
            if rect:
                self.set_hoja(rect)
            self._emitir("hoja", rect, frame.shape[1] * frame.shape[0])
        if self._reconfigurar and self.datos is not None:
            self.explorador.configurar(self.datos, self.hoja)
            self._reconfigurar = False
        if self.datos is None:
            return frame

        frame, visible = self.explorador.procesar(frame)
        if time.time() - self.explorador.t_mano > 10:
            self._hablar_guia("No veo tu mano. Colócala frente a la cámara, sobre la gráfica.", cada=15)
        x, y, w, h = self.hoja
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        for c in self._clics:
            cv2.circle(frame, tuple(int(v) for v in c), 5, (0, 255, 255), -1)
        if visible != self._ultimo_elem:
            self._ultimo_elem = visible
            self._emitir("elemento", visible or "")
        return frame

    def run(self):
        cap = self._abrir()
        if cap is None:
            self._emitir("error_camara", "No pude abrir la cámara. Comprueba que esté conectada y que "
                                         "ninguna otra aplicación la esté usando.")
            return
        captura = Captura(cap)
        captura.start()
        n = 0
        try:
            while not self._parar.is_set():
                frame, n = captura.esperar(n)
                if frame is None:
                    if not captura.ok:
                        self._emitir("error_camara", "Se perdió la señal de la cámara.")
                        break
                    continue
                try:
                    if self.modo == "qr":
                        frame = self._paso_qr(frame)
                    elif self.modo == "explorar":
                        frame = self._paso_explorar(frame)
                except Exception:
                    traceback.print_exc()
                    self._emitir("error_camara", "Ocurrió un error al procesar la imagen.")
                    self.modo = "vista"
                self._enviar_frame(frame)
        finally:
            captura.parar()
            captura.join(timeout=1.5)
            if self._hilo_qr is not None:
                self._hilo_qr.join(timeout=2)      # que no quede un decodificador a medias
            cap.release()
            if self.explorador is not None:
                self.explorador.cerrar()


# ============================================================
# INTERFAZ
# ============================================================
BG, FG, ACENTO = "#000000", "#FFFF00", "#00FFFF"
BTN_BG, OK, ERR = "#1E1E1E", "#7CFC00", "#FF7F7F"
F_TITULO = ("Arial", 28, "bold")
F_GRANDE = ("Arial", 20)
F_BOTON = ("Arial", 20, "bold")
F_ESTADO = ("Arial", 18)

INSTRUCCIONES = ("Bienvenido al explorador de gráficas con textura. "
                 "Pulsa F4 en cualquier momento para escuchar los botones disponibles y cuál está seleccionado. "
                 "Pulsa F2 para pegar el enlace de la gráfica, o F3 para escanear el código QR impreso. "
                 "Cuando la gráfica esté cargada, Enter empieza. Escape retrocede. "
                 "F1 repite estas instrucciones.")

AYUDA_CAMARA = ("Teclas de la cámara. "
                "F4: escuchar los botones de la parte inferior y cuál está seleccionado. "
                "Flechas izquierda y derecha: moverte entre esos botones. "
                "Enter: empezar la exploración, cuando esté disponible. "
                "R: repetir la descripción de la gráfica. A: autodetectar la hoja. "
                "C: calibrar la hoja, con ayuda de una persona vidente. "
                "Más y menos: agrandar o reducir el video. "
                "Escape: retroceder; si estás calibrando cancela, y si no, vuelve al inicio y apaga la cámara. "
                "F9: activar o silenciar la voz de la interfaz.")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = cargar_config()
        self.voz = Voz(self.cfg["velocidad_voz"], self.cfg.get("depurar_voz", True))
        self.voz.interfaz_activa = bool(self.cfg["voz_interfaz"])
        self.cola = queue.Queue()
        self.motor = None
        self.datos = None
        self.fase = "inicio"       # inicio | qr | qr_cargando | qr_listo | explorando
        self.calibrando, self._clics = False, []
        self._t_escape = 0.0       # para el doble Escape que cierra el programa
        self._foco_prog = 0.0      # instante del último foco dado por el programa (no se anuncia)
        self._id_foco = None
        self._dim_frame, self._escala, self._foto = (1, 1), 1.0, None
        self._caja_vista = VISTA_DEFECTO

        crear_ejemplo()
        guardar_config(self.cfg)                      # deja el archivo creado desde el primer arranque
        print("Configuración:", ARCHIVO_CONFIG)
        print("Datos descargados:", CARPETA_DATOS)
        self.title("Explorador de gráficas con textura")
        self.configure(bg=BG)
        self.geometry("1000x820")
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self._estilo()
        self._construir()

        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.bind_all("<F1>", lambda e: self._leer_instrucciones())
        self.bind_all("<F2>", lambda e: self._opcion_enlace())
        self.bind_all("<F3>", lambda e: self._opcion_qr())
        self.bind_all("<F9>", lambda e: self._alternar_voz())
        self.bind_all("<Return>", self._enter_global)
        self.bind_all("<F4>", lambda e: self._leer_botones())
        self.bind_all("<Escape>", self._escape)
        self.bind_all("<Key>", self._tecla)
        self.notebook.bind("<<NotebookTabChanged>>", self._pestana_cambiada)

        self.after(30, self._bucle)
        self._dar_foco(self.btn_enlace)            # punto de partida para Tab, flechas y F4
        self.after(700, lambda: self._leer_instrucciones(explicito=False))
        if not self.voz.disponible:
            self._mensaje("Aviso: no está instalado pyttsx3, no habrá voz propia.", "error", hablar=False)

    # ---------- construcción ----------
    def _estilo(self):
        e = ttk.Style(self)
        e.theme_use("clam")
        e.configure("TNotebook", background=BG, borderwidth=0)
        e.configure("TNotebook.Tab", font=F_GRANDE, padding=(24, 12), background="#222222", foreground=FG)
        e.map("TNotebook.Tab", background=[("selected", FG)], foreground=[("selected", "#000000"), ("disabled", "#666666")])
        e.configure("TFrame", background=BG)

    def _boton(self, padre, texto, comando, aviso, **kw):
        b = tk.Button(padre, text=texto, command=comando, font=F_BOTON, bg=BTN_BG, fg=FG,
                      activebackground=FG, activeforeground="#000000", disabledforeground="#777777",
                      highlightthickness=5, highlightbackground=BG, highlightcolor=ACENTO,
                      relief="raised", bd=4, padx=16, pady=12, takefocus=1, **kw)
        b.bind("<Return>", lambda e: (b.invoke(), "break")[1])
        b.bind("<FocusIn>", lambda e: self._anunciar_foco(aviso))
        return b

    def _etiqueta(self, padre, texto="", fuente=F_GRANDE, color=FG, **kw):
        return tk.Label(padre, text=texto, font=fuente, bg=BG, fg=color, wraplength=900, justify="left", **kw)

    def _construir(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.configure(takefocus=0)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # ----- Pestaña 1: Bienvenida -----
        self.tab_inicio = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_inicio, text="1 · Bienvenida")
        t = self.tab_inicio
        self._etiqueta(t, "Explorador de gráficas con textura", F_TITULO, ACENTO).pack(anchor="w", padx=20, pady=(20, 6))
        self._etiqueta(t, "Elige cómo vincular la gráfica que vas a explorar:").pack(anchor="w", padx=20, pady=(0, 14))

        self.btn_enlace = self._boton(
            t, "1 · Pegar enlace de la gráfica   (F2)", self._opcion_enlace,
            "Opción uno. Pegar enlace de la gráfica. Pulsa Enter para escribir o pegar el enlace.")
        self.btn_enlace.pack(fill="x", padx=20, pady=6)

        self.panel_enlace = tk.Frame(t, bg=BG)       # se muestra al elegir la opción 1
        self._etiqueta(self.panel_enlace, "Enlace al archivo de datos (por ejemplo, un CSV en Drive):").pack(anchor="w")
        self.var_enlace = tk.StringVar()
        self._enlace_validado = None
        self.var_enlace.trace_add("write", self._enlace_editado)
        self.entrada = tk.Entry(self.panel_enlace, textvariable=self.var_enlace, font=F_GRANDE,
                                bg="#111111", fg="#FFFFFF", insertbackground="#FFFFFF", insertwidth=4,
                                highlightthickness=5, highlightbackground="#444444", highlightcolor=ACENTO)
        self.entrada.pack(fill="x", pady=8)
        self.entrada.bind("<Return>", lambda e: self._enter_entrada())
        self.entrada.bind("<FocusIn>", lambda e: self._anunciar_foco(
            "Cuadro de texto para el enlace. Pega con Control V y pulsa Enter para verificar."))
        fila = tk.Frame(self.panel_enlace, bg=BG)
        fila.pack(fill="x")
        self.btn_pegar = self._boton(fila, "Pegar del portapapeles", self._pegar_portapapeles,
                                     "Botón pegar del portapapeles. Pega el enlace copiado y lo verifica.")
        self.btn_pegar.pack(side="left", padx=(0, 10), pady=6)
        self.btn_verificar = self._boton(fila, "Verificar y descargar", self._verificar_enlace,
                                         "Botón verificar y descargar el archivo de la gráfica.")
        self.btn_verificar.pack(side="left", pady=6)
        self.btn_empezar = self._boton(self.panel_enlace, "Empezar", self._empezar_desde_enlace,
                                       "Botón Empezar. Pulsa Enter para abrir la cámara y explorar la gráfica.", state="disabled")
        self.btn_empezar.pack(fill="x", pady=(10, 0))

        self.btn_qr = self._boton(
            t, "2 · Escanear código QR del impreso   (F3)", self._opcion_qr,
            "Opción dos. Escanear el código QR del impreso. Pulsa Enter para abrir la cámara.")
        self.btn_qr.pack(fill="x", padx=20, pady=6)

        self.est_inicio = self._etiqueta(t, "", F_ESTADO)
        self.est_inicio.pack(anchor="w", padx=20, pady=16)
        self._etiqueta(t, "F1 repetir instrucciones · F9 voz de la interfaz sí/no", 14, "#AAAAAA").pack(anchor="w", padx=20)
        self._etiqueta(t, f"Archivo de configuración: {ARCHIVO_CONFIG}\nDatos descargados: {CARPETA_DATOS}",
                       12, "#AAAAAA").pack(anchor="w", padx=20, pady=(6, 0))
        self.btn_config = self._boton(t, "Abrir carpeta de configuración", self._abrir_carpeta_config,
                                      "Botón abrir la carpeta donde se guarda el archivo de configuración.")
        self.btn_config.pack(anchor="w", padx=20, pady=8)

        # ----- Pestaña 2: Cámara -----
        self.tab_cam = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cam, text="2 · Cámara", state="disabled")
        c = self.tab_cam
        self.est_cam = self._etiqueta(c, "", F_ESTADO)
        self.est_cam.pack(anchor="w", padx=20, pady=(14, 4))
        self.lbl_elem = self._etiqueta(c, "", F_BOTON, ACENTO)
        self.lbl_elem.pack(anchor="w", padx=20)
        barra = tk.Frame(c, bg=BG)
        barra.pack(side="bottom", fill="x", padx=20, pady=6)
        self.marco_video = tk.Frame(c, bg=BG)                 # el video va centrado aquí
        self.marco_video.pack(fill="both", expand=True, padx=20, pady=6)
        self.marco_video.bind("<Configure>", self._ajustar_vista)
        self.video = tk.Label(self.marco_video, bg="#111111", text="Abriendo cámara…", fg="#AAAAAA", font=F_GRANDE)
        self.video.place(relx=0.5, rely=0.5, anchor="center")
        self.video.bind("<Button-1>", self._clic_video)

        self.btn_comenzar = self._boton(barra, "Empezar exploración", self._comenzar_exploracion,
                                        "Botón empezar exploración. La gráfica está cargada. Pulsa Enter.", state="disabled")
        self.btn_comenzar.pack(side="left", padx=4)
        self.btn_repetir = self._boton(barra, "Repetir descripción (R)", self._repetir_descripcion,
                                       "Botón repetir descripción de la gráfica.")
        self.btn_repetir.pack(side="left", padx=4)
        self.btn_autodet = self._boton(barra, "Autodetectar hoja (A)", self._autodetectar,
                                       "Botón autodetectar la hoja en la imagen.")
        self.btn_autodet.pack(side="left", padx=4)
        self.btn_calibrar = self._boton(barra, "Calibrar (C)", self._alternar_calibracion,
                                        "Botón calibrar la hoja con clics. Requiere ayuda de una persona vidente.")
        self.btn_calibrar.pack(side="left", padx=4)
        self.btn_volver = self._boton(barra, "Volver (Esc)", self._volver_inicio,
                                      "Botón volver al inicio y apagar la cámara.")
        self.btn_volver.pack(side="left", padx=4)

    # ---------- tamaño del video ----------
    def _ajustar_vista(self, _e=None):
        """Caja del video = fracción del ancho de la pestaña, centrada, sin pasar del alto disponible."""
        w, h = self.marco_video.winfo_width(), self.marco_video.winfo_height()
        if w < 50 or h < 50:
            return
        self._caja_vista = (max(160, int(w * self.cfg["vista_fraccion"])), max(120, h - 4))
        if self.motor is not None:
            self.motor.set_vista(*self._caja_vista)

    def _cambiar_tamano_video(self, delta):
        f = round(min(1.0, max(0.3, self.cfg["vista_fraccion"] + delta)), 2)
        if f == self.cfg["vista_fraccion"]:
            self.voz.decir("El video ya está en su tamaño límite.", interrumpir=True)
            return
        self.cfg["vista_fraccion"] = f
        guardar_config(self.cfg)
        self._ajustar_vista()
        self._mensaje(f"Video al {int(f * 100)} por ciento del ancho.")

    def _abrir_carpeta_config(self):
        try:
            if sys.platform.startswith("win"):
                os.startfile(CARPETA)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", CARPETA])
            else:
                subprocess.Popen(["xdg-open", CARPETA])
            self._mensaje("Abrí la carpeta que contiene el archivo config_exploracion.json.")
        except Exception:
            self._mensaje(f"No pude abrir la carpeta. El archivo está en: {ARCHIVO_CONFIG}", "error")

    # ---------- mensajes ----------
    def _mensaje(self, texto, tipo="info", hablar=True, interrumpir=True):
        color = {"info": FG, "ok": OK, "error": ERR}[tipo]
        for et in (self.est_inicio, self.est_cam):
            et.configure(text=texto, fg=color)
        if hablar:
            self.voz.decir(texto, interrumpir=interrumpir)

    def _leer_instrucciones(self, explicito=True):
        texto = INSTRUCCIONES if self.notebook.index("current") == 0 else AYUDA_CAMARA
        self.voz.decir(texto, interrumpir=True, esencial=explicito)

    def _alternar_voz(self):
        self.voz.interfaz_activa = not self.voz.interfaz_activa
        self.cfg["voz_interfaz"] = self.voz.interfaz_activa
        guardar_config(self.cfg)
        estado = "activada" if self.voz.interfaz_activa else "desactivada"
        self._mensaje(f"Voz de la interfaz {estado}.", hablar=False)
        was = self.voz.interfaz_activa
        self.voz.interfaz_activa = True
        self.voz.decir(f"Voz de la interfaz {estado}.", interrumpir=True)
        self.voz.interfaz_activa = was

    # ---------- opción 1: enlace ----------
    def _opcion_enlace(self):
        if self.notebook.index("current") != 0:
            return
        self.panel_enlace.pack(fill="x", padx=20, pady=6, after=self.btn_enlace)
        self._dar_foco(self.entrada)
        self.entrada.icursor("end")
        self.voz.decir("Escribe o pega el enlace de la gráfica. Pega con Control V y pulsa Enter para "
                       "verificarlo. Escape cierra este panel.", interrumpir=True)

    def _dar_foco(self, widget):
        """Foco dado por el programa: no se anuncia (el mensaje que lo acompaña ya lo dice;
        si no, el aviso del botón cortaría ese mensaje a media frase)."""
        self._foco_prog = time.time()
        widget.focus_set()

    def _anunciar_foco(self, aviso):
        """Foco movido por la persona (Tab, flechas): se anuncia el botón."""
        if time.time() - self._foco_prog < 0.5:
            return
        if self._id_foco is not None:
            self.after_cancel(self._id_foco)         # si pulsa Tab varias veces seguidas, solo el último
        self._id_foco = self.after(120, lambda: self.voz.decir(aviso, interrumpir=True))

    def _enlace_editado(self, *_):
        """Si cambia el texto, el enlace validado deja de valer: hay que verificar otra vez."""
        if hasattr(self, "btn_empezar") and self.var_enlace.get().strip() != self._enlace_validado:
            self.btn_empezar.configure(state="disabled")

    def _lista_botones(self):
        """[(botón, nombre hablado, tecla)] de la pestaña actual, en orden de izquierda a derecha."""
        if self.notebook.index("current") == 1:
            return [(self.btn_comenzar, "Empezar exploración", "Enter"),
                    (self.btn_repetir, "Repetir descripción", "R"),
                    (self.btn_autodet, "Autodetectar hoja", "A"),
                    (self.btn_calibrar, "Calibrar hoja", "C"),
                    (self.btn_volver, "Volver al inicio", "Escape")]
        lista = [(self.btn_enlace, "Pegar enlace de la gráfica", "F2")]
        if self.panel_enlace.winfo_ismapped():
            lista += [(self.btn_pegar, "Pegar del portapapeles", None),
                      (self.btn_verificar, "Verificar y descargar", None),
                      (self.btn_empezar, "Empezar", "Enter")]
        lista += [(self.btn_qr, "Escanear código QR", "F3"),
                  (self.btn_config, "Abrir carpeta de configuración", None)]
        return lista

    def _leer_botones(self):
        """F4: dice cada botón, su tecla y si está disponible, y cuál tiene el foco."""
        lista = self._lista_botones()
        foco = self.focus_get()
        partes, actual = [], None
        for b, nombre, tecla in lista:
            if str(b.cget("state")) == "disabled":
                partes.append(f"{nombre}, no disponible por ahora")
            else:
                partes.append(nombre + (f", tecla {tecla}" if tecla else ""))
            if b is foco:
                actual = nombre
        donde = "de la parte inferior" if self.notebook.index("current") == 1 else "de esta pantalla"
        texto = f"Botones {donde}, de izquierda a derecha: " + ". ".join(partes) + ". "
        if actual:
            texto += f"El botón seleccionado ahora es: {actual}."
        else:
            texto += "Ningún botón está seleccionado; usa Tab o las flechas."
        self.voz.decir(texto, interrumpir=True, esencial=True)

    def _mover_foco_botones(self, paso):
        """Flechas: pasa al botón anterior o siguiente que esté disponible."""
        lista = [b for b, _n, _t in self._lista_botones() if str(b.cget("state")) != "disabled"]
        if not lista:
            return
        foco = self.focus_get()
        i = lista.index(foco) if foco in lista else (-1 if paso > 0 else 0)
        lista[(i + paso) % len(lista)].focus_set()      # el aviso sale por <FocusIn>

    def _escape(self, _e=None):
        """Escape = retroceder un nivel, siempre. Al llegar a la pantalla de
        bienvenida sin nada que cerrar, pulsado dos veces cierra el programa."""
        if self.notebook.index("current") == 1:
            self._t_escape = 0.0
            if self.calibrando:                        # primero se cancela lo más reciente
                self._alternar_calibracion()
            else:
                self._volver_inicio()
            return
        if self.panel_enlace.winfo_ismapped():         # cerrar el panel del enlace
            self.panel_enlace.pack_forget()
            self._dar_foco(self.btn_enlace)
            self._t_escape = 0.0
            self._mensaje("Panel del enlace cerrado. Estás en la pantalla de bienvenida.")
            return
        ahora = time.time()
        if ahora - self._t_escape < 5:
            self._cerrar()
            return
        self._t_escape = ahora
        self._mensaje("¿Quieres cerrar el programa? Pulsa Escape otra vez para cerrar. "
                      "Pulsa cualquier otra tecla para cancelar.")

    def _enter_entrada(self):
        """Enter en el cuadro: verifica el enlace; si ya está verificado, empieza."""
        if (self.datos is not None and self._enlace_validado is not None
                and self.var_enlace.get().strip() == self._enlace_validado):
            self._empezar_desde_enlace()
        else:
            self._verificar_enlace()

    def _enter_global(self, _e=None):
        """Enter en cualquier parte: ejecuta 'Empezar' si está disponible.
        (Los botones y el cuadro de texto atienden su propio Enter.)"""
        if isinstance(self.focus_get(), (tk.Button, tk.Entry)):
            return
        if self.notebook.index("current") == 0:
            if str(self.btn_empezar.cget("state")) == "normal":
                self._empezar_desde_enlace()
                return
            self.voz.decir("Todavía no hay una gráfica lista para empezar. "
                           "Pulsa F2 para pegar un enlace o F3 para escanear el código QR.", interrumpir=True)
        else:
            if str(self.btn_comenzar.cget("state")) == "normal":
                self._comenzar_exploracion()
                return
            self.voz.decir("No hay nada que empezar ahora. Pulsa F1 para escuchar las teclas disponibles.",
                           interrumpir=True)

    def _pegar_portapapeles(self):
        try:
            txt = self.clipboard_get().strip()
        except tk.TclError:
            self._mensaje("El portapapeles está vacío. Copia primero el enlace.", "error")
            return
        self.var_enlace.set(txt)
        self._verificar_enlace()

    def _verificar_enlace(self):
        entrada = self.var_enlace.get().strip()
        if not entrada:
            self._mensaje("Todavía no hay ningún enlace. Pégalo en el cuadro de texto.", "error")
            return
        self.btn_empezar.configure(state="disabled")
        self._mensaje("Verificando y descargando el enlace. Un momento.")
        self._lanzar_carga(entrada, "enlace")

    def _lanzar_carga(self, entrada, origen):
        def trabajo():
            try:
                self.cola.put(("datos_ok", cargar_desde_enlace(entrada), origen))
            except ErrorDatos as e:
                self.cola.put(("datos_error", str(e), origen))
            except Exception as e:
                traceback.print_exc()
                self.cola.put(("datos_error", f"Ocurrió un error inesperado: {e}", origen))
        threading.Thread(target=trabajo, daemon=True).start()

    def _empezar_desde_enlace(self):
        if self.datos is None:
            return
        self._abrir_camara("explorar")

    # ---------- opción 2: QR ----------
    def _opcion_qr(self):
        if self.notebook.index("current") != 0:
            return
        self._abrir_camara("qr")

    # ---------- cámara ----------
    def _abrir_camara(self, modo):
        self._detener_camara()
        self.motor = MotorVision(self.cola, self.voz, self.cfg)
        self.motor.set_modo(modo)
        self.motor.set_vista(*self._caja_vista)
        if self.datos is not None:
            self.motor.set_datos(self.datos)
        self.motor.start()
        self.fase = "qr" if modo == "qr" else "explorando"
        self.btn_comenzar.configure(state="disabled")
        self.lbl_elem.configure(text="")
        self.video.configure(image="", text="Abriendo cámara…")
        self.notebook.tab(1, state="normal")
        self.notebook.select(1)
        if modo == "qr":
            self._mensaje("Cámara activada. Muestra a la cámara el código QR impreso en la gráfica. "
                          "Te guiaré con la voz.")
            self._dar_foco(self.btn_volver)
        else:
            self._anunciar_inicio_exploracion()

    def _anunciar_inicio_exploracion(self):
        self._mensaje("Exploración iniciada. Coloca la gráfica frente a la cámara. Junta las puntas del dedo "
                      "medio y el índice sobre un punto, una línea o un eje para escuchar su información; "
                      "sepáralas para silenciar. Pulsa F4 para escuchar los botones, y F1 para las teclas.")
        if self.datos is not None:                  # se dice a continuación, sin cortar lo anterior
            self.voz.decir(self.datos.resumen(), esencial=True)
        self._dar_foco(self.btn_repetir)

    def _detener_camara(self):
        if self.motor is not None:
            m, self.motor = self.motor, None
            m.parar()
            m.join(timeout=1.5)
        self.voz.detener()
        self.fase = "inicio"
        self.calibrando, self._clics = False, []

    def _volver_inicio(self):
        self.notebook.select(0)

    def _pestana_cambiada(self, _e=None):
        if self.notebook.index("current") == 0:
            if self.motor is None and self.fase == "inicio":
                return                               # arranque de la app: nada que apagar
            self._detener_camara()
            self.notebook.tab(1, state="disabled")
            self._dar_foco(self.btn_enlace)
            self._mensaje("Has vuelto al inicio. La cámara está apagada.")
        else:
            self.lbl_elem.configure(text="")

    def _comenzar_exploracion(self):
        if self.motor is None or self.datos is None:
            return
        self.motor.set_datos(self.datos)
        self.motor.set_modo("explorar")
        self.fase = "explorando"
        self.btn_comenzar.configure(state="disabled")
        self._anunciar_inicio_exploracion()

    def _repetir_descripcion(self):
        if self.datos is not None:
            self.voz.decir(self.datos.resumen(), interrumpir=True, esencial=True)
        else:
            self.voz.decir("Todavía no hay una gráfica cargada.", interrumpir=True)

    def _autodetectar(self):
        if self.motor is None or self.fase != "explorando":
            self._mensaje("La autodetección funciona durante la exploración.", "error")
            return
        self._mensaje("Buscando la hoja en la imagen.")
        self.motor.pedir_autodetectar()

    # ---------- calibración manual (con ayuda de una persona vidente) ----------
    def _alternar_calibracion(self):
        if self.motor is None or self.fase != "explorando":
            self._mensaje("La calibración funciona durante la exploración.", "error")
            return
        self.calibrando = not self.calibrando
        self._clics = []
        self.motor.set_clics([])
        if self.calibrando:
            self._mensaje("Calibración. Una persona vidente debe hacer clic en la esquina superior izquierda "
                          "de la hoja en el video, y luego en la esquina inferior derecha.")
        else:
            self._mensaje("Calibración cancelada.")

    def _clic_video(self, e):
        if not self.calibrando or self.motor is None:
            return
        self._clics.append((e.x / self._escala, e.y / self._escala))
        self.motor.set_clics(self._clics)
        if len(self._clics) == 1:
            self.voz.decir("Primera esquina marcada.", interrumpir=True)
        elif len(self._clics) == 2:
            (x1, y1), (x2, y2) = self._clics
            hoja = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            self.calibrando, self._clics = False, []
            if hoja[2] < 20 or hoja[3] < 20:
                self.motor.set_clics([])
                self._mensaje("La zona marcada es demasiado pequeña. Inténtalo de nuevo.", "error")
                return
            self.motor.set_hoja(hoja)
            self.cfg["hoja"] = [int(v) for v in hoja]
            guardar_config(self.cfg)
            self.motor.set_clics([])
            self._mensaje("Hoja calibrada y guardada.", "ok")

    # ---------- teclado ----------
    def _tecla(self, e):
        if (self._t_escape and e.keysym != "Escape"
                and not e.keysym.startswith(("Shift", "Control", "Alt", "Meta", "Caps", "Super"))):
            self._t_escape = 0.0                       # cualquier otra tecla cancela el cierre
            self.voz.decir("Cierre cancelado.", interrumpir=True)
        if self.notebook.index("current") != 1 or isinstance(e.widget, tk.Entry):
            return
        k = e.keysym.lower()
        if k == "r":
            self._repetir_descripcion()
        elif k == "c":
            self._alternar_calibracion()
        elif k == "a":
            self._autodetectar()
        elif k == "left":
            self._mover_foco_botones(-1)
        elif k == "right":
            self._mover_foco_botones(+1)
        elif k in ("plus", "equal", "kp_add"):
            self._cambiar_tamano_video(+0.1)
        elif k in ("minus", "kp_subtract"):
            self._cambiar_tamano_video(-0.1)

    # ---------- bucle de eventos ----------
    def _bucle(self):
        try:
            while True:
                self._evento(self.cola.get_nowait())
        except queue.Empty:
            pass
        except Exception:
            traceback.print_exc()
        if self.motor is not None:
            try:
                datos, dim, escala, formato = self.motor.cola_frames.get_nowait()
                if formato == "ppm":
                    self._foto = tk.PhotoImage(data=datos, format="PPM")
                else:
                    self._foto = tk.PhotoImage(data=datos)
                self._dim_frame, self._escala = dim, escala
                self.video.configure(image=self._foto, text="")
            except queue.Empty:
                pass
            except tk.TclError:
                if self.motor is not None and self.motor.formato == "ppm":
                    print("Aviso: este Tk no acepta PPM; uso PNG (más lento).")
                    self.motor.formato = "png"
        self.after(15, self._bucle)

    def _evento(self, ev):
        tipo = ev[0]
        if tipo == "datos_ok":
            self._datos_ok(ev[1], ev[2])
        elif tipo == "datos_error":
            self._datos_error(ev[1], ev[2])
        elif self.motor is None:
            return                                   # eventos tardíos de una cámara ya cerrada
        elif tipo == "qr":
            self._qr_leido(ev[1])
        elif tipo == "estado":
            self._mensaje(ev[1], hablar=ev[2], interrumpir=False)
        elif tipo == "error_camara":
            self._mensaje(ev[1], "error")
        elif tipo == "elemento":
            self.lbl_elem.configure(text=ev[1])
        elif tipo == "hoja":
            rect = ev[1]
            if rect:
                self.cfg["hoja"] = [int(v) for v in rect]
                guardar_config(self.cfg)
                pct = int(100 * rect[2] * rect[3] / ev[2])
                self._mensaje(f"Hoja detectada, ocupa el {pct} por ciento de la imagen. Si puedes, pide a una "
                              "persona vidente que lo confirme; si no es correcto, usa calibrar.", "ok")
            else:
                self._mensaje("No pude detectar la hoja. Mejora la iluminación o usa calibrar con ayuda.", "error")

    def _qr_leido(self, texto):
        if self.fase != "qr":
            return
        self.fase = "qr_cargando"
        self.motor.set_modo("vista")
        self._mensaje("Código QR detectado. Verificando el enlace y descargando.")
        self._lanzar_carga(texto, "qr")

    def _datos_ok(self, datos, origen):
        if origen == "qr" and self.fase != "qr_cargando":
            return
        self.datos = datos
        aviso = " Sin conexión: usé la última copia guardada." if datos.origen == "copia local" else ""
        resumen = f"{datos.titulo}, con {len(datos.puntos)} puntos.{aviso}"
        if origen == "enlace":
            self._enlace_validado = self.var_enlace.get().strip()
            self.btn_empezar.configure(state="normal")
            self._dar_foco(self.btn_empezar)
            self._mensaje(f"Enlace válido. Gráfica: {resumen} Pulsa Enter para empezar.", "ok")
        else:
            self.fase = "qr_listo"
            self.btn_comenzar.configure(state="normal")
            self._dar_foco(self.btn_comenzar)
            self._mensaje(f"La información de la gráfica está cargada. {resumen} "
                          "Pulsa Enter para empezar la exploración.", "ok")

    def _datos_error(self, msg, origen):
        if origen == "qr":
            if self.fase != "qr_cargando":
                return
            self.fase = "qr"
            if self.motor is not None:
                self.motor.set_modo("qr")
            self._mensaje(f"{msg} Sigo buscando un código QR válido.", "error")
        else:
            self.btn_empezar.configure(state="disabled")
            self._mensaje(f"El enlace no es válido. {msg}", "error")
            self._dar_foco(self.entrada)

    # ---------- cierre ----------
    def _cerrar(self):
        self._detener_camara()
        self.voz.cerrar()
        self.destroy()


def main():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().mainloop()


if __name__ == "__main__":
    main()
