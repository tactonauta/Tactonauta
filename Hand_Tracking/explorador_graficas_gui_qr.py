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

Formato del archivo de datos: CSV "handtracking" (coma o punto y coma)
--------------------------------------------------------------------
Una fila por elemento de la lámina táctil; coordenadas en milímetros sobre la placa:

    tipo,id,serie,x1_mm,y1_mm,x2_mm,y2_mm,valor_x,valor_y,tendencia,texto
    placa,placa,,0,0,220.0,220.0,,,,"esquina superior izquierda, y hacia abajo (la esquina recortada es la superior derecha); esquina recortada de 10.0 mm; gráfico arriba y leyenda abajo"
    descripcion,descripcion,,,,,,,,,"Gráfica 1: Temperatura media mensual. Gráfica de líneas con 1 curva. ..."
    serie,serie_1,Temperatura media,,,,,,,,Línea continua
    punto,s1_p1,Temperatura media,43.9,109.9,,,0.0,11,,Temperatura media. Enero: 11 grados.
    curva,s1_p1-p2,Temperatura media,43.9,109.9,58.3,102.0,,,aumento,"Temperatura media. De Enero a Febrero, sube de 11 a 13."
    texto,texto_1,,38.8,131.1,48.9,131.1,,,,"Categoría, abreviado con la letra A: Enero."
    leyenda,leyenda_A,,10.0,166.8,51.1,173.1,,,,Leyenda: A es Enero.
    eje,eje_x,,35.8,121.5,210.0,121.5,,,,"Eje horizontal: Mes, de Enero a Diciembre."

 · placa: tamaño de la lámina (x1,y1 → x2,y2). Su texto dice en qué esquina está el
   origen de las coordenadas, qué esquina está recortada (se anuncia para orientar la
   placa y se comprueba al autodetectar) y cómo se reparte el contenido.
 · descripcion: se lee al empezar y con R. Su primera frase da el título.
 · serie: nombre de cada curva y su textura.
 · punto (x1,y1) · curva y eje (segmento x1,y1 → x2,y2) · texto y leyenda (segmento
   si es una línea, recuadro si tiene alto y ancho). Con la pinza encima se lee la
   columna "texto". Si hay varios cerca: punto > curva > resto (el más cercano).
 · La hoja calibrada (o autodetectada) son las esquinas de la placa.
 · Sigue funcionando el formato anterior (# titulo..., etiqueta,x,y,valor,info).

Dependencias  (ver requirements.txt)
------------------------------------
    pip install -r requirements.txt
      · mediapipe actual (0.10.30 o posterior, incluida la serie 1.x). El seguimiento
        de la mano usa la API "Tasks" (HandLandmarker), que es la vigente: las
        versiones recientes ya no traen mp.solutions.hands.
      · El modelo de la mano (hand_landmarker.task, unos 8 MB) se descarga solo la
        primera vez y queda en la carpeta "modelos", junto a este script. Sin
        internet, se puede copiar a mano ahí (o indicar su ruta en "modelo_mano"
        de config_exploracion.json).
      · Si se instala un mediapipe antiguo que sí trae mp.solutions, también funciona.
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
import http.client
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
CARPETA_MODELOS = os.path.join(CARPETA, "modelos")
ARCHIVO_CONFIG = os.path.join(CARPETA, "config_exploracion.json")

# Modelo oficial de MediaPipe para el seguimiento de la mano (API Tasks).
URL_MODELO_MANO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/latest/hand_landmarker.task")
NOMBRE_MODELO_MANO = "hand_landmarker.task"

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
    "radio_texto": 12,                 # px para "estar sobre" un texto o recuadro de leyenda
    "vista_fraccion": 0.5,             # ancho del video = fracción del ancho de la pestaña
    "resolucion": [640, 480],          # se pide a la cámara; 'hoja' y los radios dependen de esto
    "mjpg": True,                      # MJPG suele dar 30 fps (YUY2 a veces baja a 5-15 fps)
    "modelo_mano": "",                 # ruta a hand_landmarker.task; vacío = carpeta "modelos" (se descarga solo)
    "confianza_mano": 0.5,             # umbral de detección/seguimiento de la mano (0 a 1)
    "complejidad_mano": 0,             # solo con mediapipe antiguo (mp.solutions): 0 rápido · 1 preciso
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
    s = sin_acentos(s).replace("→", "->").replace("°", " ").replace("«", '"').replace("»", '"')
    return s.encode("ascii", "replace").decode()


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


# Cuando el dedo está cerca de varios elementos gana el de menor prioridad;
# a igual prioridad, el más cercano. Así un punto se oye antes que la curva
# que pasa por él, y los ejes, textos y leyendas compiten solo por distancia.
PRIORIDAD = {"punto": 0, "curva": 1, "tramo": 1, "barra": 1}
PRIORIDAD_RESTO = 2
TIPOS_SEGMENTO = ("curva", "tramo", "eje", "linea")
MAX_ELEMENTOS = 3000

ESQUINAS_HABLADAS = {
    "superior izquierda": "arriba a la izquierda",
    "superior derecha": "arriba a la derecha",
    "inferior izquierda": "abajo a la izquierda",
    "inferior derecha": "abajo a la derecha",
}


class DatosGrafica:
    """Gráfica lista para explorar.

    'elementos' es una lista de diccionarios con:
        id, tipo, serie, forma ("punto" | "segmento" | "rect"),
        x1, y1, x2, y2 (en el lienzo de referencia, origen arriba a la izquierda),
        texto (lo que se lee), corto (lo que se muestra), tendencia.
    En el formato handtracking el lienzo es la placa, en milímetros."""

    def __init__(self):
        self.formato = "handtracking"     # "handtracking" | "clasico"
        self.titulo = "Gráfica sin título"
        self.descripcion = ""
        self.ancho_ref = 220.0
        self.alto_ref = 220.0
        self.estirar = True               # la hoja calibrada ES la placa: escala x e y por separado
        self.esquina_recortada = ""       # "superior derecha", ... (sirve para orientar la placa)
        self.recorte_mm = 0.0
        self.disposicion = ""             # p. ej. "gráfico arriba y leyenda abajo"
        self.series = {}                  # nombre -> textura
        self.elementos = []
        self.resumen_fijo = ""            # formato clásico: resumen ya calculado
        self.origen = ""                  # "descarga" | "copia local" | "archivo local"
        self.url = ""
        self.ruta_local = ""

    def contar(self, tipo):
        return sum(1 for e in self.elementos if e["tipo"] == tipo)

    def resumen_corto(self):
        """Para el aviso de 'enlace válido' / 'gráfica cargada'."""
        n = self.contar("punto")
        if len(self.series) > 1:
            return f"{self.titulo}, con {len(self.series)} series y {n} puntos"
        return f"{self.titulo}, con {n} puntos"

    def orientacion(self):
        """Cómo colocar la placa: tamaño, esquina recortada y disposición."""
        if self.formato != "handtracking":
            return ""
        t = f"Placa de {fmt_num(self.ancho_ref)} por {fmt_num(self.alto_ref)} milímetros."
        if self.esquina_recortada:
            t += f" La esquina recortada va {ESQUINAS_HABLADAS[self.esquina_recortada]}."
        if self.disposicion:
            t += f" {self.disposicion[0].upper()}{self.disposicion[1:]}."
        return t

    def resumen(self):
        """Texto para leer al empezar o al pedir 'repetir descripción'."""
        if self.resumen_fijo:
            return self.resumen_fijo
        partes = [self.orientacion()]
        if self.descripcion:
            partes.append(self.descripcion)
        else:
            partes.append(f"{self.titulo}. Tiene {self.contar('punto')} puntos.")
        return " ".join(p for p in partes if p)


_RE_META = re.compile(r"^#\s*([^:=]+?)\s*[:=]\s*(.*)$")


def _num(s):
    return float(s.strip().replace(",", "."))


def interpretar_csv(texto):
    """Valida y convierte el texto del archivo. Lanza ErrorDatos si no sirve.
    Reconoce el formato handtracking (columna 'tipo' y coordenadas en mm) y el
    formato clásico (etiqueta, x, y, valor)."""
    texto = texto.lstrip("﻿")
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
                         "encabezados y al menos dos elementos.")

    delim = ";" if lineas[0].count(";") > lineas[0].count(",") else ","
    filas = list(csv.reader(lineas, delimiter=delim))
    enc = [normalizar_clave(c) for c in filas[0]]
    if "tipo" in enc and "x1_mm" in enc:
        return _interpretar_handtracking(filas, meta)
    return _interpretar_clasico(filas, meta)


# ------------------------------------------------------------
# Formato handtracking: una fila por elemento de la lámina, en mm
# ------------------------------------------------------------
COLUMNAS_HT = ("tipo", "id", "serie", "x1_mm", "y1_mm", "x2_mm", "y2_mm",
               "valor_x", "valor_y", "tendencia", "texto")
OBLIGATORIAS_HT = ("tipo", "x1_mm", "y1_mm", "x2_mm", "y2_mm", "texto")


def _leer_placa(texto):
    """Del texto de la placa saca: esquina de origen de las coordenadas,
    esquina recortada, tamaño del recorte y disposición del contenido."""
    t = sin_acentos(texto).lower()
    m = re.match(r"\s*esquina\s+(superior|inferior)\s+(izquierda|derecha)", t)
    origen = f"{m.group(1)} {m.group(2)}" if m else "superior izquierda"
    m = re.search(r"recortada\s+es\s+la\s+(superior|inferior)\s+(izquierda|derecha)", t)
    recortada = f"{m.group(1)} {m.group(2)}" if m else ""
    m = re.search(r"recortada\s+de\s+([\d.,]+)\s*mm", t)
    try:
        recorte = _num(m.group(1)) if m else 0.0
    except ValueError:
        recorte = 0.0
    disposicion = "; ".join(s.strip() for s in texto.split(";")
                            if s.strip() and "esquina" not in sin_acentos(s).lower())
    return origen, recortada, recorte, disposicion


def _titulo_de(descripcion):
    """Primera frase de la descripción, sin el prefijo 'Gráfica N:'."""
    frase = re.split(r"(?<=\.)\s", descripcion.strip(), maxsplit=1)[0].rstrip(". ")
    frase = re.sub(r"^gr[aá]fica\s*\d*\s*[:.-]\s*", "", frase, flags=re.I).strip()
    return frase[:120]


def _interpretar_handtracking(filas, meta):
    enc = [normalizar_clave(c) for c in filas[0]]
    idx = {n: (enc.index(n) if n in enc else None) for n in COLUMNAS_HT}
    faltan = [n for n in OBLIGATORIAS_HT if idx[n] is None]
    if faltan:
        raise ErrorDatos("Faltan estas columnas en el archivo: " + ", ".join(faltan) +
                         ". Los encabezados encontrados son: " + ", ".join(c for c in filas[0] if c.strip()) + ".")

    def campo(fila, n):
        i = idx[n]
        return fila[i].strip() if i is not None and i < len(fila) else ""

    def numero(fila, n, nfila):
        s = campo(fila, n)
        if not s:
            return None
        try:
            v = _num(s)
        except ValueError:
            raise ErrorDatos(f"En la fila {nfila} del archivo, la columna {n} debe ser un número.")
        if not math.isfinite(v):
            raise ErrorDatos(f"En la fila {nfila} del archivo, la columna {n} no es un número válido.")
        return v

    d = DatosGrafica()
    placa, descripciones, titulo, crudos = None, [], "", []
    for nfila, fila in enumerate(filas[1:], start=2):
        if not any(c.strip() for c in fila):
            continue
        tipo = normalizar_clave(campo(fila, "tipo"))
        if not tipo:
            continue
        texto = campo(fila, "texto")
        x1, y1, x2, y2 = (numero(fila, n, nfila) for n in ("x1_mm", "y1_mm", "x2_mm", "y2_mm"))
        if tipo == "placa":
            if None in (x1, y1, x2, y2):
                raise ErrorDatos("La fila de la placa necesita x1, y1, x2 e y2.")
            placa = (x1, y1, x2, y2, texto)
        elif tipo == "descripcion":
            if texto:
                descripciones.append(texto)
        elif tipo == "titulo":
            titulo = texto
        elif tipo == "serie":
            nombre = campo(fila, "serie") or campo(fila, "id")
            if nombre:
                d.series[nombre] = texto
        elif x1 is not None and y1 is not None:          # sin posición no hay nada que tocar
            crudos.append({"tipo": tipo, "id": campo(fila, "id"), "serie": campo(fila, "serie"),
                           "x1": x1, "y1": y1, "x2": x2, "y2": y2, "texto": texto,
                           "tendencia": campo(fila, "tendencia"),
                           "valor_y": numero(fila, "valor_y", nfila), "nfila": nfila})
            if len(crudos) > MAX_ELEMENTOS:
                raise ErrorDatos(f"El archivo tiene más de {MAX_ELEMENTOS} elementos; "
                                 "es demasiado grande para explorarlo.")

    if not crudos:
        raise ErrorDatos("El archivo no tiene elementos con posición para explorar, "
                         "como puntos, curvas o ejes.")
    if not any(c["tipo"] in ("punto", "curva", "tramo", "barra") for c in crudos):
        raise ErrorDatos("El archivo no tiene puntos ni curvas de datos.")

    # Lienzo de referencia = la placa (si falta, se estima a partir de los elementos).
    if placa is not None:
        px1, py1, px2, py2, texto_placa = placa
        ox, oy = min(px1, px2), min(py1, py2)
        d.ancho_ref, d.alto_ref = abs(px2 - px1), abs(py2 - py1)
        origen, d.esquina_recortada, d.recorte_mm, d.disposicion = _leer_placa(texto_placa)
    else:
        xs = [v for c in crudos for v in (c["x1"], c["x2"]) if v is not None]
        ys = [v for c in crudos for v in (c["y1"], c["y2"]) if v is not None]
        ox = oy = 0.0
        d.ancho_ref, d.alto_ref = max(xs) + min(xs), max(ys) + min(ys)
        origen = "superior izquierda"
    if d.ancho_ref <= 0 or d.alto_ref <= 0:
        raise ErrorDatos("La placa debe tener ancho y alto positivos.")

    # Todo se pasa a un lienzo con el origen arriba a la izquierda y la y hacia abajo,
    # que es como se ve la placa en la cámara.
    invertir_x, invertir_y = origen.endswith("derecha"), origen.startswith("inferior")

    def a_lienzo(x, y):
        x, y = x - ox, y - oy
        return (d.ancho_ref - x if invertir_x else x), (d.alto_ref - y if invertir_y else y)

    usados = set()
    for n, c in enumerate(crudos, start=1):
        tipo = c["tipo"]
        x1, y1 = a_lienzo(c["x1"], c["y1"])
        if c["x2"] is None or c["y2"] is None or tipo == "punto":
            forma, x2, y2 = "punto", x1, y1
        else:
            x2, y2 = a_lienzo(c["x2"], c["y2"])
            if tipo in TIPOS_SEGMENTO or abs(x2 - x1) < 0.5 or abs(y2 - y1) < 0.5:
                forma = "segmento"          # curvas, ejes y textos de una sola línea
            else:
                forma = "rect"              # recuadros de leyenda, barras...
        base = c["id"] or f"{tipo}_{n}"
        ident, k = base, 2
        while ident in usados:
            ident, k = f"{base}_{k}", k + 1
        usados.add(ident)

        texto = c["texto"]
        if not texto:
            partes = [tipo.capitalize(), c["serie"]]
            if c["valor_y"] is not None:
                partes.append(f"valor {fmt_num(c['valor_y'])}")
            texto = ". ".join(p for p in partes if p) + "."
        corto = texto
        if tipo == "eje":
            if ident.lower().endswith("x"):
                corto = "Eje horizontal"
            elif ident.lower().endswith("y"):
                corto = "Eje vertical"
        d.elementos.append({"id": ident, "tipo": tipo, "serie": c["serie"], "forma": forma,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "texto": texto, "corto": corto,
                            "tendencia": c["tendencia"]})
        if c["serie"] and c["serie"] not in d.series and tipo in ("punto", "curva", "tramo", "barra"):
            d.series[c["serie"]] = ""

    d.descripcion = " ".join(descripciones)
    d.titulo = (meta.get("titulo") or titulo or _titulo_de(d.descripcion) or d.titulo)
    return d


# ------------------------------------------------------------
# Formato clásico: etiqueta, x, y, valor, info (+ metadatos con #)
# ------------------------------------------------------------
def _interpretar_clasico(filas, meta):
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
        raise ErrorDatos("No reconozco el formato del archivo. Faltan estas columnas: " + ", ".join(faltan) +
                         ". Los encabezados encontrados son: " + ", ".join(c for c in filas[0] if c.strip()) +
                         ". Si es una gráfica táctil, debe tener las columnas tipo, x1_mm, y1_mm, x2_mm, "
                         "y2_mm y texto.")

    puntos = {}
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
        while nombre in puntos:
            nombre, k = f"{base} ({k})", k + 1
        info = fila[i_info].strip() if i_info is not None and i_info < len(fila) else ""
        puntos[nombre] = {"x": x, "y": y, "valor": valor, "info": info}
        if len(puntos) > 500:
            raise ErrorDatos("El archivo tiene más de 500 puntos; es demasiado grande para explorarlo.")

    if len(puntos) < 2:
        raise ErrorDatos("Se necesitan al menos dos puntos para formar una gráfica.")

    d = DatosGrafica()
    d.formato, d.estirar = "clasico", False
    d.titulo = meta.get("titulo") or d.titulo
    magnitud = meta.get("magnitud") or "Valor"
    unidad = meta.get("unidad", "")
    d.descripcion = meta.get("descripcion", "")

    def vt(v):
        return f"{fmt_num(v)} {unidad}".strip()

    xs = [p["x"] for p in puntos.values()]
    ys = [p["y"] for p in puntos.values()]
    try:
        d.ancho_ref = float(meta["ancho_referencia"]) if "ancho_referencia" in meta else max(xs) + min(xs)
        d.alto_ref = float(meta["alto_referencia"]) if "alto_referencia" in meta else max(ys) + min(ys)
    except ValueError:
        raise ErrorDatos("ancho_referencia y alto_referencia deben ser números.")
    if d.ancho_ref <= 0 or d.alto_ref <= 0:
        raise ErrorDatos("El lienzo de referencia debe tener ancho y alto positivos.")

    eje_x, eje_y = meta.get("eje_x", ""), meta.get("eje_y", "")
    mx = max(0.08 * (max(xs) - min(xs)), 10.0)
    my = max(0.08 * (max(ys) - min(ys)), 10.0)

    def num_meta(clave, defecto):
        if clave not in meta:
            return defecto
        try:
            return float(meta[clave].replace(",", "."))
        except ValueError:
            raise ErrorDatos(f"{clave} debe ser un número.")

    ox = num_meta("origen_x", max(0.0, min(xs) - mx))
    oy = num_meta("origen_y", min(d.alto_ref, max(ys) + my))
    fx = num_meta("fin_x", min(d.ancho_ref, max(xs) + mx))
    fy = num_meta("fin_y", max(0.0, min(ys) - my))

    nombres = list(puntos)
    vals = [p["valor"] for p in puntos.values()]
    el = d.elementos
    for i, (nombre, p) in enumerate(puntos.items(), start=1):
        t = f"{nombre}. {magnitud} de {vt(p['valor'])}." + (" " + p["info"] if p["info"] else "")
        el.append({"id": f"p{i}", "tipo": "punto", "serie": "", "forma": "punto",
                   "x1": p["x"], "y1": p["y"], "x2": p["x"], "y2": p["y"], "texto": t,
                   "corto": f"{nombre}: {vt(p['valor'])}", "tendencia": ""})
    for i, (a, b) in enumerate(zip(nombres, nombres[1:]), start=1):
        p1, p2 = puntos[a], puntos[b]
        tend = ("aumento" if p2["valor"] > p1["valor"] else
                "disminución" if p2["valor"] < p1["valor"] else "estable")
        el.append({"id": f"t{i}", "tipo": "curva", "serie": "", "forma": "segmento",
                   "x1": p1["x"], "y1": p1["y"], "x2": p2["x"], "y2": p2["y"],
                   "texto": (f"De {a} a {b}, tendencia de {tend}. "
                             f"{magnitud} pasa de {fmt_num(p1['valor'])} a {vt(p2['valor'])}."),
                   "corto": f"{a} → {b}: {tend}", "tendencia": tend})
    tx = "Eje horizontal." + (f" {eje_x}." if eje_x else "") + f" Va de {nombres[0]} a {nombres[-1]}."
    ty = ("Eje vertical." + (f" {eje_y}." if eje_y else f" {magnitud}.") +
          f" Los valores van de {vt(min(vals))} a {vt(max(vals))}.")
    el.append({"id": "eje_x", "tipo": "eje", "serie": "", "forma": "segmento",
               "x1": ox, "y1": oy, "x2": fx, "y2": oy, "texto": tx, "corto": "Eje horizontal", "tendencia": ""})
    el.append({"id": "eje_y", "tipo": "eje", "serie": "", "forma": "segmento",
               "x1": ox, "y1": oy, "x2": ox, "y2": fy, "texto": ty, "corto": "Eje vertical", "tendencia": ""})

    partes = [d.titulo + "."]
    if d.descripcion:
        partes.append(d.descripcion)
    partes.append(f"Tiene {len(puntos)} puntos.")
    if eje_x:
        partes.append(f"Eje horizontal: {eje_x}.")
    if eje_y:
        partes.append(f"Eje vertical: {eje_y}.")
    mayor = max(puntos.items(), key=lambda kv: kv[1]["valor"])
    menor = min(puntos.items(), key=lambda kv: kv[1]["valor"])
    partes.append(f"El valor más alto es {vt(mayor[1]['valor'])} en {mayor[0]}, "
                  f"y el más bajo es {vt(menor[1]['valor'])} en {menor[0]}.")
    d.resumen_fijo = " ".join(partes)
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


def filas_ejemplo():
    """Filas del CSV de ejemplo en formato handtracking: temperatura media mensual
    en una placa de 220 × 220 mm (misma maqueta que las láminas táctiles)."""
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
             "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    temps = [11, 13, 13, 17, 20, 23, 26, 27, 25, 21, 17, 14]
    serie, textura = "Temperatura media", "línea continua"
    letras = "ABCDEFGHIJKL"
    ox, oy, fin_x, fin_y = 35.8, 121.5, 210.0, 25.8          # ejes (mm)

    def mm(v):
        return f"{v:.1f}"

    def px(i):                                                # mes i → x en mm
        return 43.9 + i * 14.37

    def py(v):                                                # 10 °C → 113.8 mm · 30 °C → 35.2 mm
        return 113.8 - (v - 10) * 3.93

    i_max = max(range(12), key=lambda i: temps[i])
    i_min = min(range(12), key=lambda i: temps[i])
    if 0 < i_max < 11:
        forma = f"sube hasta {meses[i_max]} y luego baja"
    else:
        forma = "en general aumenta" if temps[-1] > temps[0] else "en general disminuye"
    descripcion = (
        "Gráfica 1: Temperatura media mensual. Gráfica de líneas con 1 curva. "
        f"Eje horizontal: Mes, con 12 categorías: {', '.join(meses)}. "
        "Eje vertical: Temperatura en grados Celsius, con marcas de 10 a 30. "
        f"La curva «{serie}» ({textura} en la lámina) empieza en {temps[0]} en {meses[0]} "
        f"y termina en {temps[-1]} en {meses[-1]}; {forma}. "
        f"Su valor máximo es {temps[i_max]} en {meses[i_max]}, y el mínimo es {temps[i_min]} en {meses[i_min]}. "
        "En la lámina, algunos textos van con una letra que se explica en la leyenda de abajo. "
        + "; ".join(f"{letras[i]}, categoría del eje X: «{m}»" for i, m in enumerate(meses)) + ".")

    filas = [["tipo", "id", "serie", "x1_mm", "y1_mm", "x2_mm", "y2_mm",
              "valor_x", "valor_y", "tendencia", "texto"],
             ["placa", "placa", "", "0", "0", "220.0", "220.0", "", "", "",
              "esquina superior izquierda, y hacia abajo (la esquina recortada es la superior derecha); "
              "esquina recortada de 10.0 mm; gráfico arriba y leyenda abajo"],
             ["descripcion", "descripcion", "", "", "", "", "", "", "", "", descripcion],
             ["serie", "serie_1", serie, "", "", "", "", "", "", "", textura.capitalize()]]
    for i, (m, t) in enumerate(zip(meses, temps)):
        filas.append(["punto", f"s1_p{i + 1}", serie, mm(px(i)), mm(py(t)), "", "",
                      f"{float(i):.1f}", str(t), "", f"{serie}. {m}: {t} grados."])
    for i in range(11):
        t1, t2 = temps[i], temps[i + 1]
        if t2 > t1:
            tend, frase = "aumento", f"sube de {t1} a {t2}"
        elif t2 < t1:
            tend, frase = "disminución", f"baja de {t1} a {t2}"
        else:
            tend, frase = "estable", f"se mantiene en {t1}"
        filas.append(["curva", f"s1_p{i + 1}-p{i + 2}", serie, mm(px(i)), mm(py(t1)), mm(px(i + 1)), mm(py(t2)),
                      "", "", tend, f"{serie}. De {meses[i]} a {meses[i + 1]}, {frase}."])
    filas.append(["leyenda", "leyenda_1", serie, "10.0", "156.8", "150.0", "156.8", "", "", "",
                  f"Leyenda: la {textura} es {serie}."])
    n = 0
    for i in range(0, 12, 2):                                 # una letra cada dos meses, como en la lámina
        n += 1
        filas.append(["texto", f"texto_{n}", "", mm(px(i) - 5.05), "131.1", mm(px(i) + 5.05), "131.1",
                      "", "", "", f"Categoría, abreviado con la letra {letras[i]}: {meses[i]}."])
    for v in range(10, 31, 5):
        n += 1
        filas.append(["texto", f"texto_{n}", "", "10.0", mm(py(v)), "26.3", mm(py(v)), "", "", "",
                      f"Eje vertical (Temperatura en grados Celsius): {v}."])
    filas.append(["texto", f"texto_{n + 1}", "", "111.6", "141.1", "134.1", "141.1", "", "", "",
                  "Título del eje X: Mes."])
    filas.append(["texto", f"texto_{n + 2}", "", "19.5", "13.2", "150.0", "13.2", "", "", "",
                  "Título del eje Y: Temperatura en grados Celsius."])
    for i, m in enumerate(meses):                             # 4 recuadros por fila bajo el gráfico
        x, y = 10.0 + (i % 4) * 52.1, 166.8 + (i // 4) * 10.0
        filas.append(["leyenda", f"leyenda_{letras[i]}", "", mm(x), mm(y), mm(x + 41.1), mm(y + 6.3),
                      "", "", "", f"Leyenda: {letras[i]} es {m}."])
    filas.append(["eje", "eje_x", "", mm(ox), mm(oy), mm(fin_x), mm(oy), "", "", "",
                  f"Eje horizontal: Mes, de {meses[0]} a {meses[-1]}."])
    filas.append(["eje", "eje_y", "", mm(ox), mm(oy), mm(ox), mm(fin_y), "", "", "",
                  "Eje vertical: Temperatura en grados Celsius, con marcas de 10 a 30."])
    return filas


def crear_ejemplo():
    """Deja un CSV de ejemplo (formato handtracking) para probar sin internet.
    Si existe la versión antigua (formato clásico) la reemplaza."""
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    ruta = os.path.join(CARPETA_DATOS, "ejemplo_temperatura.csv")
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8-sig") as f:
                if "x1_mm" in f.readline():
                    return                  # ya es la versión handtracking
        except OSError:
            return
    try:
        with open(ruta, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(filas_ejemplo())
    except OSError as e:
        print("Aviso: no se pudo crear el archivo de ejemplo:", e)


# ============================================================
# GEOMETRÍA
# ============================================================
def distancia(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def distancia_segmento(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return distancia(px, py, x1, y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return distancia(px, py, x1 + t * dx, y1 + t * dy)


def distancia_rect(px, py, x1, y1, x2, y2):
    """0 dentro del recuadro; fuera, la distancia al borde más cercano."""
    dx = max(min(x1, x2) - px, 0, px - max(x1, x2))
    dy = max(min(y1, y2) - py, 0, py - max(y1, y2))
    return math.hypot(dx, dy)


def distancia_elemento(px, py, e):
    if e["forma"] == "punto":
        return distancia(px, py, e["x1"], e["y1"])
    if e["forma"] == "rect":
        return distancia_rect(px, py, e["x1"], e["y1"], e["x2"], e["y2"])
    return distancia_segmento(px, py, e["x1"], e["y1"], e["x2"], e["y2"])


def transformacion(ancho_ref, alto_ref, hoja, estirar=False):
    """Devuelve f(x, y) que lleva coordenadas del lienzo de referencia al
    rectángulo de la hoja. 'hoja' = (x, y, ancho, alto).
    · estirar=True: el lienzo ocupa exactamente la hoja (formato handtracking:
      la hoja calibrada son las esquinas de la placa).
    · estirar=False: un solo factor, sin deformar, centrado (formato clásico)."""
    hx, hy, hw, hh = hoja
    if estirar:
        sx, sy = hw / ancho_ref, hh / alto_ref
        return lambda x, y: (int(round(hx + x * sx)), int(round(hy + y * sy)))
    escala = min(hw / ancho_ref, hh / alto_ref)
    mx = hx + (hw - ancho_ref * escala) / 2
    my = hy + (hh - alto_ref * escala) / 2
    return lambda x, y: (int(mx + x * escala), int(my + y * escala))


def escalar_elementos(datos, hoja):
    """Copia de los elementos con las coordenadas en píxeles del frame."""
    f = transformacion(datos.ancho_ref, datos.alto_ref, hoja, datos.estirar)
    res = []
    for e in datos.elementos:
        p = dict(e)
        p["x1"], p["y1"] = f(e["x1"], e["y1"])
        p["x2"], p["y2"] = f(e["x2"], e["y2"])
        res.append(p)
    return res


def _esquina_recortada(contorno, rect):
    """¿Qué esquina del rectángulo envolvente queda lejos del contorno? Esa es la
    esquina recortada de la placa. Devuelve 'superior derecha', ... o None."""
    import numpy as np
    x, y, w, h = rect
    pts = contorno.reshape(-1, 2).astype("float32")
    esquinas = {"superior izquierda": (x, y), "superior derecha": (x + w - 1, y),
                "inferior izquierda": (x, y + h - 1), "inferior derecha": (x + w - 1, y + h - 1)}
    dist = {n: float(np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))) for n, (cx, cy) in esquinas.items()}
    (n1, d1), (_n2, d2) = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if d1 >= 0.02 * min(w, h) and d1 >= 2.5 * max(d2, 1.0):
        return n1
    return None


def detectar_hoja(frame):
    """Intenta hallar la hoja o placa como el cuadrilátero convexo más grande
    (admite una esquina recortada). Devuelve ((x, y, ancho, alto), esquina_recortada)
    o (None, None). Es una ayuda, no una garantía."""
    h, w = frame.shape[:2]
    gris = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for img in (bw, 255 - bw):                  # hoja clara sobre fondo oscuro, o al revés
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in sorted(contornos, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < 0.15 * h * w or area > 0.97 * h * w:
                continue
            aprox = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(aprox) in (4, 5) and cv2.isContourConvex(aprox):
                rect = cv2.boundingRect(c)      # del contorno: incluye la esquina recortada
                return rect, _esquina_recortada(c, rect)
    return None, None


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
        self.invertido = False              # polaridad con la que se vio el QR la última vez
        self._inv_actual = False
        try:
            from pyzbar import pyzbar
            self.zbar = pyzbar
        except Exception:
            self.zbar = None

    def _variantes(self, gris):
        """(imagen, factor) con distintos preprocesados: el QR impreso a
        distintas alturas llega con tamaños, contraste y enfoque variables.
        Se prueban las dos polaridades: un QR impreso en relieve claro sobre
        fondo oscuro (p. ej. blanco sobre azul) está INVERTIDO, y los lectores
        de OpenCV y zbar solo leen módulos oscuros sobre fondo claro. Primero
        va la polaridad con la que se vio el QR la última vez."""
        def de_polaridad(g):
            yield g, 1.0
            yield self.clahe.apply(g), 1.0
            yield cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0, 0), 3), -0.8, 0), 1.0
            yield cv2.resize(g, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC), 1.6
            yield cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10), 1.0
        orden = (True, False) if self.invertido else (False, True)
        for inv in orden:
            self._inv_actual = inv
            yield from de_polaridad(255 - gris if inv else gris)

    @staticmethod
    def poligono_plausible(pts, w, h):
        """¿Puede ser un QR? Un QR visto por la cámara es un cuadrilátero convexo
        casi cuadrado (con algo de perspectiva), dentro de la imagen y de tamaño
        razonable. El detector de OpenCV a veces 'encuentra' cuadriláteros sobre
        las texturas de la gráfica (líneas de puntos, recuadros): esos se descartan."""
        import numpy as np
        if pts is None:
            return False
        p = np.asarray(pts, dtype="float32").reshape(-1, 2)
        if len(p) != 4 or not np.all(np.isfinite(p)):
            return False
        margen = 0.03 * max(w, h)
        if (p[:, 0] < -margen).any() or (p[:, 0] > w + margen).any() or \
           (p[:, 1] < -margen).any() or (p[:, 1] > h + margen).any():
            return False
        lados = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
        if lados.min() < 12 or lados.max() > 1.8 * lados.min():
            return False
        diagonales = (np.hypot(*(p[2] - p[0])), np.hypot(*(p[3] - p[1])))
        if max(diagonales) > 1.4 * min(diagonales):
            return False
        if not cv2.isContourConvex(p.reshape(-1, 1, 2)):
            return False
        area = abs(cv2.contourArea(p))
        return 0.002 * w * h <= area <= 0.6 * w * h

    def leer(self, frame):
        """Devuelve (texto|None, polígono|None). Si solo se localiza el QR pero
        no se decodifica, devuelve (None, polígono) para poder guiar al usuario;
        ese polígono solo se acepta si tiene forma de QR."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gris.shape[:2]
        poligono, poli_inv = None, self.invertido
        for img, f in self._variantes(gris):
            if self.zbar is not None:
                try:
                    for o in self.zbar.decode(img, symbols=[self.zbar.ZBarSymbol.QRCODE]):
                        txt = o.data.decode("utf-8", "replace").strip()
                        if txt:
                            self.invertido = self._inv_actual
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
                cand = None
                if pts is not None and len(pts):
                    cand = pts.reshape(-1, 2) / f
                    if poligono is None and self.poligono_plausible(cand, w, h):
                        poligono, poli_inv = cand, self._inv_actual
                if txt:
                    self.invertido = self._inv_actual
                    return txt.strip(), (cand if self.poligono_plausible(cand, w, h) else None)
        if poligono is None:                    # último recurso: solo localizar
            for inv in ((True, False) if self.invertido else (False, True)):
                try:
                    ok, pts = self.cv.detect(255 - gris if inv else gris)
                except cv2.error:
                    continue
                if ok and pts is not None and self.poligono_plausible(pts, w, h):
                    poligono, poli_inv = pts.reshape(-1, 2), inv
                    break
        if poligono is not None:
            self.invertido = poli_inv             # la próxima vez se empieza por esa polaridad
        return None, poligono

    @staticmethod
    def _arr(pts):
        import numpy as np
        return np.array(pts, dtype="float32").reshape(-1, 1, 2)


# ============================================================
# SEGUIMIENTO DE LA MANO (MediaPipe actual: API Tasks · HandLandmarker)
# ============================================================
# Las versiones recientes de mediapipe (0.10.30 en adelante, y la serie 1.x) ya no
# incluyen mp.solutions.hands. La API vigente es HandLandmarker, que necesita un
# archivo de modelo (.task). Este bloque:
#   · descarga el modelo una sola vez (o usa el que indique "modelo_mano"),
#   · lo carga desde memoria (evita fallos con rutas que tienen tildes o eñes),
#   · usa el modo VIDEO (seguimiento entre frames: más rápido y estable que
#     detectar cada imagen por separado),
#   · y, si se instaló un mediapipe antiguo que sí trae mp.solutions, lo usa.
# Todos los errores salen como ErrorSeguimiento, con un texto para leer en voz alta.

class ErrorSeguimiento(Exception):
    """Mensaje pensado para ser leído en voz alta al usuario."""


_CANDADO_MODELO = threading.Lock()      # la descarga previa y la de la cámara no se pisan
MIN_BYTES_MODELO = 1_000_000            # el modelo real pesa unos 7,5 MB


def ruta_modelo_mano(cfg):
    ruta = str(cfg.get("modelo_mano") or "").strip().strip("\"'")
    return os.path.expanduser(ruta) if ruta else os.path.join(CARPETA_MODELOS, NOMBRE_MODELO_MANO)


def _motivo_modelo_invalido(ruta):
    """None si el archivo parece un modelo; si no, el motivo (para la consola).
    Solo descarta lo evidente (archivo pequeño o una página web): quien comprueba
    de verdad el contenido es MediaPipe al cargarlo."""
    try:
        tam = os.path.getsize(ruta)
        with open(ruta, "rb") as f:
            inicio = f.read(64)
    except OSError as e:
        return f"no se puede leer ({e})"
    if tam < MIN_BYTES_MODELO:
        return f"pesa solo {tam} bytes; inicio: {inicio[:32]!r}"
    if inicio.lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml")):
        return f"es una página web, no un modelo; inicio: {inicio[:32]!r}"
    return None


def _modelo_valido(ruta):
    return _motivo_modelo_invalido(ruta) is None


def asegurar_modelo_mano(cfg, avisar=None):
    """Devuelve la ruta del modelo de la mano; lo descarga si hace falta.
    'avisar(texto)' se llama antes de una descarga (para decirlo en voz alta)."""
    ruta = ruta_modelo_mano(cfg)
    with _CANDADO_MODELO:
        if _modelo_valido(ruta):
            return ruta
        if cfg.get("modelo_mano"):
            raise ErrorSeguimiento(
                "No encuentro el modelo de seguimiento de la mano en la ruta indicada en la configuración: "
                f"{ruta}. Corrige la clave modelo_mano o déjala vacía para descargarlo automáticamente.")
        if avisar:
            avisar("Descargando el modelo de seguimiento de la mano. Solo se hace la primera vez "
                   "y tarda unos segundos.")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        parcial = ruta + ".parcial"
        try:
            req = urllib.request.Request(URL_MODELO_MANO,
                                         headers={"User-Agent": "Mozilla/5.0 (ExploradorGraficas)"})
            recibidos = 0
            with urllib.request.urlopen(req, timeout=60) as r, open(parcial, "wb") as f:
                esperado = int(r.headers.get("Content-Length") or 0)
                while True:
                    trozo = r.read(256 * 1024)
                    if not trozo:
                        break
                    f.write(trozo)
                    recibidos += len(trozo)
            print(f"Modelo de la mano: recibidos {recibidos} bytes"
                  + (f" de {esperado}." if esperado else "."))
            if esperado and recibidos < esperado:
                raise ErrorSeguimiento("La descarga del modelo de la mano se cortó antes de terminar. "
                                       "Revisa la conexión e inténtalo de nuevo.")
            motivo = _motivo_modelo_invalido(parcial)
            if motivo:
                print("Aviso: el archivo descargado no parece el modelo de la mano:", motivo)
                raise ErrorSeguimiento(
                    "Lo que se descargó no es el modelo de la mano; puede que la red lo esté bloqueando. "
                    "Puedes descargar el archivo hand_landmarker.task en el navegador y copiarlo en la "
                    f"carpeta modelos, junto al programa: {os.path.dirname(ruta)}.")
            os.replace(parcial, ruta)
        except ErrorSeguimiento:
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            print("Aviso: no se pudo descargar el modelo de la mano:", repr(e))
            raise ErrorSeguimiento(
                "No pude descargar el modelo de seguimiento de la mano. La primera vez se necesita "
                "conexión a internet. También puedes descargar el archivo hand_landmarker.task y "
                f"copiarlo en la carpeta modelos, junto al programa: {os.path.dirname(ruta)}.")
        finally:
            if os.path.exists(parcial):
                try:
                    os.remove(parcial)
                except OSError:
                    pass
        print("Modelo de la mano guardado en:", ruta)
        return ruta


def _api_tasks(mp):
    """(vision, BaseOptions) de MediaPipe Tasks, o (None, None) si no existe.
    Prueba las rutas de importación de las distintas versiones."""
    vision = base = None
    try:
        from mediapipe.tasks.python import vision                       # noqa: F811
    except Exception:
        vision = getattr(getattr(mp, "tasks", None), "vision", None)
    try:
        from mediapipe.tasks.python import BaseOptions as base
    except Exception:
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions as base
        except Exception:
            base = getattr(getattr(mp, "tasks", None), "BaseOptions", None)
    if vision is None or base is None or not hasattr(vision, "HandLandmarker"):
        return None, None
    return vision, base


class DetectorMano:
    """Devuelve, para cada mano vista, su lista de 21 puntos (con .x e .y entre 0 y 1).
    Punto 8 = punta del índice · punto 12 = punta del dedo medio (igual en ambas APIs)."""

    def __init__(self, cfg, avisar=None):
        import mediapipe as mp                     # puede lanzar ImportError
        self.mp = mp
        self.version = getattr(mp, "__version__", "?")
        conf = float(cfg.get("confianza_mano", 0.5))
        vision, BaseOptions = _api_tasks(mp)
        self._ultimo_ts = -1
        self._t0 = time.monotonic()

        if vision is not None:
            ruta = asegurar_modelo_mano(cfg, avisar)
            with open(ruta, "rb") as f:            # desde memoria: rutas con tildes no fallan
                modelo = f.read()
            opciones = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=modelo),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=conf,
                min_hand_presence_confidence=conf,
                min_tracking_confidence=conf)
            try:
                self.motor = vision.HandLandmarker.create_from_options(opciones)
            except Exception as e:
                traceback.print_exc()
                if not cfg.get("modelo_mano"):        # descargado por el programa: se repone solo
                    try:
                        os.replace(ruta, ruta + ".defectuoso")
                    except OSError:
                        pass
                    raise ErrorSeguimiento(
                        "El modelo de la mano estaba dañado y no se pudo cargar. Lo descarté; vuelve al "
                        "inicio con Escape y empieza otra vez para descargarlo de nuevo.")
                raise ErrorSeguimiento(
                    "No pude cargar el modelo de la mano de la ruta indicada en la configuración. "
                    "Comprueba que sea el archivo hand_landmarker.task.")
            self.api = "tasks"
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            kw = dict(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=conf, min_tracking_confidence=conf)
            try:
                self.motor = mp.solutions.hands.Hands(model_complexity=int(cfg.get("complejidad_mano", 0)), **kw)
            except TypeError:                       # mediapipe muy antiguo sin ese parámetro
                self.motor = mp.solutions.hands.Hands(**kw)
            self.api = "clasica"
        else:
            raise ErrorSeguimiento(
                f"La versión {self.version} de mediapipe no tiene seguimiento de manos. "
                "Actualízala con: pip install --upgrade mediapipe")
        print(f"Seguimiento de la mano: mediapipe {self.version}, API "
              f"{'Tasks (HandLandmarker)' if self.api == 'tasks' else 'clásica (mp.solutions)'}.")

    def _marca_tiempo(self):
        """El modo VIDEO exige milisegundos estrictamente crecientes."""
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._ultimo_ts:
            ts = self._ultimo_ts + 1
        self._ultimo_ts = ts
        return ts

    def detectar(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.api == "tasks":
            import numpy as np
            imagen = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self.motor.detect_for_video(imagen, self._marca_tiempo())
            return list(res.hand_landmarks or [])
        res = self.motor.process(rgb)
        return [hl.landmark for hl in (res.multi_hand_landmarks or [])]

    def cerrar(self):
        try:
            self.motor.close()
        except Exception:
            pass


def preparar_modelo_en_segundo_plano(cfg):
    """Al arrancar la app: si falta el modelo, lo descarga sin hablar ni bloquear,
    para que al empezar la exploración ya esté listo. Si falla, no pasa nada:
    se reintenta (y se avisa en voz alta) al abrir la exploración."""
    def trabajo():
        try:
            import mediapipe as mp
            if _api_tasks(mp)[0] is not None:
                asegurar_modelo_mano(cfg)
        except Exception as e:
            print("Aviso: el modelo de la mano se descargará al empezar la exploración:", e)
    threading.Thread(target=trabajo, daemon=True).start()


# ============================================================
# EXPLORADOR (gesto de pinza + narración)
# ============================================================
# Colores BGR para las series (el cian queda para los ejes y el rojo para los puntos).
COLORES_SERIE = [(0, 200, 0), (0, 140, 255), (255, 0, 255), (255, 128, 0), (0, 220, 220), (180, 105, 255)]
COLOR_EJE, COLOR_PUNTO, COLOR_TEXTO = (255, 255, 0), (0, 0, 255), (170, 170, 170)


class Explorador:
    def __init__(self, voz, cfg, avisar=None):
        self.detector = DetectorMano(cfg, avisar)  # puede lanzar ImportError o ErrorSeguimiento
        self.voz, self.cfg = voz, cfg
        self.datos = None
        self.elementos = []
        self.colores = {}
        self.anterior = None
        self.audio_activado = False
        self.pinza_antes = False
        self.t_mano = time.time()      # última vez que se vio la mano

    def configurar(self, datos, hoja):
        self.datos = datos
        self.elementos = escalar_elementos(datos, hoja)
        self._ajustar_radios_puntos()
        series = list(datos.series) or sorted({e["serie"] for e in self.elementos})
        self.colores = {s: COLORES_SERIE[i % len(COLORES_SERIE)] for i, s in enumerate(series)}
        self.anterior = None
        self.t_mano = time.time()

    def cerrar(self):
        self.detector.cerrar()

    def _ajustar_radios_puntos(self):
        """Con puntos muy juntos, el radio_punto los taparía a ellos y a las curvas
        entre ellos. Cada punto usa como mucho el 40 % de la distancia a su vecino
        más cercano, así el tramo central de cada curva siempre se puede tocar."""
        import numpy as np
        pts = [e for e in self.elementos if e["tipo"] == "punto"]
        if len(pts) < 2:
            return
        xy = np.array([(e["x1"], e["y1"]) for e in pts], dtype="float32")
        dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(dist, np.inf)
        vecino = dist.min(axis=1)
        for e, dv in zip(pts, vecino):
            e["radio"] = max(4.0, min(float(self.cfg["radio_punto"]), 0.4 * float(dv)))

    def _radio(self, e):
        if "radio" in e:
            return e["radio"]
        if e["forma"] == "punto":
            return self.cfg["radio_punto"]
        if e["tipo"] in ("curva", "tramo"):
            return self.cfg["radio_linea"]
        if e["tipo"] == "eje":
            return self.cfg["radio_eje"]
        return self.cfg.get("radio_texto", 12)

    def _buscar(self, x, y):
        """Elemento bajo el dedo: el de mayor prioridad y, a igualdad, el más cercano."""
        mejor, clave_mejor = None, None
        for e in self.elementos:
            d = distancia_elemento(x, y, e)
            if d <= self._radio(e):
                clave = (PRIORIDAD.get(e["tipo"], PRIORIDAD_RESTO), d)
                if clave_mejor is None or clave < clave_mejor:
                    mejor, clave_mejor = e, clave
        return mejor

    def _dibujar(self, frame, actual):
        for e in self.elementos:                                  # textos y leyendas: gris fino
            if e["tipo"] in ("punto", "curva", "tramo", "eje"):
                continue
            if e["forma"] == "rect":
                cv2.rectangle(frame, (e["x1"], e["y1"]), (e["x2"], e["y2"]), COLOR_TEXTO, 1)
            elif e["forma"] == "segmento":
                cv2.line(frame, (e["x1"], e["y1"]), (e["x2"], e["y2"]), COLOR_TEXTO, 1)
            else:
                cv2.circle(frame, (e["x1"], e["y1"]), 3, COLOR_TEXTO, -1)
        for e in self.elementos:                                  # ejes en cian
            if e["tipo"] == "eje":
                cv2.line(frame, (e["x1"], e["y1"]), (e["x2"], e["y2"]), COLOR_EJE, 2)
                if e["corto"] in ("Eje horizontal", "Eje vertical"):
                    cv2.putText(frame, "X" if e["corto"] == "Eje horizontal" else "Y",
                                (e["x2"] + 6, e["y2"] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EJE, 2)
        for e in self.elementos:                                  # curvas: un color por serie
            if e["tipo"] in ("curva", "tramo"):
                cv2.line(frame, (e["x1"], e["y1"]), (e["x2"], e["y2"]),
                         self.colores.get(e["serie"], COLORES_SERIE[0]), 2)
        for e in self.elementos:                                  # puntos en rojo
            if e["tipo"] == "punto":
                cv2.circle(frame, (e["x1"], e["y1"]), 5, COLOR_PUNTO, -1)
        if actual is not None:                                    # elemento señalado: resaltado
            if actual["forma"] == "punto":
                cv2.circle(frame, (actual["x1"], actual["y1"]), 10, (255, 255, 255), 2)
            elif actual["forma"] == "rect":
                cv2.rectangle(frame, (actual["x1"], actual["y1"]), (actual["x2"], actual["y2"]), (255, 255, 255), 2)
            else:
                cv2.line(frame, (actual["x1"], actual["y1"]), (actual["x2"], actual["y2"]), (255, 255, 255), 4)

    def procesar(self, frame):
        """Dibuja sobre 'frame' y devuelve (frame, texto_visible|None)."""
        h, w = frame.shape[:2]
        manos = self.detector.detectar(frame)
        actual = None
        dedo = None

        if not manos:
            self.audio_activado = False             # mano fuera = pinza soltada
        else:
            self.t_mano = time.time()
            for puntos_mano in manos:
                medio, indice = puntos_mano[12], puntos_mano[8]
                mx, my = int(medio.x * w), int(medio.y * h)
                x, y = int(indice.x * w), int(indice.y * h)
                self.audio_activado = distancia(mx, my, x, y) <= self.cfg["distancia_pinza"]
                dedo = (mx, my, x, y)
                actual = self._buscar(x, y)

        ident = actual["id"] if actual is not None else None

        # Audio: solo con pinza activa, sin solapar, solo al cambiar de elemento.
        if self.audio_activado and ident != self.anterior and not self.voz.hablando.is_set():
            if actual is not None:
                self.voz.decir(actual["texto"], esencial=True)
                self.anterior = ident
        if not self.audio_activado:
            if self.pinza_antes:
                self.voz.detener()
            self.anterior = None
        self.pinza_antes = self.audio_activado

        self._dibujar(frame, actual)
        if dedo is not None:                                      # los dedos encima de todo
            mx, my, x, y = dedo
            cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
            cv2.line(frame, (mx, my), (x, y), (255, 0, 0), 2)

        cv2.putText(frame, "AUDIO ACTIVO" if self.audio_activado else "Junta medio + indice",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if self.audio_activado else (255, 255, 255), 2)

        visible = None
        if actual is not None:
            visible = actual["corto"]
            corto = texto_cv(visible)
            if len(corto) > 55:
                corto = corto[:52] + "..."
            cv2.putText(frame, corto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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
        self._t_res_visto = None                  # último resultado del lector ya contado
        self._racha_poli, self._centro_prev = 0, None   # QR localizado en lecturas seguidas
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
        if res is not None and res[2] != self._t_res_visto:     # resultado nuevo del lector
            self._t_res_visto = res[2]
            if res[0] is None and res[1] is not None:
                centro = res[1].reshape(-1, 2).mean(axis=0)
                cerca = (self._centro_prev is not None and
                         float(((centro - self._centro_prev) ** 2).sum()) ** 0.5 < 0.08 * max(w, h))
                self._racha_poli = self._racha_poli + 1 if cerca else 1
                self._centro_prev = centro
            else:
                self._racha_poli, self._centro_prev = 0, None
        if res is not None and ahora - res[2] < 0.8:
            texto, poli = res[0], res[1]
            if texto is None and self._racha_poli < 2:
                poli = None                     # visto una sola vez: puede ser un falso positivo
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
                self.explorador = Explorador(self.voz, self.cfg,
                                             avisar=lambda t: self._emitir("estado", t, True))
            except ImportError:
                traceback.print_exc()
                self._emitir("error_camara", "Falta la biblioteca mediapipe, o no se pudo cargar. "
                                             "Instálala con: pip install --upgrade mediapipe")
                self.modo = "vista"
                return frame
            except ErrorSeguimiento as e:
                self._emitir("error_camara", str(e))
                self.modo = "vista"
                return frame
            self._reconfigurar = True
        if self._autodetectar:
            self._autodetectar = False
            rect, esquina = detectar_hoja(frame)
            if rect:
                self.set_hoja(rect)
            self._emitir("hoja", rect, frame.shape[1] * frame.shape[0], esquina)
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
        preparar_modelo_en_segundo_plano(self.cfg)    # modelo de la mano listo antes de explorar
        print("Configuración:", ARCHIVO_CONFIG)
        print("Datos descargados:", CARPETA_DATOS)
        print("Modelo de la mano:", ruta_modelo_mano(self.cfg))
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
                      "medio y el índice sobre un punto, una curva, un eje, un texto o la leyenda para escuchar "
                      "su información; "
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
                          "de la placa en el video, y luego en la esquina inferior derecha.")
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
                texto = f"Placa detectada, ocupa el {pct} por ciento de la imagen."
                tipo_msg = "ok"
                esperada = self.datos.esquina_recortada if self.datos is not None else ""
                vista = ev[3] if len(ev) > 3 else None
                if esperada and vista:
                    if vista == esperada:
                        texto += f" La esquina recortada se ve {ESQUINAS_HABLADAS[vista]}, como debe ser."
                    else:
                        texto += (f" Atención: la esquina recortada parece estar {ESQUINAS_HABLADAS[vista]}, "
                                  f"pero debe ir {ESQUINAS_HABLADAS[esperada]}. Gira la placa y pulsa A otra vez.")
                        tipo_msg = "error"
                self._mensaje(texto + " Si puedes, pide a una persona vidente que lo confirme; "
                              "si no es correcto, usa calibrar.", tipo_msg)
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
        resumen = f"{datos.resumen_corto()}.{aviso}"
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
