#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLORADOR DE GRÁFICAS IMPRESAS CON TEXTURA  ·  GUI accesible (tkinter)
=======================================================================

Pensado para personas ciegas o de baja visión. Se ejecuta en local:

    python explorador_graficas_gui_qr.py

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
   La pinza se mide en proporción a la palma y tolera un instante con los dedos
   separados; en los cruces se sigue la misma serie y se avisa "Cruce con ...".

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
   columna "texto". Si hay varios cerca: punto > curva; ejes, textos y leyendas ganan
   solo si están claramente más cerca que el dato (ver interaccion.py).
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
    En la pestaña Cámara:  R repetir descripción · C calibrar la placa (4 esquinas)
                           A autodetectar la placa (y enderezar la imagen con el QR si se ve)
                           O cambiar la orientación de la imagen de la cámara
                           + / -  agrandar / reducir el video

Módulos
-------
    vision.py       cámara, QR, mano, registro (px ↔ mm de la placa) y dibujo (Persona A)
    interaccion.py  selección en mm, pinza, estabilidad, cruces y narración (Persona B)
    este archivo    datos, voz e interfaz; reexporta los nombres de vision.py e
                    interaccion.py que usa medir_metricas.py
"""

from __future__ import annotations

import csv
import hashlib
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

# Visión y registro (Persona A). Se reexportan: medir_metricas.py los usa como eg.<nombre>.
from vision import (  # noqa: F401
    URL_MODELO_MANO, NOMBRE_MODELO_MANO, VISTA_DEFECTO, NOMBRES_ESQUINAS,
    Registro, MuestraMano, DeteccionPlaca, detectar_placa, detectar_hoja, registro_para,
    ordenar_esquinas, esquinas_validas, esquinas_qr_mm, muestra_de_mano, dedos_px, roi_mano,
    Superposicion, COLORES_SERIE, COLOR_EJE, COLOR_PUNTO, COLOR_TEXTO,
    listar_camaras, ORIENTACIONES, GIROS, aplicar_orientacion, matriz_orientacion, cambio_orientacion,
    convertir_rect, orientacion_desde_qr, LectorQR, ErrorSeguimiento, ruta_modelo_mano,
    asegurar_modelo_mano, DetectorMano, preparar_modelo_en_segundo_plano, Captura, MotorVision,
    fourcc_texto, fijar_formato, ZoomQR, PlantillaPlaca, RESOLUCIONES_QR,
    misma_proporcion, VERSION_MODO_QR,
)
# Selección con la pinza y narración, en mm (Persona B). Se reexportan para medir_metricas.py.
from interaccion import (  # noqa: F401
    Interaccion, Seleccion, PRIORIDAD, PRIORIDAD_RESTO, DEFECTO as DEFECTO_INTERACCION,
    distancia, distancia_segmento, distancia_rect, distancia_elemento,
)

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

CONFIG_DEFECTO = {
    "nombre_camara": "WN CAM-K211L",   # nombre tal cual aparece en la app Cámara
    "indice_camara": None,             # número fijo (anula la búsqueda por nombre)
    "orientaciones": {},               # orientación guardada por cámara: {"nombre de la cámara": "180", ...}
    "orientacion_auto_qr": True,       # al leer el QR, girar la imagen para que la placa quede derecha
    "voltear": "auto",                 # (antiguo) orientación inicial para cámaras sin una guardada; auto = sin girar
    "voz": "",                         # parte del nombre de la voz a usar (p. ej. "Sabina"); vacío = una en español
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
    # --- Registro y visión (Persona A) ---
    "esquinas_placa_px": None,         # 4 esquinas de la placa en la imagen orientada (sup-izq, sup-der,
                                       # inf-der, inf-izq); si faltan se usa 'hoja' (que queda como su caja)
    "registro_origen": "",             # cómo se obtuvieron: autodeteccion | clics | qr | rect
    "registro_error_mm": None,         # error estimado del registro (None = no se sabe)
    "registro_camara": "",             # cámara y resolución con que se calibró (para avisar si cambian)
    "registro_resolucion": None,
    "autodetectar_al_empezar": True,   # al empezar a explorar, comprobar dónde está la placa
    "seguir_placa": True,              # sin mano encima, revisar cada pocos segundos si la placa se movió
    "seguir_placa_cada_s": 2.5,
    "umbral_movimiento_placa_mm": 3.0, # a partir de cuánto se considera que la placa se movió
    "qr_ancla": True,                  # si no se ve el borde de la placa, corregir su posición con el QR
    "mano_recorte_placa": True,        # buscar la mano solo en la placa y un margen alrededor
    "mano_margen_recorte": 0.4,        # margen, en fracción del lado de la placa
    "mano_lado_max_px": 0,             # reducir la imagen para el modelo de la mano (0 = no)
    "fps_camara": 30,
    "exposicion": None,                # None = automática; p. ej. -6 (DirectShow) fija ~1/64 s y evita bajar a 15 fps
    # --- Lectura del QR (F3) con zoom a la esquina de la placa ---
    "textos_cortos": True,
    "qr_url_base": "https://tactonauta.onrender.com/q/",   # un QR con SOLO el código (p. ej. 54D7E01D5C)
                                       # se completa con esto: QR más pequeño, módulos más grandes             # marcas de los ejes: "20. Marca del eje vertical." (el número primero)
    "resolucion_qr": "auto",           # "auto": la cámara prueba sus resoluciones la primera vez y elige
                                       # la de más detalle con la placa entera a la vista (queda en
                                       # "modos_qr"); [ancho, alto]: fija; "normal": no cambiarla
    "resoluciones_qr_candidatas": [[1280, 960], [1600, 1200], [2048, 1536], [2592, 1944]],
                                       # solo 4:3: las panorámicas (16:9) recortan la imagen y se descartan
    "modos_qr": {},                    # resolución elegida para el QR, por cámara (bórrala para volver a probar)
    "qr_zoom": True,                   # leer solo la esquina de la placa donde va el QR (y mostrarla ampliada)
    "qr_zoom_vista": True,             # mostrar en el video la esquina ampliada mientras se lee
    "qr_zona_placa": [0.02, 0.02, 0.21, 0.21], # zona del QR en la placa (fracción de ancho y alto): 4 a 46 mm
    "plantilla_placa": {"mm": [220, 220], "esquina_recortada": "superior derecha", "recorte_mm": 10},
    "qr_modulo_mm": 1.03,              # tamaño de un módulo del QR impreso (25,8 mm / 25 módulos)
    "qr_px_por_modulo_min": 2.2,       # con menos detalle se avisa que la cámara no alcanza a leerlo
    "qr_presupuesto_ms": 200,          # tiempo máximo por intento de lectura del QR
    # --- Escanear con el celular (F5): página temporal en la red local; ver qr_celular.py ---
    "celular_puerto": 8765,
    # --- Celular como webcam (DroidCam, Iriun, Cámaras conectadas de Windows 11) solo para F3 ---
    "camara_qr": "",                   # parte del nombre de esa cámara; vacío = F3 usa la del módulo
    "camara_qr_resolucion": [1280, 720],
    "depurar_vision": True,            # imprime cada 10 s los fps de la cámara y del proceso
    # --- Selección, pinza y narración (Persona B, interaccion.py) ---
    # Radios en mm de la placa, pinza relativa al largo de la palma, histéresis, tiempos de
    # gracia y de estabilidad... Los valores y su explicación están en interaccion.DEFECTO.
    # (Los radios en px y 'distancia_pinza' de arriba solo se usan con el formato clásico.)
    **DEFECTO_INTERACCION,
}

TIMEOUT_DESCARGA = 15                  # segundos
MAX_BYTES = 2_000_000                  # tamaño máximo del archivo de datos


def cargar_config():
    cfg = dict(CONFIG_DEFECTO)
    try:
        with open(ARCHIVO_CONFIG, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    AJUSTES_CARGA["textos_cortos"] = bool(cfg.get("textos_cortos", True))
    if cfg.get("qr_url_base"):
        AJUSTES_CARGA["qr_url_base"] = str(cfg["qr_url_base"])
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
PISTAS_VOZ_ES = ("es-", "es_", "spanish", "español", "espanol", "castellano", "helena", "sabina",
                 "laura", "pablo", "raul", "raúl", "elvira", "dalia", "jorge", "alvaro", "álvaro",
                 "paulina", "monica", "mónica")
PISTAS_VOZ_ESPANA = ("es-es", "spain", "españa", "espana", "helena", "laura", "pablo", "elvira", "alvaro", "álvaro")
CATEGORIA_ONECORE = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"


def _lcid_es_espanol(texto):
    """'C0A;40A' u '80A' (atributo Language de SAPI, en hexadecimal) → ¿es español?
    El idioma principal son los 10 bits bajos del LCID: 0x0A = español."""
    for parte in str(texto or "").replace(",", ";").split(";"):
        try:
            if int(parte.strip(), 16) & 0x3FF == 0x0A:
                return True
        except ValueError:
            pass
    return False


def _parece_voz_espanol(*textos):
    t = " ".join(str(x or "") for x in textos).lower()
    return any(p in t for p in PISTAS_VOZ_ES)


def _voces_onecore():
    """Voces 'OneCore' de Windows 10/11: [(nombre, idioma, token)].
    Son las que se instalan en Configuración > Hora e idioma > Voz. pyttsx3 no las
    lista (solo ve las voces de escritorio de SAPI5), pero SAPI puede usarlas si se
    le pasa el token de esta otra categoría del registro. En un Windows en inglés
    suele ser la única forma de llegar a la voz en español."""
    res = []
    try:
        import comtypes.client
        cat = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
        cat.SetId(CATEGORIA_ONECORE, False)
        tokens = cat.EnumerateTokens()
        for i in range(tokens.Count):
            t = tokens.Item(i)
            try:
                idioma = t.GetAttribute("Language")
            except Exception:
                idioma = ""
            res.append((t.GetDescription(), idioma, t))
    except Exception as e:
        print("Aviso: no se pudieron leer las voces OneCore de Windows:", e)
    return res


class Voz:
    """Síntesis de voz en segundo plano con pyttsx3.

    - decir(): encola texto sin bloquear.
    - detener(): corta el audio en curso y vacía la cola.
    - Los mensajes de la interfaz (esencial=False) se pueden silenciar con F9
      para quien ya usa un lector de pantalla; la narración de la gráfica y lo
      que se pide explícitamente (F1, F4) es esencial=True y nunca se silencia.
    - Con depurar=True imprime en la consola cada frase que dice: sirve para
      distinguir "el programa no la envió" de "el motor de voz no la sonó".
    - Elige sola una voz en español aunque la voz predeterminada del equipo esté
      en otro idioma: primero las voces de escritorio (SAPI5) y, en Windows, también
      las OneCore. 'voz_preferida' (clave "voz" de la configuración) fuerza una voz
      por parte de su nombre, p. ej. "Sabina". Al terminar de elegir se activa
      'voz_lista'; 'voz_en_espanol' dice si se encontró una.
    """

    def __init__(self, velocidad=200, depurar=True, voz_preferida=""):
        self.cola = queue.Queue()
        self.hablando = threading.Event()
        self._detener = threading.Event()
        self._terminada = threading.Event()
        self.interfaz_activa = True
        self.disponible = False
        self.velocidad = velocidad
        self.depurar = depurar
        self.voz_preferida = str(voz_preferida or "").strip()
        self.voz_lista = threading.Event()
        self.nombre_voz = ""
        self.voz_en_espanol = None
        self._origen_voz = "sapi"
        self._token_defecto = None
        try:
            import pyttsx3  # noqa: F401
            self.disponible = True
        except ImportError:
            print("Aviso: falta 'pyttsx3' (pip install pyttsx3). Sin voz.")
            self.voz_lista.set()
        if self.disponible:
            threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _sapi(engine):
        """Objeto SAPI.SpVoice interno de pyttsx3 (solo Windows), o None."""
        return getattr(getattr(getattr(engine, "proxy", None), "_driver", None), "_tts", None)

    def _elegir_voz(self, engine):
        candidatos = []                     # (nombre, es_español, origen, id_o_token, idioma)
        try:
            for v in engine.getProperty("voices") or []:
                idiomas = [l.decode("utf-8", "ignore") if isinstance(l, bytes) else str(l)
                           for l in (getattr(v, "languages", None) or [])]
                es = (any(l.lower().lstrip("\x05").startswith("es") for l in idiomas)
                      or _parece_voz_espanol(v.id, v.name))
                candidatos.append((v.name or v.id, es, "sapi", v.id, " ".join(idiomas)))
        except Exception as e:
            print("Aviso: no se pudo listar las voces:", e)
        if sys.platform.startswith("win"):
            for nombre, idioma, token in _voces_onecore():
                es = _lcid_es_espanol(idioma) or _parece_voz_espanol(nombre)
                candidatos.append((nombre, es, "onecore", token, idioma))

        print("Voces encontradas en este equipo:")
        for c in candidatos:
            print(f"  [{c[2]}] {c[0]}  ({c[4] or 'idioma desconocido'})" + ("   <- español" if c[1] else ""))

        elegido = None
        pref = self.voz_preferida.lower()
        if pref:
            elegido = next((c for c in candidatos if pref in c[0].lower()), None)
            if elegido is None:
                print(f"Aviso: no hay ninguna voz que contenga «{self.voz_preferida}»; busco una en español.")
        if elegido is None:
            es = [c for c in candidatos if c[1]]
            # Primero las de escritorio (más fiables con pyttsx3); entre ellas, las latinoamericanas.
            es.sort(key=lambda c: (c[2] != "sapi", any(p in f"{c[0]} {c[4]}".lower() for p in PISTAS_VOZ_ESPANA)
                                   or str(c[4]).strip().lower().startswith("c0a")))
            elegido = es[0] if es else None

        tts = self._sapi(engine)
        if tts is not None:
            try:
                self._token_defecto = tts.Voice
            except Exception:
                pass
        if elegido is None:
            try:
                actual = engine.getProperty("voice")
                self.nombre_voz = next((c[0] for c in candidatos if c[3] == actual), str(actual))
            except Exception:
                self.nombre_voz = "predeterminada"
            self.voz_en_espanol = False
            print("Aviso: no hay ninguna voz en español instalada; uso:", self.nombre_voz)
            return
        if elegido[2] == "sapi":
            engine.setProperty("voice", elegido[3])
        else:
            if tts is None:
                raise RuntimeError("no encuentro el objeto SAPI de pyttsx3 para usar la voz OneCore")
            tts.Voice = elegido[3]
        self._origen_voz = elegido[2]
        self.nombre_voz, self.voz_en_espanol = elegido[0], elegido[1]
        print(f"Voz elegida: {self.nombre_voz} ({'OneCore' if elegido[2] == 'onecore' else 'escritorio'}).")

    def _volver_a_voz_defecto(self, engine):
        """Si una voz OneCore no funciona con SAPI en este equipo, se vuelve a la
        voz de escritorio: mejor una voz en otro idioma que ninguna."""
        tts = self._sapi(engine)
        if tts is None or self._token_defecto is None:
            return False
        try:
            tts.Voice = self._token_defecto
            engine.setProperty("rate", self.velocidad)
        except Exception:
            return False
        self._origen_voz, self.voz_en_espanol = "sapi", False
        self.nombre_voz = "predeterminada del sistema"
        print("Aviso: la voz OneCore falló; vuelvo a la voz de escritorio predeterminada. "
              "Puedes elegir otra voz con la clave \"voz\" de config_exploracion.json.")
        return True

    def _worker(self):
        import pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty("volume", 1.0)
            engine.connect("finished-utterance", lambda n, c: self._terminada.set())
        except Exception as e:
            print("Aviso: no se pudo iniciar el motor de voz:", e)
            self.disponible = False
            self.voz_lista.set()
            return
        try:
            self._elegir_voz(engine)
        except Exception as e:
            print("Aviso: no se pudo elegir la voz en español:", e)
            self.voz_en_espanol = False
        finally:
            try:
                engine.setProperty("rate", self.velocidad)   # tras elegir: la velocidad depende de la voz
            except Exception:
                pass
            self.voz_lista.set()

        while True:
            texto = self.cola.get()
            if texto is None:
                break
            self._detener.clear()
            self._terminada.clear()
            for intento in (1, 2):
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
                    break
                except Exception as e:
                    print("Aviso: error en el motor de audio:", e)
                    try:
                        engine.endLoop()
                    except Exception:
                        pass
                    if not (intento == 1 and self._origen_voz == "onecore" and self._volver_a_voz_defecto(engine)):
                        break
            # Si ya espera otra frase (p. ej. la que interrumpió a esta), se sigue "hablando":
            # así nadie cree que la voz quedó libre y encola encima.
            if self.cola.empty():
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


# Prioridades de selección (PRIORIDAD, PRIORIDAD_RESTO): ver interaccion.py.
TIPOS_SEGMENTO = ("curva", "tramo", "eje", "linea")
MAX_ELEMENTOS = 3000

ESQUINAS_HABLADAS = {
    "superior izquierda": "arriba a la izquierda",
    "superior derecha": "arriba a la derecha",
    "inferior izquierda": "abajo a la izquierda",
    "inferior derecha": "abajo a la derecha",
}

# Bordes de la placa (vision.Registro.bordes_fuera) → (cómo se nombra, hacia dónde moverla).
BORDES_HABLADOS = {"superior": ("de arriba", "hacia abajo"), "inferior": ("de abajo", "hacia arriba"),
                   "izquierdo": ("izquierdo", "hacia la derecha"), "derecho": ("derecho", "hacia la izquierda")}


def texto_bordes(bordes):
    """Aviso hablado cuando la placa se sale de la imagen de la cámara."""
    bordes = [b for b in bordes if b in BORDES_HABLADOS]
    if not bordes:
        return ""
    if {"superior", "inferior"} <= set(bordes) or {"izquierdo", "derecho"} <= set(bordes):
        return ("La placa no cabe entera en la imagen de la cámara: hay que subir la cámara o alejarla "
                "un poco, o pedir ayuda para colocarla.")
    nombres = [BORDES_HABLADOS[b][0] for b in bordes]
    hacia = " y ".join(BORDES_HABLADOS[b][1] for b in bordes)
    if len(nombres) == 1:
        return f"El borde {nombres[0]} de la placa queda fuera de la cámara: desliza la placa un poco {hacia}."
    return (f"Los bordes {' y '.join(nombres)} de la placa quedan fuera de la cámara: desliza la placa "
            f"un poco {hacia}.")


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


# Textos de las marcas y categorías de los ejes: lo que importa (el número, la categoría) va
# PRIMERO. Con "Eje vertical (Unidades vendidas (miles)): 20." la frase empieza igual que la del
# propio eje, y al mover el dedo la siguiente frase la corta antes de llegar al número: parecía
# que las marcas nunca se anunciaban.
_RE_MARCA = re.compile(r"^\s*eje\s+(vertical|horizontal|x|y)\s*(\(.*\))?\s*:\s*(.+?)\s*\.?\s*$", re.I)
_RE_NUMERO = re.compile(r"^[-+−]?\d[\d.,\s]*%?$")
_RE_CATEGORIA = re.compile(r"^\s*categor[ií]a\s*,?\s*abreviad[oa]\s+con\s+la\s+letra\s+(\S+?)\s*:\s*(.+?)\s*\.?\s*$", re.I)


def texto_para_voz(tipo, texto):
    """Reordena los textos de marcas ("Eje vertical (...): 20." → "20. Marca del eje vertical.")
    y de categorías abreviadas ("Categoría, abreviado con la letra A: Ene." → "Ene. Letra A.").
    Los demás textos no cambian. Se desactiva con "textos_cortos": false (configuración)."""
    if tipo != "texto" or not texto or not AJUSTES_CARGA.get("textos_cortos", True):
        return texto
    m = _RE_MARCA.match(texto)
    if m and (m.group(2) or _RE_NUMERO.match(m.group(3))):     # "(título)" o un número: es una marca
        eje = {"x": "horizontal", "y": "vertical"}.get(m.group(1).lower(), m.group(1).lower())
        return f"{m.group(3)}. Marca del eje {eje}."
    m = _RE_CATEGORIA.match(texto)
    if m:
        return f"{m.group(2)}. Letra {m.group(1)}."
    return texto


# Ajustes que usan las funciones de carga (las llama también medir_metricas.py); cargar_config()
# los actualiza con la configuración.
AJUSTES_CARGA = {"textos_cortos": True, "qr_url_base": "https://tactonauta.onrender.com/q/"}


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
        texto = texto_para_voz(tipo, texto)
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
    # QR corto: solo el código de la lámina (10 caracteres caben en un QR versión 1 con la máxima
    # corrección de errores; con el enlace completo hace falta uno de 25-29 módulos, más finos).
    if re.fullmatch(r"[0-9A-Za-z]{6,16}", url) and re.search(r"\d", url):
        url = AJUSTES_CARGA.get("qr_url_base", "https://tactonauta.onrender.com/q/").rstrip("/") + "/" + url.upper()
    # Un QR puede llevar el enlace sin "https://" (más corto: cabe con más corrección de errores).
    if re.match(r"^[a-z0-9-]+(\.[a-z0-9-]+)+\.[a-z]{2,}(/|$)", url, re.I):
        url = "https://" + url
    if not re.match(r"^https?://", url, re.I):
        raise ErrorDatos("Eso no parece un enlace web ni el código de una lámina. El enlace debe empezar con http o https; el código son las letras y números del final del enlace, por ejemplo 54D7E01D5C.")
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
# distancia, distancia_segmento, distancia_rect y distancia_elemento: ver interaccion.py.


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


def escalar_elementos(datos, registro):
    """Copia de los elementos con las coordenadas en píxeles del frame (sin redondear).
    'registro' es un vision.Registro o, como antes, el rectángulo 'hoja' (x, y, ancho, alto)."""
    if not isinstance(registro, Registro):
        registro = Registro.desde_rect(registro, datos.ancho_ref, datos.alto_ref, datos.estirar)
    res = []
    for e in datos.elementos:
        p = dict(e)
        p["x1"], p["y1"] = registro.placa_a_imagen_f(e["x1"], e["y1"])
        p["x2"], p["y2"] = registro.placa_a_imagen_f(e["x2"], e["y2"])
        res.append(p)
    return res


# ============================================================
# EXPLORADOR (gesto de pinza + narración)
# ============================================================
class Explorador(Interaccion):
    """Interaccion (interaccion.py, en mm) con su propio detector de la mano.

    La aplicación ya no lo usa (MotorVision crea una Interaccion). Se mantiene para la
    prueba 'tareas' de medir_metricas.py, que hace mano → actualizar → dibujo con procesar().
    """

    def __init__(self, voz, cfg, avisar=None, detector=None):
        super().__init__(voz, cfg, avisar)
        self.detector = detector if detector is not None else DetectorMano(cfg, avisar)
        self._superposicion = None

    def configurar(self, datos, registro):
        super().configurar(datos, registro)
        if getattr(self, "_superposicion", None) is not None:
            self._superposicion.configurar(self.datos, self.registro)

    def cerrar(self):
        if self.detector is not None:
            self.detector.cerrar()

    def procesar(self, frame):
        """Mano → actualizar → dibujo en un solo paso. Devuelve (frame, texto_visible|None)."""
        h, w = frame.shape[:2]
        manos = self.detector.detectar(frame)
        puntos = manos[0] if manos else None
        sel = self.actualizar(muestra_de_mano(puntos, w, h, self.registro))
        if self._superposicion is None:
            self._superposicion = Superposicion()
            self._superposicion.configurar(self.datos, self.registro)
        actual = self._superposicion.por_id.get(sel.elemento_id) if sel.elemento_id else None
        visible = actual["corto"] if actual is not None else None
        dedos = dedos_px(puntos, w, h)[:2] if puntos is not None else None
        self._superposicion.dibujar(frame, sel.elemento_id, dedos, sel.pinza, visible)
        return frame, visible


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
                 "Pulsa F2 para pegar el enlace de la gráfica, F3 para escanear el código QR impreso con la "
                 "cámara, o F5 para escanearlo con tu celular. "
                 "Cuando la gráfica esté cargada, Enter empieza. Escape retrocede. "
                 "F1 repite estas instrucciones.")

AYUDA_CAMARA = ("Teclas de la cámara. "
                "F4: escuchar los botones de la parte inferior y cuál está seleccionado. "
                "Flechas izquierda y derecha: moverte entre esos botones. "
                "Enter: empezar la exploración, cuando esté disponible. "
                "R: repetir la descripción de la gráfica. A: autodetectar la placa. "
                "C: calibrar la placa marcando sus cuatro esquinas, con ayuda de una persona vidente. "
                "O: cambiar la orientación de la imagen, si la cámara la muestra girada o en espejo. "
                "Más y menos: agrandar o reducir el video. "
                "Para escuchar la gráfica, junta las puntas del índice y del dedo medio sobre un elemento; "
                "un instante con los dedos separados no corta el audio, y al separarlos del todo se calla. "
                "Si sigues una curva y llegas a un cruce, se sigue la misma curva y se avisa con qué serie se cruza. "
                "Escape: retroceder; si estás calibrando cancela, y si no, vuelve al inicio y apaga la cámara. "
                "F9: activar o silenciar la voz de la interfaz.")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = cargar_config()
        self.voz = Voz(self.cfg["velocidad_voz"], self.cfg.get("depurar_voz", True), self.cfg.get("voz", ""))
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
        self.bind_all("<F5>", lambda e: self._opcion_celular())
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
        self.after(400, self._comprobar_voz, 0)

    def _comprobar_voz(self, intentos):
        """Cuando la voz termina de elegirse: si no hay ninguna en español, lo dice
        (en español y una frase en inglés, por si la voz que suena es inglesa)."""
        if not self.voz.voz_lista.is_set():
            if intentos < 40:
                self.after(250, self._comprobar_voz, intentos + 1)
            return
        if not self.voz.disponible:
            self._mensaje("No se pudo iniciar la voz del sistema. Revisa que pyttsx3 esté instalado.",
                          "error", hablar=False)
            return
        if self.voz.voz_en_espanol is False:
            texto = (f"No encontré una voz en español en este equipo; uso la voz {self.voz.nombre_voz}. "
                     "Para instalarla: Configuración, Hora e idioma, Voz, Agregar voces, y elige Español. "
                     "Después vuelve a abrir el programa.")
            self._mensaje(texto, "error", hablar=False)
            self.voz.decir(texto + " No Spanish voice is installed: open Settings, Time and language, Speech, "
                           "Add voices, and add Spanish.", esencial=True)

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

        self.btn_celular = self._boton(
            t, "3 · Escanear el código con el celular   (F5)", self._opcion_celular,
            "Opción tres. Escanear el código con la cámara de tu celular. Pulsa Enter para empezar.")
        self.btn_celular.pack(fill="x", padx=20, pady=6)
        self.panel_celular = tk.Frame(t, bg=BG)          # se muestra al elegir la opción 3
        self.lbl_qr_celular = tk.Label(self.panel_celular, bg=BG)
        self.lbl_qr_celular.pack(side="left", padx=(0, 16))
        self.lbl_url_celular = self._etiqueta(self.panel_celular, "", F_GRANDE, ACENTO)
        self.lbl_url_celular.pack(side="left", anchor="n")
        self.btn_empezar_cel = self._boton(self.panel_celular, "Empezar", self._empezar_desde_enlace,
                                           "Botón Empezar. Pulsa Enter para abrir la cámara y explorar la gráfica.",
                                           state="disabled")
        self.btn_empezar_cel.pack(side="left", anchor="s", padx=16)
        fila_cel = tk.Frame(self.panel_celular, bg=BG)
        fila_cel.pack(side="left", anchor="s")
        self.btn_otra_ip = self._boton(fila_cel, "Otra dirección", self._otra_direccion_celular,
                                       "Botón otra dirección. Muestra la dirección de otra red de la computadora, "
                                       "por ejemplo la de su zona con cobertura.")
        self.btn_otra_ip.pack(fill="x", pady=(0, 6))
        self.btn_firewall = self._boton(fila_cel, "Permitir en el firewall", self._permitir_firewall,
                                        "Botón permitir en el firewall. Deja que el celular se conecte a la "
                                        "computadora. Windows pedirá permiso de administrador.")
        self.btn_firewall.pack(fill="x")
        self.servidor_celular = None
        self._foto_qr_celular = None
        self._i_dir_cel = 0
        self._id_vigia_cel = None

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
        self.btn_autodet = self._boton(barra, "Autodetectar placa (A)", self._autodetectar,
                                       "Botón autodetectar la placa en la imagen.")
        self.btn_autodet.pack(side="left", padx=4)
        self.btn_calibrar = self._boton(barra, "Calibrar (C)", self._alternar_calibracion,
                                        "Botón calibrar la placa con cuatro clics. Requiere ayuda de una persona vidente.")
        self.btn_calibrar.pack(side="left", padx=4)
        self.btn_orientacion = self._boton(barra, "Orientación (O)", self._siguiente_orientacion,
                                           "Botón orientación. Gira o pone en espejo la imagen de la cámara.")
        self.btn_orientacion.pack(side="left", padx=4)
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
                    (self.btn_autodet, "Autodetectar placa", "A"),
                    (self.btn_calibrar, "Calibrar placa", "C"),
                    (self.btn_orientacion, "Orientación de la cámara", "O"),
                    (self.btn_volver, "Volver al inicio", "Escape")]
        lista = [(self.btn_enlace, "Pegar enlace de la gráfica", "F2")]
        if self.panel_enlace.winfo_ismapped():
            lista += [(self.btn_pegar, "Pegar del portapapeles", None),
                      (self.btn_verificar, "Verificar y descargar", None),
                      (self.btn_empezar, "Empezar", "Enter")]
        lista += [(self.btn_qr, "Escanear código QR", "F3"),
                  (self.btn_celular, "Escanear con el celular", "F5")]
        if self.panel_celular.winfo_ismapped():
            lista += [(self.btn_empezar_cel, "Empezar", "Enter"),
                      (self.btn_otra_ip, "Otra dirección", None),
                      (self.btn_firewall, "Permitir en el firewall", None)]
        lista += [(self.btn_config, "Abrir carpeta de configuración", None)]
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
        if self.panel_celular.winfo_ismapped():        # cerrar el escaneo con el celular
            self._cerrar_celular()
            self._dar_foco(self.btn_celular)
            self._t_escape = 0.0
            self._mensaje("Escaneo con el celular cerrado. Estás en la pantalla de bienvenida.")
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
            if (str(self.btn_empezar.cget("state")) == "normal"
                    or str(self.btn_empezar_cel.cget("state")) == "normal"):
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

    # ---------- opción 3: escanear con el celular ----------
    def _opcion_celular(self):
        """F5: abre una página temporal en la red local; el celular fotografía el QR y la envía.
        La cámara del módulo no se enciende hasta empezar la exploración."""
        if self.notebook.index("current") != 0:
            return
        try:
            from qr_celular import ServidorCelular, direccion_hablada, imagen_qr_ppm
        except ImportError:
            self._mensaje("Falta el archivo qr_celular.py junto al programa.", "error")
            return
        nuevo = self.servidor_celular is None or not self.servidor_celular.activo
        if nuevo:
            self.servidor_celular = ServidorCelular(self.cfg, lambda t: self.cola.put(("qr_celular", t)))
            try:
                self.servidor_celular.iniciar()
            except OSError as e:
                self.servidor_celular = None
                self._mensaje(f"No pude abrir la página para el celular: {e}", "error")
                return
            self._i_dir_cel = 0
            self._mostrar_direccion_celular()
            threading.Thread(target=self._revisar_firewall, daemon=True).start()
            if self._id_vigia_cel is not None:
                self.after_cancel(self._id_vigia_cel)
            self._id_vigia_cel = self.after(60000, self._vigilar_celular)
        url = self.servidor_celular.url_de(self._i_dir_cel)
        self.fase = "celular"
        self.btn_empezar_cel.configure(state="disabled")
        multiples = len(self.servidor_celular.direcciones) > 1
        self.btn_otra_ip.configure(state="normal" if multiples else "disabled")
        self.btn_firewall.configure(state="normal" if sys.platform.startswith("win") else "disabled")
        self.panel_celular.pack(fill="x", padx=20, pady=6, after=self.btn_celular)
        self._dar_foco(self.btn_celular)
        self._mensaje("Escaneo con el celular. Conecta el celular a la misma red wifi que la computadora y abre "
                      "en su navegador esta dirección: " + direccion_hablada(url) + ". "
                      "También puedes apuntar la cámara del celular a la pantalla de la computadora: muestra un "
                      "código con la dirección. En la página, toca Tomar foto del código, y fotografía la esquina "
                      "superior izquierda de la placa a unos 15 centímetros. F5 repite la dirección; Escape cancela."
                      + (" Si la página no carga, prueba el botón Otra dirección." if multiples else ""))

    def _mostrar_direccion_celular(self):
        from qr_celular import imagen_qr_ppm
        srv = self.servidor_celular
        url = srv.url_de(self._i_dir_cel)
        d = srv.direcciones[self._i_dir_cel % len(srv.direcciones)]
        red = {"wifi": "red wifi", "cable": "red por cable", "hotspot": "zona con cobertura de esta computadora",
               "otra": "red"}[d["tipo"]]
        self._foto_qr_celular = tk.PhotoImage(data=imagen_qr_ppm(url), format="PPM")
        self.lbl_qr_celular.configure(image=self._foto_qr_celular)
        otras = [srv.url_de(i).replace("http://", "") for i in range(len(srv.direcciones)) if i != self._i_dir_cel % len(srv.direcciones)]
        self.lbl_url_celular.configure(text=f"En el celular abre ({red}):\n" + url.replace("http://", "")
                                       + ("\n\nOtras: " + " · ".join(otras) if otras else ""))
        return url, red

    def _otra_direccion_celular(self):
        from qr_celular import direccion_hablada
        if self.servidor_celular is None:
            return
        self._i_dir_cel += 1
        url, red = self._mostrar_direccion_celular()
        self._mensaje(f"Dirección de la {red}: {direccion_hablada(url)}. El celular debe estar conectado a esa red.")

    def _revisar_firewall(self):
        """En otro hilo: si Windows no tiene la regla para el celular, se avisa (por la cola)."""
        try:
            from qr_celular import estado_firewall
            est = estado_firewall()
        except Exception:
            traceback.print_exc()
            return
        if est is not None:
            print("Firewall:", "regla para el celular presente" if est["regla"] else "SIN regla para el celular",
                  "· tipo de red:", ", ".join(est["perfiles"]) or "?")
            self.cola.put(("firewall", est))

    def _permitir_firewall(self):
        from qr_celular import permitir_firewall
        if permitir_firewall():
            self._mensaje("Windows va a pedir permiso de administrador para dejar que el celular se conecte. "
                          "Acepta el aviso. Después vuelve a abrir la página en el celular.")
            self.after(8000, lambda: threading.Thread(target=self._revisar_firewall, daemon=True).start())
        else:
            self._mensaje("No pude pedir el permiso del firewall. Un administrador puede permitir los puertos "
                          "8765 a 8770 en el Firewall de Windows.", "error")

    def _vigilar_celular(self):
        """Un minuto sin que ningún celular abra la página: se explica qué hacer."""
        self._id_vigia_cel = None
        srv = self.servidor_celular
        if srv is None or self.fase != "celular" or srv.visitas:
            return
        self._mensaje("El celular todavía no ha abierto la página. Revisa tres cosas. Uno: pulsa el botón "
                      "Permitir en el firewall y acepta el aviso de Windows. Dos: muchas redes de universidad "
                      "no dejan que los equipos se vean entre sí; en ese caso activa la zona con cobertura del "
                      "celular, conecta la computadora a ella y vuelve a pulsar F5. Tres: también puedes leer el "
                      "código con la cámara del celular y escribir en la opción uno, pegar enlace, el código que "
                      "aparece al final del enlace, por ejemplo cinco cuatro D siete.", "error")

    def _cerrar_celular(self):
        if self._id_vigia_cel is not None:
            self.after_cancel(self._id_vigia_cel)
            self._id_vigia_cel = None
        if self.servidor_celular is not None:
            self.servidor_celular.detener()
            self.servidor_celular = None
        self.panel_celular.pack_forget()
        if self.fase in ("celular", "celular_cargando"):
            self.fase = "inicio"

    def _qr_celular(self, texto):
        if self.fase != "celular":
            return
        self.fase = "celular_cargando"
        self._mensaje("El celular leyó el código. Verificando el enlace y descargando.")
        self._lanzar_carga(texto, "celular")

    # ---------- cámara ----------
    def _abrir_camara(self, modo):
        self._detener_camara()
        if self.servidor_celular is not None:
            self._cerrar_celular()
        cfg = self.cfg
        self._motor_celular = bool(modo == "qr" and str(self.cfg.get("camara_qr") or "").strip())
        if self._motor_celular:
            # F3 con el celular como webcam: otra cámara, solo para leer el QR. Se lee la imagen
            # completa (el celular va en la mano) y sin probar resoluciones.
            cfg = dict(self.cfg)
            cfg.update({"nombre_camara": self.cfg["camara_qr"], "indice_camara": None,
                        "resolucion": self.cfg.get("camara_qr_resolucion") or [1280, 720],
                        "resolucion_qr": "normal", "qr_zoom": False, "exposicion": None,
                        "voltear": "ninguno"})
        self.motor = MotorVision(self.cola, self.voz, cfg, crear_interaccion=self._crear_interaccion)
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

    def _crear_interaccion(self, detector, avisar):
        """Lo que decide qué se anuncia (Persona B, interaccion.py). MotorVision lo llama en su
        hilo y le pasa la MuestraMano de cada frame; el detector de la mano es de MotorVision."""
        return Interaccion(self.voz, self.cfg, avisar)

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
        if getattr(self, "_motor_celular", False):   # el QR se leyó con el celular: ahora la cámara del módulo
            self._abrir_camara("explorar")
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

    def _siguiente_orientacion(self):
        """Tecla O: pasa a la siguiente orientación (primero los giros, luego los espejos)."""
        if self.motor is None:
            self._mensaje("La orientación se cambia con la cámara encendida.", "error")
            return
        orden = list(ORIENTACIONES)
        pendiente = self.motor._orient_pedida
        actual = pendiente[0] if pendiente else self.motor.orientacion
        actual = actual if actual in orden else "ninguno"
        self.motor.set_orientacion(orden[(orden.index(actual) + 1) % len(orden)])

    def _autodetectar(self):
        if self.motor is None or self.fase != "explorando":
            self._mensaje("La autodetección funciona durante la exploración.", "error")
            return
        self._mensaje("Buscando la placa en la imagen.")
        self.motor.pedir_autodetectar()

    # ---------- calibración manual (con ayuda de una persona vidente) ----------
    ESQUINAS_CLIC = ("arriba a la izquierda", "arriba a la derecha", "abajo a la derecha", "abajo a la izquierda")

    def _alternar_calibracion(self):
        if self.motor is None or self.fase != "explorando":
            self._mensaje("La calibración funciona durante la exploración.", "error")
            return
        self.calibrando = not self.calibrando
        self._clics = []
        self.motor.set_clics([])
        if self.calibrando:
            self._mensaje("Calibración de la placa. Una persona vidente debe hacer clic en el video sobre las "
                          "cuatro esquinas de la placa, en este orden: arriba a la izquierda, arriba a la derecha, "
                          "abajo a la derecha y abajo a la izquierda. En la esquina recortada, haz clic donde se "
                          "juntarían los dos bordes.")
        else:
            self._mensaje("Calibración cancelada.")

    def _clic_video(self, e):
        """Cuatro clics = cuatro esquinas de la placa → registro con homografía."""
        if not self.calibrando or self.motor is None:
            return
        self._clics.append((e.x / self._escala, e.y / self._escala))
        self.motor.set_clics(self._clics)
        n = len(self._clics)
        if n < 4:
            self.voz.decir(f"Esquina {n} marcada. Ahora, {self.ESQUINAS_CLIC[n]}.", interrumpir=True)
            return
        esq, self.calibrando, self._clics = list(self._clics), False, []
        self.motor.set_clics([])
        area = 0.5 * abs(sum(esq[i][0] * esq[(i + 1) % 4][1] - esq[(i + 1) % 4][0] * esq[i][1] for i in range(4)))
        if area < 400:
            self._mensaje("La zona marcada es demasiado pequeña. Inténtalo de nuevo con C.", "error")
            return
        # Se ordenan por su posición en la imagen: da igual en qué orden se hicieron los clics.
        self.motor.set_esquinas(ordenar_esquinas(esq).tolist(), "clics")   # el motor responde con un evento "placa"

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
        elif k == "o":
            self._siguiente_orientacion()
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
        elif tipo == "firewall":
            est = ev[1]
            if est.get("regla"):
                self.btn_firewall.configure(text="Firewall: permitido")
            elif self.fase == "celular":
                publica = any(p.lower().startswith("public") for p in est.get("perfiles", []))
                self.btn_firewall.configure(text="Permitir en el firewall")
                self._mensaje("Aviso: el Firewall de Windows puede estar bloqueando al celular"
                              + (", porque esta red wifi está marcada como pública" if publica else "")
                              + ". Si la página no carga, pulsa el botón Permitir en el firewall.",
                              hablar=True, interrumpir=False)
        elif tipo == "qr_celular":
            self._qr_celular(ev[1])
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
        elif tipo == "orientacion":
            modo, origen, camara, hoja, _cambia = ev[1:6]
            esquinas = ev[6] if len(ev) > 6 else None
            self.cfg.setdefault("orientaciones", {})[camara] = modo
            self.cfg["hoja"] = [int(v) for v in hoja]
            if esquinas is not None:
                self.cfg["esquinas_placa_px"] = esquinas
            guardar_config(self.cfg)
            if origen == "qr":
                self._mensaje(f"Enderecé la imagen de la cámara usando el código QR. Corrección: "
                              f"{ORIENTACIONES[modo]}. Queda guardada para esta cámara.", "ok", interrumpir=False)
            else:
                self._mensaje(f"Orientación de la cámara: {ORIENTACIONES[modo]}. Guardada para esta cámara. "
                              "Pulsa O otra vez para probar la siguiente.")
        elif tipo == "placa":
            self._evento_placa(ev[1])
        elif tipo == "borde":
            self._mensaje(texto_bordes(ev[1]), "error", interrumpir=False)
        elif tipo == "modo_qr":                       # resolución elegida para leer el QR con esta cámara
            self.cfg.setdefault("modos_qr", {})[ev[1]] = ev[2]
            guardar_config(self.cfg)

    # ---------- registro de la placa (mensajes de vision.MotorVision) ----------
    def _guardar_registro(self, info):
        if info.get("esquinas") is not None:
            self.cfg["esquinas_placa_px"] = info["esquinas"]
        self.cfg["hoja"] = [int(v) for v in info["hoja"]]
        self.cfg["registro_origen"] = info["origen"] if info["origen"] in ("clics", "qr") else "autodeteccion"
        err = info.get("error_mm")
        self.cfg["registro_error_mm"] = None if err is None else round(float(err), 2)
        self.cfg["registro_camara"], self.cfg["registro_resolucion"] = info["camara"], info["resolucion"]
        guardar_config(self.cfg)

    def _evento_placa(self, info):
        """Resultado de la autodetección (A), de la comprobación al empezar, del seguimiento
        de la placa, de la calibración con clics o de la corrección con el QR."""
        origen = info["origen"]
        if not info["ok"]:
            if origen == "autodeteccion":
                self._mensaje("No pude detectar la placa. Comprueba que se vea entera y sin la mano encima, "
                              "mejora la iluminación, o usa calibrar con ayuda.", "error")
            elif origen == "inicio" and info.get("guardada_otra_camara"):
                self._mensaje("No pude comprobar dónde está la placa, y la calibración guardada es de otra cámara "
                              "o de otra resolución. Pulsa A para buscar la placa, o C para calibrar con ayuda.",
                              "error", interrumpir=False)
            elif origen == "inicio":
                self._mensaje("No pude comprobar dónde está la placa; uso la última calibración. Si lo que oyes "
                              "no coincide con lo que tocas, pulsa A.", interrumpir=False)
            return
        self._guardar_registro(info)
        partes, tipo_msg = [], "ok"
        esperada, vista = info.get("esquina_esperada"), info.get("esquina_vista")
        aviso_esquina = ""
        if esperada and vista and info.get("giro"):
            aviso_esquina = (f"Atención: la esquina recortada parece estar {ESQUINAS_HABLADAS[vista]}, pero debe ir "
                             f"{ESQUINAS_HABLADAS[esperada]}. Gira la placa y pulsa A otra vez. Si la placa ya está "
                             "bien colocada, la cámara muestra la imagen girada: pulsa O hasta que la esquina quede "
                             "en su sitio.")
        d = info.get("desplazamiento_mm")
        if origen == "clics":
            partes.append("Placa calibrada con sus cuatro esquinas y guardada.")
        elif origen == "autodeteccion":
            pct = info.get("pct")
            partes.append("Placa detectada" + (f"; ocupa el {pct:.0f} por ciento de la imagen." if pct else "."))
            if aviso_esquina:
                partes.append(aviso_esquina)
            elif esperada and vista:
                partes.append(f"La esquina recortada se ve {ESQUINAS_HABLADAS[vista]}, como debe ser.")
            elif esperada:
                partes.append(f"No distingo la esquina recortada; comprueba que vaya {ESQUINAS_HABLADAS[esperada]}.")
            partes.append("Si puedes, pide a una persona vidente que lo confirme; si no es correcto, usa calibrar.")
        elif origen == "inicio":
            if d is not None and d >= 5:
                partes.append(f"La placa no está donde se calibró la última vez (se movió unos {fmt_num(round(d))} "
                              "milímetros); ajusté su posición.")
            if aviso_esquina:
                partes.append(aviso_esquina)
        elif origen == "seguimiento":
            partes.append("La placa se movió; ajusté su posición.")
        elif origen == "qr":
            partes.append("No distingo bien el borde de la placa; ajusté su posición con el código QR"
                          + (f" (se había movido unos {fmt_num(round(d))} milímetros)." if d else "."))
        if info.get("bordes"):
            partes.append(texto_bordes(info["bordes"]))
            tipo_msg = "error"
        if aviso_esquina:
            tipo_msg = "error"
        if partes:
            self._mensaje(" ".join(partes), tipo_msg, interrumpir=origen in ("autodeteccion", "clics"))

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
        if origen == "celular":
            if self.fase != "celular_cargando":
                return
            if self.servidor_celular is not None:     # ya no hace falta: se apaga la página
                self.servidor_celular.detener()
                self.servidor_celular = None
            self.fase = "inicio"
            self.btn_empezar_cel.configure(state="normal")
            self._dar_foco(self.btn_empezar_cel)
            self._mensaje(f"La información de la gráfica está cargada. {resumen} "
                          "Pulsa Enter para abrir la cámara y empezar la exploración.", "ok")
            return
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
        if origen == "celular":
            if self.fase == "celular_cargando":
                self.fase = "celular"
                self._mensaje(f"{msg} Toma otra foto desde el celular.", "error")
            return
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
        if self.servidor_celular is not None:
            self.servidor_celular.detener()
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
