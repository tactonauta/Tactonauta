#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDIR MÉTRICAS  ·  Explorador de gráficas impresas con textura
==============================================================

Programa de evaluación para explorador_graficas_gui_qr.py. Va en la MISMA
carpeta que ese archivo y reutiliza su código (lectura del CSV, calibración,
autodetección de la placa, QR, seguimiento de la mano y elección del elemento
bajo el dedo), así que mide el sistema tal cual funciona.

Modos
-----
  geometria  Sin cámara. Analiza el CSV de la gráfica con los radios (en mm) de
             config_exploracion.json: separación entre series, cruces,
             zonas ambiguas y conflictos de prioridad. Guarda un mapa PNG.
             Además simula una yema que recorre cada serie con temblor y compara
             la interacción completa (suavizado, estabilidad, serie seguida en los
             cruces) con la búsqueda sin estado de cada frame.
  camara     Batería con la cámara y la placa impresa. Pruebas (en este orden):
               hoja       autodetección de la placa (4 esquinas, vision.detectar_placa),
                          contraste con el fondo, desfase respecto de la calibración
                          guardada y recolocación
               qr         tasa y tiempo de lectura del QR
               registro   error del registro en uso (esquinas o 'hoja') y del
                          autodetectado: una persona vidente hace clic en puntos
                          de referencia de la placa
               mano       detección de la mano (recortada a la placa, como la
                          aplicación), fps de la cámara y del proceso, latencia
               objetivos  se anuncia un elemento, se pone el dedo encima con la
                          pinza y se mide si el sistema elige ese elemento (con la
                          misma Interaccion y el mismo registro que la aplicación; se
                          compara con la búsqueda sin suavizado ni estabilidad)
               falsos     pinzas falsas (dedos separados) y cortes (pinza cerrada),
                          con la pinza nueva (palma, histéresis y gracia) y con el
                          umbral fijo en px de antes, para comparar
  tareas     Exploración real con voz (como la aplicación) y preguntas de
             comprensión cronometradas; opcional: cuestionario SUS.
  imagen     Analiza una foto o captura guardada (placa, contraste, QR) y,
             con --registro, mide el error de calibración con clics.
  fps        Sin placa ni voz. Prueba la cámara con MJPG y sin él, con exposición automática
             y fija (-4 a -8), mide los fps reales y el brillo, y recomienda qué poner en
             "mjpg" y "exposicion". Con --ajustes abre el panel del driver (DirectShow), donde
             se puede apagar la "compensación de poca luz", que baja la cámara a 15 fps.
  reporte    Junta todos los resultados en metricas/reporte_metricas.md y
             metricas/resumen_metricas.csv, una columna por placa.

Ejemplos
--------
  python medir_metricas.py geometria --placa morado_g2 --datos https://tactonauta.onrender.com/q/XXXX
  python medir_metricas.py camara --placa morado_g2 --color oscuro --grafica 2
  python medir_metricas.py camara --placa salmon_g2 --color claro --grafica 2 --pruebas objetivos,falsos
  python medir_metricas.py tareas --placa turquesa_g3 --sus
  python medir_metricas.py reporte
  python medir_metricas.py fps                     (¿por qué la cámara da 15 fps?)
  python medir_metricas.py camara --placa salmon_g2 --camara 1      (forzar la cámara número 1)

Teclas (ventana "Medir metricas")
--------------------------------
  Espacio  empezar / medir        R  repetir la indicación en voz alta
  S        saltar este paso       Z  deshacer el último clic (registro)
  Esc      terminar la prueba     1 / 0 / 2  respuesta correcta / incorrecta / parcial (tareas)

Todo lo que se pide también se dice en voz alta (pyttsx3, la misma voz que la
aplicación) y se escribe en la consola.
"""

import argparse
import csv
import datetime
import glob
import json
import math
import os
import queue
import random
import sys
import time
import traceback

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
if AQUI not in sys.path:
    sys.path.insert(0, AQUI)
CARPETA_METRICAS = os.path.join(AQUI, "metricas")
VERSION = "1.2 · 2026-11-05"      # 1.1: pruebas de la Persona A con vision.Registro (4 esquinas)
                                  # 1.2: pruebas de la Persona B con interaccion.Interaccion (mm)
TIPOS_DATOS = ("punto", "curva", "tramo", "barra")
VENTANA = "Medir metricas"

eg = None  # módulo explorador_graficas_gui_qr (se importa al arrancar)


def importar_explorador():
    global eg
    try:
        import explorador_graficas_gui_qr as modulo
    except ModuleNotFoundError as e:
        if e.name == "explorador_graficas_gui_qr":
            sys.exit("No encuentro explorador_graficas_gui_qr.py. Copia medir_metricas.py en la "
                     "misma carpeta que el explorador y vuelve a intentarlo.")
        raise
    eg = modulo
    return modulo


# ============================================================
# UTILIDADES
# ============================================================
def marca_tiempo():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def slug(s):
    s = eg.sin_acentos(str(s)).lower() if eg else str(s).lower()
    return "".join(c if c.isalnum() else "_" for c in s).strip("_") or "placa"


def mediana(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else None


def percentil(v, q):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, q)) if v else None


def proporcion(k, n):
    """Proporción con su intervalo de confianza del 95 % (Wilson)."""
    if not n:
        return {"k": 0, "n": 0, "p": None, "ic95": [None, None]}
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": int(k), "n": int(n), "p": p, "ic95": [max(0.0, centro - margen), min(1.0, centro + margen)]}


def limpiar(obj):
    """Para JSON: redondea floats, convierte numpy y tuplas. Las claves que empiezan por
    '_' son de uso interno (p. ej. la detección completa) y no se guardan."""
    if isinstance(obj, dict):
        return {str(k): limpiar(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [limpiar(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else round(v, 4)
    if isinstance(obj, np.ndarray):
        return limpiar(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------- color ----------------
def bgr_a_lab(bgr):
    a = np.asarray(bgr, dtype=np.float32)
    forma = a.shape[:-1]
    lab = cv2.cvtColor((a.reshape(-1, 1, 3) / 255.0).astype(np.float32), cv2.COLOR_BGR2Lab)
    return lab.reshape(forma + (3,)).astype(float)


def delta_e2000(lab1, lab2):
    """Diferencia de color CIEDE2000 (vectorizada)."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, float), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, float), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lpm, Cpm, hs = (L1 + L2) / 2, (C1p + C2p) / 2, h1p + h2p
    hpm = np.where(C1p * C2p == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-((hpm - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def gris_de(bgr):
    b, g, r = np.moveaxis(np.asarray(bgr, float), -1, 0)
    return 0.114 * b + 0.587 * g + 0.299 * r


# ---------------- geometría de la placa ----------------
def escala_px_mm(datos, hoja):
    """(px por mm en x, px por mm en y) de la calibración, como la usa el explorador."""
    hx, hy, hw, hh = hoja
    if datos.estirar:
        return hw / datos.ancho_ref, hh / datos.alto_ref
    e = min(hw / datos.ancho_ref, hh / datos.alto_ref)
    return e, e


def directa(datos, hoja):
    """mm de la placa (lienzo) → px del frame, sin redondear (misma fórmula que eg.transformacion)."""
    hx, hy, hw, hh = hoja
    if datos.estirar:
        sx, sy = hw / datos.ancho_ref, hh / datos.alto_ref
        return lambda x, y: (hx + x * sx, hy + y * sy)
    e = min(hw / datos.ancho_ref, hh / datos.alto_ref)
    mx, my = hx + (hw - datos.ancho_ref * e) / 2, hy + (hh - datos.alto_ref * e) / 2
    return lambda x, y: (mx + x * e, my + y * e)


def inversa(datos, hoja):
    """px del frame → mm de la placa, según la calibración."""
    hx, hy, hw, hh = hoja
    if datos.estirar:
        sx, sy = hw / datos.ancho_ref, hh / datos.alto_ref
        return lambda u, v: ((u - hx) / sx, (v - hy) / sy)
    e = min(hw / datos.ancho_ref, hh / datos.alto_ref)
    mx, my = hx + (hw - datos.ancho_ref * e) / 2, hy + (hh - datos.alto_ref * e) / 2
    return lambda u, v: ((u - mx) / e, (v - my) / e)


def selector(cfg, datos, hoja):
    """Una Interaccion sin voz: la MISMA lógica de radios, prioridades, pinza y estabilidad
    que la aplicación. 'hoja' es el rectángulo (x, y, ancho, alto) o un vision.Registro.
    Trabaja en las unidades del lienzo (mm en el formato handtracking)."""
    ex = eg.Interaccion(None, cfg)
    ex.configurar(datos, hoja)
    return ex


def campo_distancia(U, V, e):
    """Distancia de cada punto (U, V) al elemento e (vectorizada; misma geometría que eg.distancia_elemento)."""
    x1, y1, x2, y2 = (float(e[k]) for k in ("x1", "y1", "x2", "y2"))
    if e["forma"] == "punto":
        return np.hypot(U - x1, V - y1)
    if e["forma"] == "rect":
        dx = np.maximum(np.maximum(min(x1, x2) - U, 0), U - max(x1, x2))
        dy = np.maximum(np.maximum(min(y1, y2) - V, 0), V - max(y1, y2))
        return np.hypot(dx, dy)
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return np.hypot(U - x1, V - y1)
    t = np.clip(((U - x1) * dx + (V - y1) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(U - (x1 + t * dx), V - (y1 + t * dy))


def punto_cercano(x, y, e):
    """Punto del elemento e más cercano a (x, y)."""
    x1, y1, x2, y2 = e["x1"], e["y1"], e["x2"], e["y2"]
    if e["forma"] == "punto":
        return x1, y1
    if e["forma"] == "rect":
        return min(max(x, min(x1, x2)), max(x1, x2)), min(max(y, min(y1, y2)), max(y1, y2))
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return x1, y1
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
    return x1 + t * dx, y1 + t * dy


def centro(e):
    return (e["x1"] + e["x2"]) / 2, (e["y1"] + e["y2"]) / 2


def interseccion(a, b):
    p, r = np.array([a["x1"], a["y1"]], float), np.array([a["x2"] - a["x1"], a["y2"] - a["y1"]], float)
    q, s = np.array([b["x1"], b["y1"]], float), np.array([b["x2"] - b["x1"], b["y2"] - b["y1"]], float)
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-9:
        return None
    qp = q - p
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return p + t * r
    return None


def caja_grafico(datos):
    """Rectángulo (mm) del área del gráfico: el que forman los ejes; si no hay, el de los datos."""
    ejes = [e for e in datos.elementos if e["tipo"] == "eje"]
    base = ejes or [e for e in datos.elementos if e["tipo"] in TIPOS_DATOS]
    xs = [v for e in base for v in (e["x1"], e["x2"])]
    ys = [v for e in base for v in (e["y1"], e["y2"])]
    return min(xs), min(ys), max(xs), max(ys)


def cruces(datos):
    """Puntos donde se cruzan curvas de series distintas (mm)."""
    curvas = [e for e in datos.elementos if e["tipo"] in ("curva", "tramo") and e["forma"] == "segmento"]
    res = []
    for i, a in enumerate(curvas):
        for b in curvas[i + 1:]:
            if not a["serie"] or a["serie"] == b["serie"]:
                continue
            p = interseccion(a, b)
            if p is None:
                continue
            if any(math.hypot(p[0] - c["x"], p[1] - c["y"]) < 1.0 and {a["serie"], b["serie"]} == set(c["series"])
                   for c in res):
                continue
            res.append({"x": float(p[0]), "y": float(p[1]), "series": [a["serie"], b["serie"]],
                        "curvas": [a["id"], b["id"]]})
    return res


def huecos(datos):
    """Pares de puntos consecutivos de una serie que no están unidos por una curva (saltos o datos faltantes)."""
    res = []
    curvas = [e for e in datos.elementos if e["tipo"] in ("curva", "tramo")]

    def unidos(p, q):
        for c in curvas:
            if c["serie"] != p["serie"]:
                continue
            for (ax, ay, bx, by) in ((c["x1"], c["y1"], c["x2"], c["y2"]), (c["x2"], c["y2"], c["x1"], c["y1"])):
                if math.hypot(ax - p["x1"], ay - p["y1"]) < 0.6 and math.hypot(bx - q["x1"], by - q["y1"]) < 0.6:
                    return True
        return False

    for s in series_de(datos):
        pts = [e for e in datos.elementos if e["tipo"] == "punto" and e["serie"] == s]
        for p, q in zip(pts, pts[1:]):
            if not unidos(p, q):
                res.append({"serie": s, "desde": p["id"], "hasta": q["id"],
                            "clase": "salto vertical" if abs(q["x1"] - p["x1"]) < 1.0 else "tramo sin datos",
                            "x1": p["x1"], "y1": p["y1"], "x2": q["x1"], "y2": q["y1"]})
    return res


def series_de(datos):
    s = list(datos.series)
    for e in datos.elementos:
        if e["tipo"] in TIPOS_DATOS and e["serie"] and e["serie"] not in s:
            s.append(e["serie"])
    return s


# ============================================================
# ANÁLISIS DE GEOMETRÍA (sin cámara)
# ============================================================
PALETA = [(60, 160, 60), (30, 120, 230), (200, 60, 200), (220, 140, 30), (40, 200, 200), (150, 90, 230)]


def analizar_geometria(cfg, datos, hoja, paso_mm=0.5, imagen=None, sigmas=(2, 4, 6), muestras_mc=200):
    # La selección (interaccion.Interaccion) trabaja en el lienzo: en el formato handtracking,
    # mm de la placa. La 'hoja' solo da la escala en px para informar.
    ex = selector(cfg, datos, hoja)
    els_px, els_mm = ex.elementos, datos.elementos      # (els_px: nombre histórico; están en el lienzo)
    sx, sy = escala_px_mm(datos, hoja)
    mm_por_px = 2.0 / (sx + sy)
    mm_por_unidad = 1.0 if ex._en_mm else mm_por_px / ex._escala
    W, H = datos.ancho_ref, datos.alto_ref
    xs = np.arange(paso_mm / 2, W, paso_mm)
    ys = np.arange(paso_mm / 2, H, paso_mm)
    GX, GY = np.meshgrid(xs, ys)
    U, V = GX, GY

    series = series_de(datos)
    s_idx = {s: i for i, s in enumerate(series)}
    # Misma regla que Interaccion._buscar (sin elemento actual ni serie seguida):
    # datos por (prioridad, distancia); ejes/textos/leyendas por (distancia, prioridad);
    # estos ganan solo si están más cerca que el dato por más de margen_resto_mm.
    dat_p = np.full(U.shape, 99, int)
    dat_d = np.full(U.shape, np.inf)
    dat_i = np.full(U.shape, -1, int)
    res_p = np.full(U.shape, 99, int)
    res_d = np.full(U.shape, np.inf)
    res_i = np.full(U.shape, -1, int)
    cerca = np.zeros((len(series),) + U.shape, bool)
    cerc_d = np.full(U.shape, np.inf)
    cerc_s = np.full(U.shape, -1, int)
    for i, e in enumerate(els_px):
        d = campo_distancia(U, V, e)
        m = d <= ex._radio(e)
        p = eg.PRIORIDAD.get(e["tipo"], eg.PRIORIDAD_RESTO)
        if p < eg.PRIORIDAD_RESTO:
            pk, dk = ex.clave_dato(e, d)               # (prioridad, distancia efectiva), como _buscar
            gana = m & ((pk < dat_p) | ((pk == dat_p) & (dk < dat_d)))
            dat_p = np.where(gana, pk, dat_p)
            dat_d = np.where(gana, dk, dat_d)
            dat_i = np.where(gana, i, dat_i)
        else:
            gana = m & ((d < res_d) | ((d == res_d) & (p < res_p)))
            res_p = np.where(gana, p, res_p)
            res_d = np.where(gana, d, res_d)
            res_i = np.where(gana, i, res_i)
        if e["tipo"] in TIPOS_DATOS and e["serie"] in s_idx:
            k = s_idx[e["serie"]]
            cerca[k] |= m
            mas = m & (d < cerc_d)
            cerc_d = np.where(mas, d, cerc_d)
            cerc_s = np.where(mas, k, cerc_s)
    usa_resto = (res_i >= 0) & ((dat_i < 0) | (res_d + ex._mm("margen_resto_mm") < dat_d))
    mejor_i = np.where(usa_resto, res_i, dat_i)
    # Datos que quedarían tapados por un eje/texto con la regla antigua (todo por prioridad)
    # o al revés: cuánto cambia la corrección de prioridad de los ejes.
    cambia_prioridad = (res_i >= 0) & (dat_i >= 0) & usa_resto

    # Comprobación: el mapa vectorizado elige lo mismo que la función real de la aplicación.
    rng = np.random.default_rng(1)
    muestras = rng.integers(0, U.size, size=min(400, U.size))
    coinciden = 0
    for m in muestras:
        r, c = divmod(int(m), U.shape[1])
        real = ex._buscar(float(U[r, c]), float(V[r, c]))
        coinciden += (real["id"] if real else None) == (els_px[mejor_i[r, c]]["id"] if mejor_i[r, c] >= 0 else None)

    x0, y0, x1, y1 = caja_grafico(datos)
    en_caja = (GX >= x0) & (GX <= x1) & (GY >= y0) & (GY <= y1)
    serie_elem = np.array([s_idx.get(e["serie"], -1) if e["tipo"] in TIPOS_DATOS else -1 for e in els_px] + [-1])
    gana_serie = serie_elem[mejor_i]                       # -1 si gana algo que no es un dato (o nada)
    banda = cerca.any(axis=0)
    n_series = cerca.sum(axis=0)
    ambigua = n_series >= 2
    conflicto = (gana_serie >= 0) & (cerc_s >= 0) & (gana_serie != cerc_s)

    def frac(mask, base):
        nb = int(base.sum())
        return float((mask & base).sum()) / nb if nb else None

    puntos = [e for e in els_px if e["tipo"] == "punto"]
    radios_pt = [ex._radio(e) * mm_por_unidad for e in puntos]
    if ex._en_mm:
        r_cfg = {"punto": ex._mm("radio_punto_mm"), "linea": ex._mm("radio_linea_mm"),
                 "eje": ex._mm("radio_eje_mm"), "texto": ex._mm("radio_texto_mm")}
    else:
        r_cfg = {"punto": ex._px("radio_punto", 20), "linea": ex._px("radio_linea", 15),
                 "eje": ex._px("radio_eje", 20), "texto": ex._px("radio_texto", 12)}
    pts_mm = [e for e in els_mm if e["tipo"] == "punto"]
    sep_series, sep_misma, par = math.inf, math.inf, None
    for i, a in enumerate(pts_mm):
        for b in pts_mm[i + 1:]:
            d = math.hypot(a["x1"] - b["x1"], a["y1"] - b["y1"])
            if a["serie"] != b["serie"]:
                if d < sep_series:
                    sep_series, par = d, (a["id"], b["id"])
            else:
                sep_misma = min(sep_misma, d)
    lista_cruces = cruces(datos)
    lista_huecos = huecos(datos)

    res = {
        "elementos": {t: sum(1 for e in els_mm if e["tipo"] == t)
                      for t in sorted({e["tipo"] for e in els_mm})},
        "series": series,
        "placa_mm": [W, H],
        "hoja_px": list(hoja),
        "px_por_mm": [sx, sy],
        "anisotropia_calibracion": sx / sy,
        "radios_mm": {
            "punto_config": r_cfg["punto"] * mm_por_unidad,
            "punto_efectivo_min": min(radios_pt) if radios_pt else None,
            "punto_efectivo_mediana": mediana(radios_pt),
            "puntos_con_radio_reducido": sum(1 for e in puntos if ex._radio(e) < r_cfg["punto"] - 1e-9),
            "linea": r_cfg["linea"] * mm_por_unidad,
            "eje": r_cfg["eje"] * mm_por_unidad,
            "texto": r_cfg["texto"] * mm_por_unidad,
            "margen_resto": ex._mm("margen_resto_mm") * mm_por_unidad,
            # La pinza se mide contra la palma; esto es el umbral de reserva (sin palma fiable).
            "distancia_pinza": float(ex.c("pinza_cerrar_mm")) if ex._en_mm
            else ex._px("distancia_pinza", 35) * mm_por_unidad,
        },
        "pinza_palma": {"cerrar": float(ex.c("pinza_cerrar_palma")), "abrir": float(ex.c("pinza_abrir_palma"))},
        "eje_o_texto_gana_a_dato_pct_banda": None if not banda.any() else 100 * frac(cambia_prioridad, banda & en_caja),
        "sep_min_puntos_series_mm": None if math.isinf(sep_series) else sep_series,
        "par_mas_cercano": par,
        "sep_min_puntos_misma_serie_mm": None if math.isinf(sep_misma) else sep_misma,
        "cruces": lista_cruces,
        "n_cruces": len(lista_cruces),
        "huecos": lista_huecos,
        "n_huecos": len(lista_huecos),
        "area_grafico_mm": [x0, y0, x1, y1],
        "cobertura_pct": 100 * frac(mejor_i >= 0, en_caja),
        "banda_datos_pct": 100 * frac(banda, en_caja),
        "ambigua_pct_banda": None if not banda.any() else 100 * frac(ambigua, banda & en_caja),
        "conflicto_pct_banda": None if not banda.any() else 100 * frac(conflicto, banda & en_caja),
        "verificacion_selector_pct": 100.0 * coinciden / len(muestras),
    }

    # Acierto esperado: el dedo cae alrededor del objetivo con un error gaussiano de sigma mm.
    objetivos = elegir_objetivos(datos, n=10 ** 6)
    s_de = {e["id"]: e["serie"] for e in els_mm}
    t_de = {e["id"]: e["tipo"] for e in els_mm}
    rng = np.random.default_rng(5)
    esperado = {}
    for sigma in sigmas:
        por_zona = {}
        for o in objetivos:
            dx = rng.normal(0, sigma, (muestras_mc, 2))
            ok = ok_serie = 0
            for ddx, ddy in dx:
                e = ex._buscar(o["x"] + ddx, o["y"] + ddy)
                if e is None:
                    continue
                ok += e["id"] == o["id"]
                ok_serie += (e["id"] == o["id"]) or (o["tipo"] in TIPOS_DATOS and t_de[e["id"]] in TIPOS_DATOS
                                                     and s_de[e["id"]] == o["serie"])
            claves = {o["zona"]} | ({f'{o["zona"]}_{o["clase"]}'} if o["zona"] in ("aislado", "cruce") else set())
            for clave in claves:
                z = por_zona.setdefault(clave, [0, 0, 0])
                z[0] += ok
                z[1] += ok_serie
                z[2] += muestras_mc
        esperado[f"{sigma:g}"] = {zona: {"estricto_pct": 100.0 * a / n, "serie_pct": 100.0 * b / n}
                                  for zona, (a, b, n) in por_zona.items()}
        for zona, v in esperado[f"{sigma:g}"].items():
            res[f"esperado_{sigma:g}mm_{zona}_pct"] = v["estricto_pct"]
            res[f"esperado_{sigma:g}mm_{zona}_serie_pct"] = v["serie_pct"]
    res["acierto_esperado"] = esperado
    res["acierto_esperado_nota"] = ("% de pinzas que anunciarían el objetivo si la yema cae con un error "
                                    "gaussiano de sigma mm alrededor de él (calibración perfecta)")

    if ex._en_mm:
        trazos = simular_trazos(cfg, datos, hoja)
        res["trazos"] = trazos
        for modo, zonas in trazos.get("modos", {}).items():
            for zona, v in zonas.items():
                for clave in ("serie_pct", "con_dato_pct", "cambios_s", "anuncios_s"):
                    if v.get(clave) is not None:
                        res[f"trazo_{modo}_{zona}_{clave}"] = v[clave]

    if imagen:
        k = 2                                             # px de imagen por celda
        hh, ww = U.shape
        img = np.full((hh, ww, 3), 255, np.uint8)
        for i, e in enumerate(els_px):
            sel = mejor_i == i
            if not sel.any():
                continue
            if e["tipo"] in TIPOS_DATOS and e["serie"] in s_idx:
                c = np.array(PALETA[s_idx[e["serie"]] % len(PALETA)], float)
                c = c * (0.65 if e["tipo"] == "punto" else 1.0) + (0 if e["tipo"] == "punto" else 0.45 * (255 - c))
            elif e["tipo"] == "eje":
                c = np.array((170, 170, 170), float)
            else:
                c = np.array((215, 215, 215), float)
            img[sel] = c.astype(np.uint8)
        rayas = ((np.add.outer(np.arange(hh), np.arange(ww)) % 6) < 2)
        img[ambigua & rayas] = (0, 0, 0)
        img = cv2.resize(img, (ww * k, hh * k), interpolation=cv2.INTER_NEAREST)
        banda = 22 + 18 * (len(series) + 1)
        img = cv2.copyMakeBorder(img, banda, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        g = lambda x, y: (int(round(x / paso_mm * k)), banda + int(round(y / paso_mm * k)))
        for e in els_mm:
            if e["forma"] == "punto":
                cv2.circle(img, g(e["x1"], e["y1"]), 3, (0, 0, 0), -1)
            elif e["forma"] == "rect":
                cv2.rectangle(img, g(e["x1"], e["y1"]), g(e["x2"], e["y2"]), (90, 90, 90), 1)
            else:
                cv2.line(img, g(e["x1"], e["y1"]), g(e["x2"], e["y2"]), (0, 0, 0), 1, cv2.LINE_AA)
        for c in lista_cruces:
            cv2.circle(img, g(c["x"], c["y"]), 9, (0, 0, 255), 2)
        cv2.rectangle(img, g(0, 0), g(W - 0.01, H - 0.01), (0, 0, 0), 1)
        leyenda = [f"{eg.texto_cv(s)[:40]}" for s in series]
        for j, s in enumerate(leyenda):
            y = 18 + 18 * j
            cv2.rectangle(img, (8, y - 11), (22, y + 2), PALETA[j % len(PALETA)], -1)
            cv2.putText(img, s, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        y = 18 + 18 * len(leyenda)
        cv2.putText(img, "rayado = 2+ series al alcance; circulo rojo = cruce", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(imagen, img)
        res["imagen"] = os.path.basename(imagen)
    return res


def simular_trazos(cfg, datos, hoja, ruido_mm=1.5, fps=15.0, vel_mm_s=40.0, repeticiones=4, semilla=3):
    """Sin cámara: una yema virtual recorre cada serie de punto a punto con la pinza cerrada
    y un temblor gaussiano de 'ruido_mm', a 'fps' muestras por segundo. Compara:
      · completa    la Interaccion de la aplicación (suavizado, estabilidad, serie seguida)
      · sin_estado  la búsqueda de cada frame por separado, sin nada de eso
    en dos zonas: 'cruce' (a menos de 5 mm de un cruce con otra serie) y 'resto'.
      serie_pct     de los frames con un dato elegido, % que son de la serie recorrida
      con_dato_pct  % de frames con algún dato elegido
      cambios_s     cambios de elemento por segundo (parpadeo)
      anuncios_s    anuncios de voz por segundo (solo 'completa')"""
    rng = np.random.default_rng(semilla)
    cr = cruces(datos)
    ex = selector(cfg, datos, hoja)
    serie_de = {e["id"]: e["serie"] for e in ex.elementos if e["tipo"] in TIPOS_DATOS}
    modos = {m: {z: {"frames": 0, "dato": 0, "serie": 0, "cambios": 0, "anuncios": 0} for z in ("cruce", "resto")}
             for m in ("completa", "sin_estado")}
    trazadas = 0
    for s in series_de(datos):
        if not any(e["tipo"] in ("curva", "tramo") and e["serie"] == s for e in ex.elementos):
            continue                                   # sin curva no hay nada que recorrer
        pts = sorted((e for e in ex.elementos if e["tipo"] == "punto" and e["serie"] == s), key=lambda e: e["x1"])
        if len(pts) < 2:
            continue
        poli = np.array([(p["x1"], p["y1"]) for p in pts], float)
        acum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(poli, axis=0).T))]
        n = int(acum[-1] / (vel_mm_s / fps)) + 1
        arco = np.linspace(0.0, acum[-1], n)
        xs, ys = np.interp(arco, acum, poli[:, 0]), np.interp(arco, acum, poli[:, 1])
        zona = ["cruce" if any(s in c["series"] and math.hypot(x - c["x"], y - c["y"]) < 5.0 for c in cr) else "resto"
                for x, y in zip(xs, ys)]
        trazadas += 1
        for _rep in range(repeticiones):
            ex.reiniciar()
            ruido = rng.normal(0.0, ruido_mm, (n, 2))
            antes = {"completa": None, "sin_estado": None}
            anunciado = None
            for k in range(n):
                x, y = xs[k] + ruido[k, 0], ys[k] + ruido[k, 1]
                mu = eg.MuestraMano(k / fps, (x, y), (x + 3.0, y), 90.0, True)   # pinza cerrada
                sel = ex.actualizar(mu).elemento_id
                crudo = ex._buscar(x, y)
                for modo, ident in (("completa", sel), ("sin_estado", crudo["id"] if crudo else None)):
                    z = modos[modo][zona[k]]
                    z["frames"] += 1
                    if ident in serie_de:
                        z["dato"] += 1
                        z["serie"] += serie_de[ident] == s
                    z["cambios"] += k > 0 and ident != antes[modo]
                    antes[modo] = ident
                if ex.anterior is not None and ex.anterior != anunciado:
                    modos["completa"][zona[k]]["anuncios"] += 1
                anunciado = ex.anterior
    res = {"series_trazadas": trazadas, "ruido_mm": ruido_mm, "fps": fps, "vel_mm_s": vel_mm_s,
           "repeticiones": repeticiones, "modos": {}}
    for modo, zonas in modos.items():
        res["modos"][modo] = {}
        for zona, z in zonas.items():
            if not z["frames"]:
                continue
            seg = z["frames"] / fps
            res["modos"][modo][zona] = {
                "frames": z["frames"],
                "serie_pct": 100.0 * z["serie"] / z["dato"] if z["dato"] else None,
                "con_dato_pct": 100.0 * z["dato"] / z["frames"],
                "cambios_s": z["cambios"] / seg,
                "anuncios_s": z["anuncios"] / seg if modo == "completa" else None}
    return res


# ============================================================
# ANÁLISIS DE UNA IMAGEN (placa, contraste, QR)
# ============================================================
def poligono_px(registro_o_esquinas):
    """Contorno de la placa en px (int32) para dibujar o hacer máscaras."""
    e = registro_o_esquinas.esquinas_px() if hasattr(registro_o_esquinas, "esquinas_px") else registro_o_esquinas
    return np.round(np.asarray(e, float)).astype(np.int32).reshape(-1, 1, 2)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def contraste_regiones(frame, mascara_placa):
    """Placa (máscara) frente a un anillo de fondo alrededor: Δgris, ΔE2000, separabilidad de Otsu."""
    h, w = frame.shape[:2]
    m = (mascara_placa > 0).astype(np.uint8)
    dentro = cv2.erode(m, np.ones((9, 9), np.uint8))
    anillo = cv2.dilate(m, np.ones((41, 41), np.uint8)) - cv2.dilate(m, np.ones((11, 11), np.uint8))
    if dentro.sum() < 50 or anillo.sum() < 50:
        return None
    pd, pa = frame[dentro > 0].astype(float), frame[anillo > 0].astype(float)
    gd, ga = gris_de(pd), gris_de(pa)
    lab_d, lab_a = bgr_a_lab(np.median(pd, axis=0)), bgr_a_lab(np.median(pa, axis=0))
    todos = np.concatenate([gd, ga]).astype(np.uint8).reshape(-1, 1)
    umbral, _ = cv2.threshold(todos, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = todos.ravel().astype(float)
    c0, c1 = t[t <= umbral], t[t > umbral]
    var_t = t.var()
    eta = (len(c0) * len(c1) / len(t) ** 2 * (c0.mean() - c1.mean()) ** 2 / var_t
           if len(c0) and len(c1) and var_t > 0 else 0.0)
    return {"gris_placa": float(np.median(gd)), "gris_fondo": float(np.median(ga)),
            "delta_gris": float(abs(np.median(gd) - np.median(ga))),
            "delta_e": float(delta_e2000(lab_d, lab_a)),
            "lab_placa": lab_d.tolist(), "lab_fondo": lab_a.tolist(),
            "fondo_desv_gris": float(ga.std()), "separabilidad_otsu": float(eta)}


def analizar_placa(frame, datos=None, registro_ref=None):
    """Autodetección como la de la aplicación (vision.detectar_placa: 4 esquinas) y
    contraste placa–fondo. 'registro_ref' (el registro en uso) da la máscara si no se detecta."""
    h, w = frame.shape[:2]
    if registro_ref is not None and not hasattr(registro_ref, "esquinas_px") and datos is not None:
        registro_ref = eg.Registro.desde_rect(registro_ref, datos.ancho_ref, datos.alto_ref, datos.estirar)
    t0 = time.perf_counter()
    det = eg.detectar_placa(frame, datos)
    res = {"detectada": det is not None, "rect": list(det.rect) if det else None,
           "esquina": det.esquina_vista if det else None, "ms": 1000 * (time.perf_counter() - t0)}
    mascara = np.zeros((h, w), np.uint8)
    if det is not None:
        ang = det.rotacion_grados
        while ang > 45:
            ang -= 90
        while ang <= -45:
            ang += 90
        e = det.esquinas
        ancho = (np.hypot(*(e[1] - e[0])) + np.hypot(*(e[2] - e[3]))) / 2
        alto = (np.hypot(*(e[3] - e[0])) + np.hypot(*(e[2] - e[1]))) / 2
        res.update({"esquinas": np.round(e, 2).tolist(), "metodo": det.metodo, "confianza": det.confianza,
                    "reconstruida": det.reconstruida, "giro_corregido": det.giro, "residuo_px": det.residuo_px,
                    "error_estimado_mm": det.error_estimado_mm, "area_pct": det.area_pct,
                    "rotacion_grados": float(ang), "bordes_imagen": det.bordes_imagen,
                    "toca_borde": bool(det.bordes_imagen)})
        if datos is not None:
            bordes = det.registro(datos).bordes_fuera(w, h)
            res["bordes_fuera"] = bordes
            res["toca_borde"] = bool(bordes or det.bordes_imagen)
            res["aspecto_px_vs_mm"] = (ancho / alto) / (datos.ancho_ref / datos.alto_ref)
            esperada = datos.esquina_recortada
            if esperada:
                res["esquina_esperada"] = esperada
                res["esquina_ok"] = det.esquina_vista == esperada
        cv2.fillPoly(mascara, [poligono_px(e)], 255)
    elif registro_ref is not None:
        cv2.fillPoly(mascara, [poligono_px(registro_ref)], 255)
    if mascara.any():
        res["contraste"] = contraste_regiones(frame, mascara)
    res["_det"] = det                                  # para quien llama (no se guarda)
    return res


def analizar_qr(lector, frame, presupuesto_ms=None, zoom=None, orientacion="ninguno"):
    """Un intento de lectura. Con 'zoom' (vision.ZoomQR) se lee como la aplicación en F3:
    solo la esquina de la placa donde va el QR."""
    t0 = time.perf_counter()
    if zoom is not None:
        texto, poli = zoom.leer(frame, orientacion, presupuesto_ms)
    else:
        texto, poli = lector.leer(frame, presupuesto_ms=presupuesto_ms)
    res = {"leido": bool(texto), "ms": 1000 * (time.perf_counter() - t0), "texto": texto or None,
           "localizado": poli is not None, "resolucion": [frame.shape[1], frame.shape[0]]}
    if zoom is not None:
        u = zoom.ultima
        res["zoom_placa"] = bool(u.get("placa"))
        if u.get("placa"):
            res["px_por_mm"] = u.get("px_por_mm")
            res["px_por_modulo"] = u.get("px_por_modulo")
    if texto:
        res["polaridad"] = "invertida (claro sobre oscuro)" if lector.invertido else "normal (oscuro sobre claro)"
        res["orientacion_qr"] = lector.orientacion
        res["variante"] = "/".join(str(v) for v in lector.variante[1:]) if lector.variante else None
    if poli is not None:
        p = np.asarray(poli, dtype=np.float32).reshape(-1, 2)
        h, w = frame.shape[:2]
        area = abs(cv2.contourArea(p.reshape(-1, 1, 2)))
        res["lado_px"] = math.sqrt(area)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [p.astype(np.int32)], 255)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[m > 0].astype(float)
        if g.size > 20:
            lo, hi = np.percentile(g, [5, 95])
            res["contraste_michelson"] = float((hi - lo) / (hi + lo)) if hi + lo > 0 else 0.0
    return res


# ============================================================
# REGISTRO (error de la calibración con puntos de referencia)
# ============================================================
def puntos_referencia(datos, maximo=12):
    """Lugares de la placa fáciles de señalar en la imagen, con su posición en mm."""
    W, H = datos.ancho_ref, datos.alto_ref
    refs = []
    recortada = datos.esquina_recortada
    for nombre, x, y in (("esquina superior izquierda de la placa", 0, 0),
                         ("esquina superior derecha de la placa", W, 0),
                         ("esquina inferior izquierda de la placa", 0, H),
                         ("esquina inferior derecha de la placa", W, H)):
        if recortada and recortada in nombre:
            continue
        refs.append((nombre, x, y))
    for e in datos.elementos:
        if e["tipo"] != "eje":
            continue
        horizontal = abs(e["x2"] - e["x1"]) >= abs(e["y2"] - e["y1"])
        extremos = sorted([(e["x1"], e["y1"]), (e["x2"], e["y2"])], key=lambda p: p[0] if horizontal else -p[1])
        nombres = (("extremo izquierdo del eje horizontal", "extremo derecho del eje horizontal") if horizontal
                   else ("extremo inferior del eje vertical", "extremo superior del eje vertical"))
        for n, (x, y) in zip(nombres, extremos):
            if any(math.hypot(x - rx, y - ry) < 2 for _, rx, ry in refs):
                continue
            refs.append(("origen de los ejes" if "izquierdo" in n or "inferior" in n else n, x, y))
    qr = next((e for e in datos.elementos if e["id"].lower() == "qr"), None)
    if qr is not None and qr["forma"] == "rect":
        refs.append(("esquina superior izquierda del QR", min(qr["x1"], qr["x2"]), min(qr["y1"], qr["y2"])))
        refs.append(("esquina inferior derecha del QR", max(qr["x1"], qr["x2"]), max(qr["y1"], qr["y2"])))
    for s in series_de(datos):
        pts = sorted((e for e in datos.elementos if e["tipo"] == "punto" and e["serie"] == s), key=lambda e: e["x1"])
        for e, cual in ((pts[0], "primer"), (pts[-1], "último")) if pts else ():
            refs.append((f"{cual} punto de {s}", e["x1"], e["y1"]))
    vistos, final = [], []
    for r in refs:
        if all(math.hypot(r[1] - v[1], r[2] - v[2]) >= 2 for v in vistos):
            vistos.append(r)
            final.append(r)
    return final[:maximo]


def calcular_registro(datos, registro, refs, clics, registro_auto=None):
    """Error del registro en uso ('registro': vision.Registro, o el rectángulo 'hoja' como
    antes) y, si se da, del autodetectado, frente a los clics de una persona vidente."""
    def como_registro(r):
        if r is None or hasattr(r, "imagen_a_placa"):
            return r
        return eg.Registro.desde_rect(r, datos.ancho_ref, datos.alto_ref, datos.estirar)

    registro, registro_auto = como_registro(registro), como_registro(registro_auto)
    hoja = registro.caja_px()
    pares = [(r, c) for r, c in zip(refs, clics) if c is not None]
    if len(pares) < 3:
        return {"error": "Hacen falta al menos 3 puntos marcados.", "n_puntos": len(pares)}
    mm = np.array([[r[1], r[2]] for r, _ in pares], np.float64)
    px = np.array([c for _, c in pares], np.float64)
    if len(pares) >= 4:
        Hm, _ = cv2.findHomography(mm, px, 0)
        modelo = "homografía"
    else:
        A, _ = cv2.estimateAffine2D(mm, px)
        Hm = np.vstack([A, [0, 0, 1]])
        modelo = "afín"
    if Hm is None:
        return {"error": "Los puntos marcados no permiten calcular la posición de la placa."}
    Hinv = np.linalg.inv(Hm)

    def aplicar(M, pts):
        p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ M.T
        return p[:, :2] / p[:, 2:3]

    pred_px = registro.placa_a_imagen_arr([[r[1], r[2]] for r, _ in pares])
    pred_mm_real = aplicar(Hinv, pred_px)                 # dónde cae en la placa lo que el sistema dibuja
    err_mm = np.hypot(*(pred_mm_real - mm).T)
    err_px = np.hypot(*(pred_px - px).T)
    residuo = np.hypot(*(aplicar(Hinv, px) - mm).T)       # precisión de los clics (ajuste del modelo)

    x0, y0, x1, y1 = caja_grafico(datos)

    def mapa_error(reg):
        res = {}
        for nombre, (a, b, c, d) in (("grafico", (x0, y0, x1, y1)), ("placa", (0, 0, datos.ancho_ref, datos.alto_ref))):
            gx, gy = np.meshgrid(np.linspace(a, c, 25), np.linspace(b, d, 25))
            reales = np.c_[gx.ravel(), gy.ravel()]
            px_reales = aplicar(Hm, reales)
            cree = reg.imagen_a_placa_arr(px_reales)
            e = np.hypot(*(cree - reales).T)
            res[nombre] = {"mediana_mm": float(np.median(e)), "p95_mm": float(np.percentile(e, 95)),
                           "max_mm": float(e.max())}
            if nombre == "grafico":
                dvec = (reales - cree).mean(axis=0)       # cuánto hay que mover lo dibujado para que coincida
                res["desplazamiento_mm"] = [float(dvec[0]), float(dvec[1])]
                ex_ = np.polyfit(reales[:, 0], cree[:, 0], 1)[0]
                ey_ = np.polyfit(reales[:, 1], cree[:, 1], 1)[0]
                res["escala_x"], res["escala_y"] = float(1 / ex_), float(1 / ey_)
        return res

    vx = aplicar(Hm, [[x0, y0], [x1, y0]])
    rot = math.degrees(math.atan2(vx[1][1] - vx[0][1], vx[1][0] - vx[0][0]))
    esquinas = aplicar(Hm, [[0, 0], [datos.ancho_ref, 0], [datos.ancho_ref, datos.alto_ref], [0, datos.alto_ref]])
    rx0, ry0 = esquinas.min(axis=0)
    rx1, ry1 = esquinas.max(axis=0)
    rect_real = [float(rx0), float(ry0), float(rx1 - rx0), float(ry1 - ry0)]
    res = {
        "modelo": modelo, "n_puntos": len(pares),
        "error_mediana_mm": float(np.median(err_mm)), "error_max_puntos_mm": float(err_mm.max()),
        "error_mediana_px": float(np.median(err_px)),
        "residuo_clics_mm": float(np.sqrt(np.mean(residuo ** 2))),
        "rotacion_placa_grados": rot,
        "rect_placa_real_px": rect_real,
        "esquinas_placa_reales_px": esquinas.tolist(),
        "iou_calibracion": iou(rect_real, hoja),
        "registro_origen": registro.origen,
        "registro_error_estimado_mm": registro.error_estimado_mm,
        "calibracion": mapa_error(registro),
        "puntos": [{"nombre": r[0], "mm": [r[1], r[2]], "clic_px": list(c), "dibujado_px": list(p),
                    "error_mm": float(e), "residuo_mm": float(q)}
                   for (r, c), p, e, q in zip(pares, pred_px, err_mm, residuo)],
    }
    res["error_max_grafico_mm"] = res["calibracion"]["grafico"]["max_mm"]
    if registro_auto is not None:
        res["autodeteccion"] = mapa_error(registro_auto)
        res["iou_autodeteccion"] = iou(rect_real, registro_auto.caja_px())
        res["error_auto_mediana_grafico_mm"] = res["autodeteccion"]["grafico"]["mediana_mm"]
        res["error_auto_max_grafico_mm"] = res["autodeteccion"]["grafico"]["max_mm"]
        res["autodeteccion_error_estimado_mm"] = registro_auto.error_estimado_mm
    return res


# ============================================================
# PANTALLA, VOZ Y CÁMARA
# ============================================================
class Pantalla:
    """Ventana de OpenCV: muestra imágenes, lee teclas y clics."""

    def __init__(self, nombre=VENTANA):
        self.nombre, self.escala, self._clic = nombre, 1.0, None
        cv2.namedWindow(nombre, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(nombre, self._raton)

    def _raton(self, evento, x, y, _flags, _param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            self._clic = (x / self.escala, y / self.escala)

    def mostrar(self, img, escala=1.0):
        self.escala = escala
        if escala != 1.0:
            img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.nombre, img)

    def tecla(self, ms=1):
        k = cv2.waitKey(ms)
        return -1 if k < 0 else k & 0xFF

    def tomar_clic(self):
        c, self._clic = self._clic, None
        return c

    def cerrar(self):
        try:
            cv2.destroyWindow(self.nombre)
        except cv2.error:
            pass


def rotulo(img, lineas, abajo=False):
    """Banda de texto (ASCII) sobre la imagen."""
    texto = []
    for l in lineas:
        l = eg.texto_cv(str(l))
        while len(l) > 78:
            corte = l.rfind(" ", 0, 78)
            corte = corte if corte > 20 else 78
            texto.append(l[:corte])
            l = l[corte:].lstrip()
        texto.append(l)
    alto = 22 * len(texto) + 8
    h = img.shape[0]
    y0 = h - alto if abajo else 0
    capa = img.copy()
    cv2.rectangle(capa, (0, y0), (img.shape[1], y0 + alto), (0, 0, 0), -1)
    cv2.addWeighted(capa, 0.65, img, 0.35, 0, img)
    for i, l in enumerate(texto):
        cv2.putText(img, l, (8, y0 + 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def minimapa(datos, objetivo=None, tam=170):
    """Esquema de la placa a partir del CSV (no depende de la calibración)."""
    W, H = datos.ancho_ref, datos.alto_ref
    k = (tam - 10) / max(W, H)
    img = np.full((int(H * k) + 10, int(W * k) + 10, 3), 255, np.uint8)
    g = lambda x, y: (int(5 + x * k), int(5 + y * k))
    cv2.rectangle(img, g(0, 0), g(W, H), (0, 0, 0), 1)
    series = series_de(datos)
    for e in datos.elementos:
        col = (150, 150, 150)
        if e["serie"] in series and e["tipo"] in TIPOS_DATOS:
            col = PALETA[series.index(e["serie"]) % len(PALETA)]
        if e["forma"] == "punto":
            cv2.circle(img, g(e["x1"], e["y1"]), 1, col, -1)
        elif e["forma"] == "rect":
            cv2.rectangle(img, g(e["x1"], e["y1"]), g(e["x2"], e["y2"]), (200, 200, 200), 1)
        else:
            cv2.line(img, g(e["x1"], e["y1"]), g(e["x2"], e["y2"]), col, 1)
    if objetivo is not None:
        cv2.circle(img, g(objetivo["x"], objetivo["y"]), 7, (255, 0, 255), 2)
    return img


def pegar(img, peq, margen=6):
    h, w = peq.shape[:2]
    H, W = img.shape[:2]
    if h + margen < H and w + margen < W:
        img[margen:margen + h, W - w - margen:W - margen] = peq
    return img


def dibujar_elementos(img, elementos, color=(0, 255, 255)):
    for e in elementos:
        p1, p2 = (int(e["x1"]), int(e["y1"])), (int(e["x2"]), int(e["y2"]))
        if e["forma"] == "punto":
            cv2.circle(img, p1, 3, (0, 0, 255), -1)
        elif e["forma"] == "rect":
            cv2.rectangle(img, p1, p2, (170, 170, 170), 1)
        else:
            cv2.line(img, p1, p2, color if e["tipo"] != "eje" else (255, 255, 0), 1)
    return img


class ErrorCamara(Exception):
    """Mensaje pensado para leerse en voz alta."""


def nombres_camaras():
    """Nombres de las cámaras en el orden de DirectShow (Windows, con pygrabber); si no, []."""
    try:
        return list(eg.listar_camaras())
    except Exception:
        return []


def elegir_camara(cfg, pedida, nombres):
    """(índice, nombre, es_la_configurada). Nunca elige otra cámara por su cuenta: si la
    pedida o la configurada no está, lanza ErrorCamara con la lista de las que hay."""
    lista = "; ".join(f"{i}: {n}" for i, n in enumerate(nombres)) or "no pude leer sus nombres"

    def por_nombre(texto):
        return next((i for i, n in enumerate(nombres) if texto.lower() in n.lower()), None)

    if pedida not in (None, ""):
        s = str(pedida).strip()
        if s.isdigit():
            i = int(s)
            if nombres and i >= len(nombres):
                raise ErrorCamara(f"No hay cámara con el número {i}. Cámaras conectadas: {lista}.")
            return i, nombres[i] if i < len(nombres) else f"cámara {i}", True
        i = por_nombre(s)
        if i is None:
            raise ErrorCamara(f"No hay ninguna cámara cuyo nombre contenga «{s}». Cámaras conectadas: {lista}.")
        return i, nombres[i], True
    if cfg.get("indice_camara") is not None:
        i = int(cfg["indice_camara"])
        return i, nombres[i] if i < len(nombres) else f"cámara {i}", True
    buscada = str(cfg.get("nombre_camara") or "").strip()
    if nombres:
        if buscada:
            i = por_nombre(buscada)
            if i is None:
                raise ErrorCamara(f"No encuentro la cámara «{buscada}» de config_exploracion.json. "
                                  f"Cámaras conectadas: {lista}. Conéctala o indica cuál usar con --camara "
                                  "seguido del número.")
            return i, nombres[i], True
        return 0, nombres[0], False
    print("Aviso: no pude leer los nombres de las cámaras (en Windows: pip install pygrabber). "
          "Uso la cámara 0; si no es la de la placa, usa --camara 1.")
    return 0, "cámara 0", False


def abrir_camara(cfg, voz, pedida=None, intentos=6, espera=1.0):
    """Abre SOLO la cámara elegida, con los mismos ajustes que el explorador (resolución, MJPG,
    búfer). En Windows, si se conocen los nombres, solo con DirectShow: su numeración es la de
    pygrabber, y Media Foundation numera las cámaras en otro orden. Si la cámara no responde
    (normalmente porque otro programa la tiene abierta) reintenta unos segundos y luego avisa:
    no pasa a otra cámara, para no medir con la equivocada."""
    nombres = nombres_camaras()
    if nombres:
        print("Cámaras conectadas:", "; ".join(f"{i}: {n}" for i, n in enumerate(nombres)))
    idx, nombre, _ = elegir_camara(cfg, pedida, nombres)
    # Como en el explorador: el giro de 180° por defecto es solo para la cámara configurada (la del soporte).
    buscada = str(cfg.get("nombre_camara") or "").strip().lower()
    configurada = (bool(buscada) and buscada in nombre.lower()) or (
        cfg.get("indice_camara") is not None and int(cfg["indice_camara"]) == idx and pedida in (None, ""))
    motor = eg.MotorVision(queue.Queue(), voz, cfg)
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW] if nombres else [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        backends = [cv2.CAP_ANY]
    formatos = [True, False] if cfg.get("mjpg", True) else [False]
    cap = None
    for intento in range(1, intentos + 1):
        for b in backends:
            for mjpg in formatos:
                cap = motor._probar_camara(idx, b, mjpg)
                if cap is not None:
                    break
            if cap is not None:
                break
        if cap is not None:
            break
        if intento < intentos:
            print(f"La cámara {nombre} no responde (intento {intento} de {intentos}); reintento en {espera:.0f} s.")
            time.sleep(espera)
    if cap is None:
        raise ErrorCamara(
            f"No pude abrir la cámara {nombre}. Casi siempre es porque otro programa la tiene abierta: "
            "cierra el explorador y cualquier otra ventana o terminal de medir métricas, y vuelve a intentar. "
            "Si sigue igual, desconecta la cámara, conéctala de nuevo y espera unos segundos. "
            "No uso otra cámara para no medir con la equivocada.")
    motor.indice, motor.clave_camara = idx, nombre
    motor.orientacion = motor._orientacion_inicial(configurada)
    return motor, cap, nombres


class SesionCamara:
    """Abre la cámara de la placa con los ajustes del explorador y su orientación guardada."""

    def __init__(self, cfg, voz, orientacion=None, camara=None):
        self.motor, cap, self.nombres = abrir_camara(cfg, voz, camara)
        if orientacion:
            self.motor.orientacion = orientacion
        self.cap = cap
        self.captura = eg.Captura(cap)
        self.captura.start()
        self.n = 0

    @property
    def orientacion(self):
        return self.motor.orientacion

    def leer(self, timeout=1.0):
        """(imagen tal cual llega, imagen orientada) o (None, None)."""
        f, self.n = self.captura.esperar(self.n, timeout)
        if f is None:
            return None, None
        return f, eg.aplicar_orientacion(f, self.motor.orientacion)

    def cerrar(self):
        self.captura.parar()
        self.captura.join(timeout=1.5)
        try:
            self.cap.release()
        except Exception:
            pass


class Contexto:
    def __init__(self, args, cfg, datos, voz, pantalla=None, sesion=None):
        self.args, self.cfg, self.datos, self.voz = args, cfg, datos, voz
        self.pantalla, self.sesion = pantalla, sesion
        self.placa = slug(args.placa)
        self.hoja = tuple(int(v) for v in cfg["hoja"])
        # Registro como en la aplicación: las 4 esquinas guardadas o, si no hay, la 'hoja'.
        self.esquinas = eg.esquinas_validas(cfg.get("esquinas_placa_px"))
        self.origen_registro = cfg.get("registro_origen") or ("esquinas" if self.esquinas is not None else "rect")
        self.calibracion = "guardada en config_exploracion.json" + (
            " (4 esquinas)" if self.esquinas is not None else " (rectángulo 'hoja')")
        self._detector = None
        self.objetivo_actual = None
        self.carpeta = os.path.join(CARPETA_METRICAS, self.placa)
        os.makedirs(self.carpeta, exist_ok=True)

    def registro(self):
        """vision.Registro en uso (None sin gráfica)."""
        return eg.registro_para(self.datos, self.esquinas, self.hoja, self.origen_registro)

    def usar_deteccion(self, det, como):
        """Usar una autodetección como registro (y su caja como 'hoja', para las pruebas de B)."""
        self.esquinas, self.origen_registro = det.esquinas, "autodeteccion"
        self.hoja = tuple(int(v) for v in det.rect)
        self.calibracion = como

    def decir(self, texto):
        print("[voz]", texto)
        try:
            self.voz.decir(texto, interrumpir=True, esencial=True)
        except Exception:
            pass

    def detector(self):
        if self._detector is None:
            self.decir("Cargando el seguimiento de la mano.")
            self._detector = eg.DetectorMano(self.cfg, avisar=self.decir)
        return self._detector

    def cerrar(self):
        if self._detector is not None:
            self._detector.cerrar()

    def ruta(self, nombre):
        return os.path.join(self.carpeta, nombre)

    def condiciones(self):
        a = self.args
        return {k: getattr(a, k, None) for k in ("color", "grafica", "fondo", "luz", "nota")
                if getattr(a, k, None)}

    def guardar(self, prueba, metricas, detalle=None, extra=None):
        d = {"prueba": prueba, "placa": self.placa, "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
             "version": VERSION, "condiciones": self.condiciones(), "calibracion": self.calibracion,
             "config": {k: self.cfg.get(k) for k in ("hoja", "distancia_pinza", "radio_punto", "radio_linea",
                                                     "radio_eje", "radio_texto", "resolucion", "confianza_mano",
                                                     "esquinas_placa_px", "registro_origen", "exposicion",
                                                     "mano_recorte_placa", "mano_lado_max_px")},
             "hoja_usada": list(self.hoja),
             "registro_usado": {"origen": self.origen_registro,
                                "esquinas_px": None if self.esquinas is None else np.round(self.esquinas, 2).tolist()},
             "metricas": metricas}
        if self.datos is not None:
            d["grafica"] = {"titulo": self.datos.titulo, "series": series_de(self.datos),
                            "fuente": self.datos.url or self.datos.ruta_local,
                            "placa_mm": [self.datos.ancho_ref, self.datos.alto_ref]}
        if self.sesion is not None:
            d["camara"] = {"nombre": self.sesion.motor.clave_camara, "orientacion": self.sesion.orientacion}
        if extra:
            d.update(extra)
        if detalle is not None:
            d["detalle"] = detalle
        ruta = self.ruta(f"{marca_tiempo()}_{prueba}.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(limpiar(d), f, ensure_ascii=False, indent=1)
        print(f"  → guardado: {ruta}")
        return ruta

    # ---- interacción ----
    def mostrar(self, img, lineas, mapa_obj=False):
        if self.pantalla is None:
            return
        img = rotulo(img.copy(), lineas)
        if self.datos is not None and mapa_obj is not False:
            pegar(img, minimapa(self.datos, mapa_obj))
        self.pantalla.mostrar(img)

    def esperar_espacio(self, mensaje, dibujar=None, mapa_obj=False):
        """Muestra la cámara hasta que se pulse Espacio (True), S (None) o Esc (False)."""
        self.decir(mensaje)
        while True:
            _crudo, f = self.sesion.leer()
            if f is None and not self.sesion.captura.ok:
                self.decir("Se perdió la señal de la cámara.")
                return False
            if f is not None:
                if dibujar:
                    f = dibujar(f)
                self.mostrar(f, [mensaje, "Espacio: empezar  ·  R: repetir  ·  S: saltar  ·  Esc: terminar"], mapa_obj)
            k = self.pantalla.tecla(1)
            if k == 32:
                return True
            if k in (ord("s"), ord("S")):
                return None
            if k in (27, ord("q")):
                return False
            if k in (ord("r"), ord("R")):
                self.decir(mensaje)


def mano_px(lm, w, h):
    """(punta del índice, punta del medio, largo de la palma) en px, como en la aplicación."""
    indice = (int(lm[8].x * w), int(lm[8].y * h))
    medio = (int(lm[12].x * w), int(lm[12].y * h))
    palma = math.hypot((lm[9].x - lm[0].x) * w, (lm[9].y - lm[0].y) * h)
    return indice, medio, palma


# ============================================================
# PRUEBAS CON CÁMARA
# ============================================================
def prueba_hoja(ctx, repeticiones=3, frames=15):
    """Autodetección de la placa (como la tecla A y la comprobación al empezar a explorar),
    retirando y recolocando la placa entre repeticiones."""
    datos, reps, ultimo = ctx.datos, [], None
    ref = ctx.registro()
    for r in range(repeticiones):
        msg = (f"Prueba de la placa, repetición {r + 1} de {repeticiones}. "
               + ("Coloca la placa en su lugar de exploración, sin la mano encima, y pulsa espacio."
                  if r == 0 else "Retira la placa, vuelve a colocarla en su lugar y pulsa espacio."))
        ok = ctx.esperar_espacio(msg)
        if ok is False:
            break
        if ok is None:
            continue
        lecturas = []
        while len(lecturas) < frames:
            _c, f = ctx.sesion.leer()
            if f is None:
                continue
            a = analizar_placa(f, datos, ref)
            lecturas.append(a)
            ultimo = (f, a)
            vista = f.copy()
            if a["_det"] is not None:
                cv2.polylines(vista, [poligono_px(a["_det"].esquinas)], True, (0, 255, 0), 2)
            if ref is not None:
                cv2.polylines(vista, [poligono_px(ref)], True, (0, 255, 255), 1)
            ctx.mostrar(vista, [f"Midiendo la placa... {len(lecturas)}/{frames}",
                                "verde = autodetectada · amarillo = registro guardado"])
            ctx.pantalla.tecla(1)
        reps.append(lecturas)
    todas = [a for rep in reps for a in rep]
    if not todas:
        return None
    ok = [a for a in todas if a["detectada"]]
    m = {"frames": len(todas), "repeticiones": len(reps),
         "exito": proporcion(len(ok), len(todas)), "exito_pct": 100.0 * len(ok) / len(todas),
         "ms_mediana": mediana([a["ms"] for a in todas])}
    if ok:
        m["confianza_alta_pct"] = 100.0 * sum(a["confianza"] == "alta" for a in ok) / len(ok)
        m["reconstruida_pct"] = 100.0 * sum(bool(a["reconstruida"]) for a in ok) / len(ok)
        metodos = [a["metodo"] for a in ok]
        m["metodo"] = max(set(metodos), key=metodos.count)
        m["error_estimado_mm"] = mediana([a.get("error_estimado_mm") for a in ok])
        bordes = {}
        for a in ok:
            for b in a.get("bordes_fuera") or []:
                bordes[b] = bordes.get(b, 0) + 1
        m["bordes_fuera"] = bordes
    con_esq = [a for a in ok if "esquina_ok" in a]
    if con_esq:
        m["esquina_ok_pct"] = 100.0 * sum(a["esquina_ok"] for a in con_esq) / len(con_esq)
        m["esquina_vista"] = max({a["esquina"] for a in con_esq}, key=lambda s: sum(a["esquina"] == s for a in con_esq))
    m["rotacion_med_grados"] = mediana([abs(a["rotacion_grados"]) for a in ok if "rotacion_grados" in a])
    m["toca_borde_pct"] = 100.0 * sum(a.get("toca_borde", False) for a in ok) / len(ok) if ok else None
    m["area_pct"] = mediana([a.get("area_pct") for a in ok])
    m["aspecto_px_vs_mm"] = mediana([a.get("aspecto_px_vs_mm") for a in ok])
    for k in ("delta_gris", "delta_e", "fondo_desv_gris", "separabilidad_otsu", "gris_placa", "gris_fondo"):
        m[k] = mediana([a["contraste"][k] for a in todas if a.get("contraste")])
    if datos is not None:
        sx, sy = escala_px_mm(datos, ctx.hoja)
        m["anisotropia_calibracion"] = sx / sy
        W2, H2 = datos.ancho_ref / 2, datos.alto_ref / 2
        base, por_rep = None, []
        for rep in reps:
            regs = [a["_det"].registro(datos) for a in rep if a["_det"] is not None]
            if not regs:
                continue
            base = base or regs[0]                   # referencia fija para expresar todo en mm
            c = np.array([base.imagen_a_placa(*r.placa_a_imagen_f(W2, H2)) for r in regs])
            m.setdefault("estabilidad_mm", []).append(float(np.hypot(*c.std(axis=0))))
            esq = np.median(np.array([r.esquinas_px() for r in regs]), axis=0)
            por_rep.append(eg.Registro.desde_esquinas(esq, datos.ancho_ref, datos.alto_ref, datos.estirar))
        if m.get("estabilidad_mm"):
            m["estabilidad_mm"] = max(m["estabilidad_mm"])
        if len(por_rep) >= 2:
            cc = np.array([base.imagen_a_placa(*r.placa_a_imagen_f(W2, H2)) for r in por_rep])
            m["reposicion_mm"] = float(np.hypot(*cc.std(axis=0)))
            m["reposicion_max_mm"] = float(np.hypot(*(cc - cc.mean(axis=0)).T).max())
        if por_rep and ref is not None:
            # Cuánto se equivoca el registro guardado si la placa está donde se detecta ahora
            # (en el área del gráfico): es lo que causó la superposición corrida en la turquesa.
            x0, y0, x1, y1 = caja_grafico(datos)
            med, mx = ref.diferencia_mm(por_rep[-1], x0, y0, x1, y1)
            m["desfase_guardada_mm"], m["desfase_guardada_mediana_mm"] = mx, med
            m["iou_guardada"] = mediana([iou(a["rect"], ref.caja_px()) for a in ok])
    if ultimo is not None:
        f, a = ultimo
        img = f.copy()
        if a["_det"] is not None:
            cv2.polylines(img, [poligono_px(a["_det"].esquinas)], True, (0, 255, 0), 2)
        if ref is not None:
            cv2.polylines(img, [poligono_px(ref)], True, (0, 255, 255), 1)
        cv2.imwrite(ctx.ruta(f"{marca_tiempo()}_hoja.png"), img)
        cv2.imwrite(ctx.ruta(f"{marca_tiempo()}_frame.png"), f)
    ctx.guardar("hoja", m, detalle=[[{k: v for k, v in a.items() if k != "contraste"} for a in rep] for rep in reps])
    return m


def prueba_qr(ctx, condiciones=("en la bandeja",), segundos=15.0, max_lecturas=10):
    lector = eg.LectorQR()
    resultados, todos = [], []
    presupuesto = ctx.cfg.get("qr_presupuesto_ms")
    res_qr = ctx.cfg.get("resolucion_qr")          # como la aplicación (F3): más resolución solo para el QR
    if not isinstance(res_qr, (list, tuple)):      # "auto": la que eligió la aplicación para esta cámara
        res_qr = (ctx.cfg.get("modos_qr") or {}).get(ctx.sesion.motor.clave_camara)
        if isinstance(res_qr, dict):
            res_qr = res_qr.get("res") if res_qr.get("v", 0) >= eg.VERSION_MODO_QR else None
        if not res_qr:
            print("QR: sin resolución elegida para esta cámara; abre una vez la app con F3 para que la elija.")
    normal = tuple(ctx.cfg.get("resolucion", [640, 480]))
    if res_qr and not eg.misma_proporcion(res_qr, normal):
        print("QR: %dx%d tiene otra proporción que %dx%d (recorta la imagen); uso la normal." % (tuple(res_qr) + normal))
        res_qr = None
    zoom = eg.ZoomQR(ctx.cfg, lector) if ctx.cfg.get("qr_zoom", True) else None
    if res_qr:
        ctx.sesion.captura.pedir_resolucion(*res_qr)
    for cond in condiciones:
        msg = (f"Prueba del código QR: {cond}. "
               + ("Deja la placa en su lugar de exploración." if "bandeja" in cond
                  else "Sostén la placa así frente a la cámara.") + " Pulsa espacio para empezar.")
        ok = ctx.esperar_espacio(msg)
        if ok is False:
            break
        if ok is None:
            continue
        intentos, t0, salir = [], time.time(), False
        while time.time() - t0 < segundos and sum(a["leido"] for a in intentos) < max_lecturas:
            crudo, f = ctx.sesion.leer()
            if crudo is None:
                continue
            a = analizar_qr(lector, crudo, presupuesto, zoom, ctx.sesion.orientacion)   # como la aplicación
            a["t"] = time.time() - t0
            intentos.append(a)
            n_ok = sum(x["leido"] for x in intentos)
            ctx.mostrar(f, [f"QR ({cond}): {n_ok} lecturas en {len(intentos)} intentos",
                            "Esc: terminar esta condición"])
            if ctx.pantalla.tecla(1) == 27:
                salir = True
                break
        k = sum(a["leido"] for a in intentos)
        primera = next((a["t"] for a in intentos if a["leido"]), None)
        r = {"condicion": cond, "intentos": len(intentos), "lecturas": k,
             "tasa": proporcion(k, len(intentos)), "tasa_pct": 100.0 * k / len(intentos) if intentos else None,
             "t_primera_s": primera, "ms_mediana": mediana([a["ms"] for a in intentos]),
             "ms_p95": percentil([a["ms"] for a in intentos], 95),
             "contraste_michelson": mediana([a.get("contraste_michelson") for a in intentos]),
             "lado_px": mediana([a.get("lado_px") for a in intentos]),
             "polaridad": next((a["polaridad"] for a in intentos if a["leido"]), None),
             "orientacion_qr": next((a.get("orientacion_qr") for a in intentos if a.get("orientacion_qr")), None),
             "variantes": sorted({a["variante"] for a in intentos if a.get("variante")}),
             "resolucion": intentos[-1]["resolucion"] if intentos else None,
             "zoom_placa_pct": (100.0 * sum(a.get("zoom_placa", False) for a in intentos) / len(intentos)
                                if intentos and zoom is not None else None),
             "px_por_modulo": mediana([a.get("px_por_modulo") for a in intentos]),
             "texto": next((a["texto"] for a in intentos if a["leido"]), None)}
        r["orientacion_coincide"] = (None if not r["orientacion_qr"] else r["orientacion_qr"] == "ninguno")
        resultados.append(r)
        todos += intentos
        ctx.decir(f"{k} lecturas en {len(intentos)} intentos." + (f" Primera lectura a los {primera:.1f} segundos."
                                                                   if primera is not None else " No se leyó el código."))
        if salir:
            break
    if res_qr:                                       # vuelve a la resolución de explorar
        ctx.sesion.captura.pedir_resolucion(*ctx.cfg.get("resolucion", [640, 480]))
    if not resultados:
        return None
    k = sum(a["leido"] for a in todos)
    m = {"condiciones": resultados, "tasa": proporcion(k, len(todos)),
         "tasa_pct": 100.0 * k / len(todos) if todos else None,
         "t_primera_s": mediana([r["t_primera_s"] for r in resultados]),
         "ms_mediana": mediana([a["ms"] for a in todos])}
    texto = next((r["texto"] for r in resultados if r["texto"]), None)
    if texto and ctx.datos is None:
        try:
            ctx.datos = eg.cargar_desde_enlace(texto)
            ctx.decir(f"Gráfica cargada desde el QR: {ctx.datos.titulo}.")
        except eg.ErrorDatos as e:
            ctx.decir(f"El QR se leyó pero no pude cargar la gráfica: {e}")
    ctx.guardar("qr", m)
    return m


def pedir_clics(ctx, frame, refs):
    """Una persona vidente hace clic en cada referencia. Devuelve [(x, y) | None] por referencia."""
    clics, i = [None] * len(refs), 0
    escala = max(0.5, min(2.0, 1200 / frame.shape[1], 860 / frame.shape[0]))
    ctx.decir("Ahora una persona vidente debe hacer clic en la imagen sobre cada punto que te pida. "
              "S salta un punto que no se vea, Z deshace el anterior, Escape termina.")
    while i < len(refs):
        nombre, x, y = refs[i]
        ctx.decir(f"Clic en: {nombre}.")
        hecho = False
        while not hecho:
            img = frame.copy()
            for c in clics:
                if c is not None:
                    cv2.drawMarker(img, (int(c[0]), int(c[1])), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
            img = rotulo(img, [f"Punto {i + 1} de {len(refs)}: clic en {nombre}",
                               "S: saltar  ·  Z: deshacer  ·  Esc: terminar"])
            pegar(img, minimapa(ctx.datos, {"x": x, "y": y}))
            ctx.pantalla.mostrar(img, escala)
            c = ctx.pantalla.tomar_clic()
            k = ctx.pantalla.tecla(15)
            if c is not None:
                clics[i] = c
                i, hecho = i + 1, True
            elif k in (ord("s"), ord("S")):
                i, hecho = i + 1, True
            elif k in (ord("z"), ord("Z")) and i > 0:
                i -= 1
                clics[i] = None
                hecho = True
            elif k == 27:
                return clics
    return clics


def imagen_registro(frame, r):
    img = frame.copy()
    for p in r.get("puntos", []):
        c = tuple(int(v) for v in p["clic_px"])
        d = tuple(int(v) for v in p["dibujado_px"])
        cv2.arrowedLine(img, d, c, (0, 0, 255), 1, tipLength=0.15)
        cv2.circle(img, c, 4, (0, 255, 0), -1)
        cv2.circle(img, d, 4, (0, 0, 255), 1)
    rotulo(img, [f"verde = real (clic) · rojo = donde lo dibuja el sistema · error mediano "
                 f"{r.get('error_mediana_mm', 0):.1f} mm"], abajo=True)
    return img


def prueba_registro(ctx):
    if ctx.datos is None:
        print("  (registro: falta la gráfica)")
        return None
    reg = ctx.registro()
    ok = ctx.esperar_espacio("Prueba de registro. Retira la mano de la placa y pulsa espacio para congelar la imagen.",
                             dibujar=lambda f: dibujar_elementos(f, escalar(ctx)))
    if not ok:
        return None
    frame = None
    while frame is None:
        _c, frame = ctx.sesion.leer()
    refs = puntos_referencia(ctx.datos)
    clics = pedir_clics(ctx, frame, refs)
    det = eg.detectar_placa(frame, ctx.datos)
    r = calcular_registro(ctx.datos, reg, refs, clics, det.registro(ctx.datos) if det else None)
    if det is not None:
        r["autodeteccion_metodo"], r["autodeteccion_confianza"] = det.metodo, det.confianza
    if "error" not in r:
        cv2.imwrite(ctx.ruta(f"{marca_tiempo()}_registro.png"), imagen_registro(frame, r))
        texto = (f"Error de registro mediano: {r['error_mediana_mm']:.0f} milímetros; máximo en el área del gráfico: "
                 f"{r['error_max_grafico_mm']:.0f} milímetros.")
        if r.get("error_auto_max_grafico_mm") is not None:
            texto += f" Con la autodetección, máximo de {r['error_auto_max_grafico_mm']:.0f} milímetros."
        ctx.decir(texto)
    else:
        ctx.decir(r["error"])
    ctx.guardar("registro", r)
    return r


def escalar(ctx):
    return eg.escalar_elementos(ctx.datos, ctx.registro()) if ctx.datos is not None else []


def prueba_mano(ctx, segundos=20.0):
    """Como la aplicación: la mano se busca en la placa y un margen (si "mano_recorte_placa"),
    y los dedos se pasan a mm con el registro (MuestraMano)."""
    det = ctx.detector()
    ok = ctx.esperar_espacio(f"Prueba de la mano. Durante {int(segundos)} segundos mueve la mano sobre la placa, "
                             "abriendo y cerrando la pinza. Pulsa espacio para empezar.")
    if not ok:
        return None
    reg = ctx.registro()
    recortar = reg is not None and ctx.cfg.get("mano_recorte_placa", True)
    margen = float(ctx.cfg.get("mano_margen_recorte", 0.4))
    t0, n0 = time.time(), ctx.sesion.captura.n
    ms, palmas, pinzas, color_mano, color_placa, n, con = [], [], [], [], [], 0, 0
    palmas_mm, pinzas_mm, dentro, roi = [], [], 0, None
    while time.time() - t0 < segundos:
        _c, f = ctx.sesion.leer()
        if f is None:
            continue
        h, w = f.shape[:2]
        roi = eg.roi_mano(reg, w, h, margen) if recortar else None
        t = time.perf_counter()
        manos = det.detectar(f, roi)
        ms.append(1000 * (time.perf_counter() - t))
        n += 1
        vista = f.copy()
        if roi is not None:
            cv2.rectangle(vista, roi[:2], roi[2:], (255, 255, 0), 1)
        if manos:
            con += 1
            lm = manos[0]
            ind, med, palma = mano_px(lm, w, h)
            palmas.append(palma)
            pinzas.append(math.hypot(ind[0] - med[0], ind[1] - med[1]))
            if reg is not None:
                mu = eg.muestra_de_mano(lm, w, h, reg)
                palmas_mm.append(mu.palma_mm)
                pinzas_mm.append(math.hypot(mu.indice_mm[0] - mu.medio_mm[0], mu.indice_mm[1] - mu.medio_mm[1]))
                dentro += mu.dentro_placa
            if n % 5 == 0:
                pts = [(int(lm[j].x * w), int(lm[j].y * h)) for j in (0, 5, 9, 13, 17)]
                cx, cy = int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts]))
                r = max(3, int(palma * 0.12))
                parche = f[max(0, cy - r):cy + r, max(0, cx - r):cx + r].reshape(-1, 3)
                if parche.size:
                    color_mano.append(np.median(parche, axis=0))
                if reg is not None:                    # la placa sin la zona de la mano
                    xs = [int(p.x * w) for p in lm]
                    ys = [int(p.y * h) for p in lm]
                    mask = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(mask, [poligono_px(reg)], 255)
                    mask[max(0, min(ys) - 15):max(0, max(ys) + 15), max(0, min(xs) - 15):max(0, max(xs) + 15)] = 0
                    if mask.any():
                        color_placa.append(np.median(f[mask > 0].astype(float), axis=0))
            for p in (ind, med):
                cv2.circle(vista, p, 6, (255, 0, 0), -1)
        ctx.mostrar(vista, [f"Mano: {con}/{n} frames · {int(segundos - (time.time() - t0))} s",
                            "Esc: terminar"])
        if ctx.pantalla.tecla(1) == 27:
            break
    dur = max(1e-6, time.time() - t0)
    fps_cap = (ctx.sesion.captura.n - n0) / dur
    m = {"frames": n, "deteccion": proporcion(con, n), "deteccion_pct": 100.0 * con / n if n else None,
         "fps_captura": fps_cap, "fps_proceso": n / dur,
         "ms_mano_mediana": mediana(ms), "ms_mano_p95": percentil(ms, 95),
         "latencia_vision_p95_ms": (1000.0 / fps_cap if fps_cap else 0) + (percentil(ms, 95) or 0),
         "palma_px_mediana": mediana(palmas),
         "pinza_px": {"p10": percentil(pinzas, 10), "p50": percentil(pinzas, 50), "p90": percentil(pinzas, 90)},
         "umbral_pinza_px": ctx.cfg["distancia_pinza"],
         "recorte_a_placa": roi is not None, "roi_px": list(roi) if roi else None,
         "camara_formato": eg.fourcc_texto(ctx.sesion.cap), "exposicion": ctx.cfg.get("exposicion"),
         "mano_lado_max_px": ctx.cfg.get("mano_lado_max_px") or 0}
    if fps_cap and fps_cap < 20:
        m["aviso_fps"] = ("la cámara entrega menos de 20 fps: probar \"exposicion\" fija (p. ej. -6) o más luz; "
                          "el modelo de la mano tarda " + (f"{m['ms_mano_mediana']:.0f}" if ms else "?") + " ms")
    if palmas_mm:
        m["palma_mm_mediana"] = mediana(palmas_mm)
        m["pinza_mm"] = {"p10": percentil(pinzas_mm, 10), "p50": percentil(pinzas_mm, 50), "p90": percentil(pinzas_mm, 90)}
        m["dentro_placa_pct"] = 100.0 * dentro / max(con, 1)
    if m["palma_px_mediana"]:
        # Umbral antiguo (px fijos) expresado en palmas, para comparar con el nuevo.
        m["umbral_pinza_px_sobre_palma"] = ctx.cfg["distancia_pinza"] / m["palma_px_mediana"]
    # La pinza se decide ahora con índice–medio / palma (interaccion.py): su distribución sirve
    # para ajustar "pinza_cerrar_palma" y "pinza_abrir_palma".
    m["umbral_pinza_sobre_palma"] = float(ctx.cfg.get("pinza_cerrar_palma", eg.DEFECTO_INTERACCION["pinza_cerrar_palma"]))
    proporciones = [p / q for p, q in zip(pinzas, palmas) if q]
    if proporciones:
        m["pinza_sobre_palma"] = {"p10": percentil(proporciones, 10), "p50": percentil(proporciones, 50),
                                  "p90": percentil(proporciones, 90)}
    if ctx.datos is not None:
        sx, sy = escala_px_mm(ctx.datos, ctx.hoja)
        m["umbral_pinza_mm"] = ctx.cfg["distancia_pinza"] * 2 / (sx + sy)
    if color_mano and color_placa:
        lm_, lp_ = bgr_a_lab(np.median(color_mano, axis=0)), bgr_a_lab(np.median(color_placa, axis=0))
        m["delta_e_mano_placa"] = float(delta_e2000(lm_, lp_))
        m["delta_gris_mano_placa"] = float(abs(gris_de(np.median(color_mano, axis=0)) - gris_de(np.median(color_placa, axis=0))))
        m["lab_mano"], m["lab_placa"] = lm_.tolist(), lp_.tolist()
    ctx.decir(f"Mano detectada en el {m['deteccion_pct']:.0f} por ciento de los frames, a {m['fps_proceso']:.0f} "
              f"imágenes por segundo; la cámara da {fps_cap:.0f}." if n else "No llegaron imágenes de la cámara.")
    ctx.guardar("mano", m)
    return m


def elegir_objetivos(datos, n=15, semilla=7):
    """Elementos a señalar: extremos y centro de cada serie, tramos junto a los cruces, ejes y algún texto."""
    rnd = random.Random(semilla)
    cr = cruces(datos)
    por_id = {e["id"]: e for e in datos.elementos}
    cand = []

    def cerca_de_cruce(x, y, serie):
        return any(serie in c["series"] and math.hypot(x - c["x"], y - c["y"]) < 8 for c in cr)

    for s in series_de(datos):
        pts = sorted((e for e in datos.elementos if e["tipo"] == "punto" and e["serie"] == s), key=lambda e: e["x1"])
        for e in {id(p): p for p in (pts[:1] + pts[len(pts) // 2:len(pts) // 2 + 1] + pts[-1:])}.values():
            zona = "cruce" if cerca_de_cruce(e["x1"], e["y1"], s) else "aislado"
            cand.append({"id": e["id"], "tipo": "punto", "clase": "punto", "serie": s, "zona": zona, "x": e["x1"], "y": e["y1"],
                         "indicacion": e["texto"]})
        curvas = [e for e in datos.elementos if e["tipo"] in ("curva", "tramo") and e["serie"] == s]
        libres = [e for e in curvas if not cerca_de_cruce(*centro(e), s)]
        if libres:
            e = libres[len(libres) // 2]
            x, y = centro(e)
            cand.append({"id": e["id"], "tipo": e["tipo"], "clase": "linea", "serie": s, "zona": "aislado", "x": x, "y": y,
                         "indicacion": f"la línea de {s}, en el tramo: {e['texto']}"})
    for c in cr:
        for cid, otra in zip(c["curvas"], reversed(c["series"])):
            e = por_id[cid]
            # A 3 mm del cruce, sobre esa misma línea, hacia su extremo más lejano.
            d1 = math.hypot(e["x1"] - c["x"], e["y1"] - c["y"])
            d2 = math.hypot(e["x2"] - c["x"], e["y2"] - c["y"])
            (ax, ay), lejos = ((e["x1"], e["y1"]), d1) if d1 >= d2 else ((e["x2"], e["y2"]), d2)
            paso = min(3.0, 0.5 * lejos) / max(lejos, 1e-9)
            x, y = c["x"] + (ax - c["x"]) * paso, c["y"] + (ay - c["y"]) * paso
            cand.append({"id": cid, "tipo": e["tipo"], "clase": "linea", "serie": e["serie"], "zona": "cruce", "x": x, "y": y,
                         "indicacion": f"la línea de {e['serie']} justo donde se cruza con {otra}"})
    for e in datos.elementos:
        if e["tipo"] == "eje":
            x, y = centro(e)
            cand.append({"id": e["id"], "tipo": "eje", "clase": "eje", "serie": "", "zona": "eje", "x": x, "y": y,
                         "indicacion": f"el {e['corto'].lower()}, por la mitad"})
    textos = [e for e in datos.elementos if e["tipo"] in ("texto", "leyenda") and e["id"].lower() != "qr"]
    for e in rnd.sample(textos, min(2, len(textos))):
        x, y = centro(e)
        cand.append({"id": e["id"], "tipo": e["tipo"], "clase": "texto", "serie": e["serie"], "zona": "texto", "x": x, "y": y,
                     "indicacion": e["texto"]})
    cruce = [c for c in cand if c["zona"] == "cruce"]
    resto = [c for c in cand if c["zona"] != "cruce"]
    rnd.shuffle(cruce)
    rnd.shuffle(resto)
    elegidos = cruce[:max(0, n // 3)] + resto
    elegidos = elegidos[:n]
    if len(elegidos) < n:
        elegidos += [c for c in cruce if c not in elegidos][:n - len(elegidos)]
    rnd.shuffle(elegidos)
    return elegidos


def resumir_objetivos(intentos):
    """Métricas de la prueba de objetivos a partir de los intentos (sirve también para juntar sesiones)."""
    validos = [t for t in intentos if t["frames"]]
    if not validos:
        return {"intentos": 0}
    m = {"intentos": len(validos)}
    k = sum(t["acierto_estricto"] for t in validos)
    m["acierto"] = proporcion(k, len(validos))
    m["acierto_pct"] = 100.0 * k / len(validos)
    m["acierto_serie_pct"] = 100.0 * sum(t["acierto_serie"] for t in validos) / len(validos)
    for zona in ("aislado", "cruce", "eje", "texto", "aislado_punto", "aislado_linea", "cruce_punto", "cruce_linea"):
        z = [t for t in validos if t["zona"] == zona or f'{t["zona"]}_{t.get("clase", "")}' == zona]
        if z:
            kz = sum(t["acierto_estricto"] for t in z)
            m[f"acierto_{zona}"] = proporcion(kz, len(z))
            m[f"acierto_{zona}_pct"] = 100.0 * kz / len(z)
    m["sin_seleccion_pct"] = 100.0 * sum(t["seleccionado"] is None for t in validos) / len(validos)
    m["error_mediana_mm"] = mediana([t["error_mm"] for t in validos])
    m["error_p90_mm"] = percentil([t["error_mm"] for t in validos if t["error_mm"] is not None], 90)
    sesgos = [t["sesgo_mm"] for t in validos if t.get("sesgo_mm")]
    if sesgos:
        s = np.mean(sesgos, axis=0)
        m["sesgo_mm"] = [float(s[0]), float(s[1])]
        m["sesgo_modulo_mm"] = float(np.hypot(*s))
    m["jitter_mm"] = mediana([t["jitter_mm"] for t in validos])
    m["pinza_mantenida_pct"] = mediana([t["pinza_pct"] for t in validos])
    m["mano_pct"] = mediana([t["mano_pct"] for t in validos])
    m["t_primer_acierto_ms"] = mediana([t["t_primer_acierto_ms"] for t in validos])
    # Interacción de la Persona B (sesiones 1.2 en adelante): estabilidad y comparación sin ella.
    con_b = [t for t in validos if "cambios_sel" in t]
    if con_b:
        m["cambios_sel_mediana"] = mediana([t["cambios_sel"] for t in con_b])
        m["cambios_crudo_mediana"] = mediana([t["cambios_crudo"] for t in con_b])
        m["acierto_crudo_pct"] = 100.0 * sum(t["acierto_crudo"] for t in con_b) / len(con_b)
        m["pinza_px_mantenida_pct"] = mediana([t["pinza_px_pct"] for t in con_b])
    conf = {}
    for t in validos:
        if not t["acierto_estricto"]:
            clave = f"{t['objetivo']} → {t['seleccionado']}"
            conf[clave] = conf.get(clave, 0) + 1
    m["confusiones"] = sorted(conf.items(), key=lambda kv: -kv[1])[:15]
    return m


def _cambios(secuencia):
    """Cuántas veces cambia el valor a lo largo de la secuencia (parpadeo de la selección)."""
    return sum(1 for a, b in zip(secuencia, secuencia[1:]) if a != b)


def muestra_camara(ctx, det, reg, f):
    """Como la aplicación: mano (recortada a la placa si "mano_recorte_placa") → MuestraMano en mm.
    Devuelve (landmarks | None, MuestraMano)."""
    h, w = f.shape[:2]
    roi = eg.roi_mano(reg, w, h, float(ctx.cfg.get("mano_margen_recorte", 0.4))) \
        if reg is not None and ctx.cfg.get("mano_recorte_placa", True) else None
    manos = det.detectar(f, roi)
    lm = manos[0] if manos else None
    return lm, eg.muestra_de_mano(lm, w, h, reg)


def medir_objetivo(ctx, det, sel, obj, ventana_s, reg=None):
    """'sel' es una Interaccion (la de la aplicación, sin voz): se mide lo que anunciaría.
    Para comparar se guarda también la búsqueda sin suavizado ni estabilidad ('sel_crudo')
    y la pinza con el umbral fijo en px de antes ('pinza_px')."""
    reg = reg if reg is not None else ctx.registro()
    destino = next(e for e in ctx.datos.elementos if e["id"] == obj["id"])
    sel.reiniciar()
    registros, t0 = [], time.time()
    while time.time() - t0 < ventana_s:
        _c, f = ctx.sesion.leer()
        if f is None:
            continue
        h, w = f.shape[:2]
        lm, mu = muestra_camara(ctx, det, reg, f)
        s = sel.actualizar(mu)
        r = {"t_ms": 1000 * (time.time() - t0), "mano": lm is not None, "pinza": bool(s.pinza),
             "sel": s.elemento_id, "medida_pinza": sel.medida_pinza}
        vista = f.copy()
        if lm is not None:
            ind, med, _ = mano_px(lm, w, h)
            r["pinza_px"] = math.hypot(ind[0] - med[0], ind[1] - med[1]) <= ctx.cfg["distancia_pinza"]
            xm, ym = mu.indice_mm
            e = sel._buscar(xm, ym)
            r["sel_crudo"] = e["id"] if e else None
            cx, cy = punto_cercano(xm, ym, destino)
            r["mm"] = [xm, ym]
            r["error_mm"] = math.hypot(xm - cx, ym - cy)
            r["sesgo"] = [xm - cx, ym - cy]
            cv2.circle(vista, ind, 7, (0, 255, 0) if r["pinza"] else (255, 0, 0), -1)
        registros.append(r)
        ctx.mostrar(vista, [f"Midiendo: {obj['indicacion'][:70]}"], obj)
        ctx.pantalla.tecla(1)
    con_mano = [r for r in registros if r["mano"]]
    pinza = [r for r in con_mano if r.get("pinza")]
    base = pinza or con_mano

    def votar(clave):
        votos = {}
        for r in base:
            votos[r.get(clave)] = votos.get(r.get(clave), 0) + 1
        return votos, (max(votos.items(), key=lambda kv: kv[1])[0] if votos else None)

    votos, elegido = votar("sel")
    _votos_crudo, elegido_crudo = votar("sel_crudo")
    el = next((e for e in ctx.datos.elementos if e["id"] == elegido), None)
    acierto = elegido == obj["id"]
    acierto_serie = acierto or (obj["tipo"] in TIPOS_DATOS and el is not None and el["tipo"] in TIPOS_DATOS
                                and el["serie"] == obj["serie"])
    mm = np.array([r["mm"] for r in base]) if base else np.zeros((0, 2))
    primero = next((r["t_ms"] for r in registros if r.get("pinza") and r.get("sel") == obj["id"]), None)
    return {"objetivo": obj["id"], "tipo": obj["tipo"], "clase": obj.get("clase", ""), "serie": obj["serie"],
            "zona": obj["zona"],
            "objetivo_mm": [obj["x"], obj["y"]], "frames": len(registros),
            "mano_pct": 100.0 * len(con_mano) / len(registros) if registros else 0.0,
            "pinza_pct": 100.0 * len(pinza) / len(con_mano) if con_mano else 0.0,
            "pinza_px_pct": 100.0 * sum(bool(r.get("pinza_px")) for r in con_mano) / len(con_mano) if con_mano else 0.0,
            "seleccionado": elegido, "votos": votos, "acierto_estricto": bool(acierto),
            "acierto_serie": bool(acierto_serie),
            "seleccionado_crudo": elegido_crudo, "acierto_crudo": elegido_crudo == obj["id"],
            "cambios_sel": _cambios([r.get("sel") for r in con_mano]),
            "cambios_crudo": _cambios([r.get("sel_crudo") for r in con_mano]),
            "error_mm": mediana([r["error_mm"] for r in base if "error_mm" in r]),
            "sesgo_mm": np.mean([r["sesgo"] for r in base if "sesgo" in r], axis=0).tolist()
            if any("sesgo" in r for r in base) else None,
            "jitter_mm": float(np.sqrt(np.trace(np.cov(mm.T)))) if len(mm) >= 3 else None,
            "t_primer_acierto_ms": primero}


def prueba_objetivos(ctx, n=15, ventana_s=2.0):
    if ctx.datos is None:
        print("  (objetivos: falta la gráfica)")
        return None
    det = ctx.detector()
    reg = ctx.registro()
    sel = selector(ctx.cfg, ctx.datos, reg)
    objetivos = elegir_objetivos(ctx.datos, n)
    ctx.decir(f"Prueba de objetivos: {len(objetivos)} elementos. Para cada uno, pon la yema del índice encima, "
              "junta el dedo medio y, sin mover la mano, pulsa espacio. Mantén la pinza hasta oír el resultado.")
    intentos = []
    for i, obj in enumerate(objetivos, start=1):
        ctx.objetivo_actual = obj
        ok = ctx.esperar_espacio(f"Objetivo {i} de {len(objetivos)}: {obj['indicacion']}", mapa_obj=obj)
        if ok is False:
            break
        if ok is None:
            continue
        t = medir_objetivo(ctx, det, sel, obj, ventana_s, reg)
        intentos.append(t)
        if not t["frames"] or t["mano_pct"] == 0:
            ctx.decir("No vi la mano.")
        elif t["acierto_estricto"]:
            ctx.decir("Correcto.")
        else:
            el = next((e for e in sel.elementos if e["id"] == t["seleccionado"]), None)
            ctx.decir("Incorrecto: " + (el["corto"] if el else "no anunció nada") + ".")
    ctx.objetivo_actual = None
    m = resumir_objetivos(intentos)
    if m.get("intentos"):
        ctx.decir(f"Acierto: {m['acierto_pct']:.0f} por ciento en {m['intentos']} objetivos. "
                  f"Error mediano de la yema: {m['error_mediana_mm']:.0f} milímetros.")
    ctx.guardar("objetivos", m, detalle=intentos)
    return m


def prueba_falsos(ctx, segundos=15.0):
    if ctx.datos is None:
        print("  (falsos: falta la gráfica)")
        return None
    det = ctx.detector()
    reg = ctx.registro()
    sel = selector(ctx.cfg, ctx.datos, reg)
    fases = {}
    for fase, msg in (("sin_pinza", f"Pinzas falsas. Durante {int(segundos)} segundos recorre la gráfica solo con "
                                    "el índice, con los dedos bien separados. Pulsa espacio para empezar."),
                      ("con_pinza", f"Ahora, durante {int(segundos)} segundos, recorre las líneas con la pinza "
                                    "cerrada todo el tiempo. Pulsa espacio para empezar.")):
        ok = ctx.esperar_espacio(msg)
        if ok is False:
            break
        if ok is None:
            continue
        # Pinza nueva (Interaccion: palma, histéresis y gracia) y, para comparar, la de antes
        # (umbral fijo en px, sin gracia: un frame con los dedos separados la cortaba).
        sel.reiniciar()
        t0, n, con, pin, activ, cortes, anuncios = time.time(), 0, 0, 0, 0, 0, 0
        pin_px, activ_px, cortes_px = 0, 0, 0
        antes = antes_px = False
        anunciado, medidas = None, []
        while time.time() - t0 < segundos:
            _c, f = ctx.sesion.leer()
            if f is None:
                continue
            h, w = f.shape[:2]
            lm, mu = muestra_camara(ctx, det, reg, f)
            s = sel.actualizar(mu)
            n += 1
            pinza, pinza_px = bool(s.pinza), False
            if lm is not None:
                con += 1
                ind, med, _ = mano_px(lm, w, h)
                pinza_px = math.hypot(ind[0] - med[0], ind[1] - med[1]) <= ctx.cfg["distancia_pinza"]
                if sel.medida_pinza is not None:
                    medidas.append(sel.medida_pinza)
                pin += pinza                              # % de frames con mano, como antes
                pin_px += pinza_px
            activ += pinza and not antes
            cortes += antes and not pinza
            activ_px += pinza_px and not antes_px
            cortes_px += antes_px and not pinza_px
            if sel.anterior is not None and sel.anterior != anunciado:   # lo que la aplicación diría
                anuncios += 1
            anunciado = sel.anterior
            antes, antes_px = pinza, pinza_px
            ctx.mostrar(f, [f"{fase.replace('_', ' ')}: pinza en {pin}/{max(con, 1)} frames con mano "
                            f"(umbral antiguo: {pin_px})", "Esc: terminar"])
            if ctx.pantalla.tecla(1) == 27:
                break
        dur_min = max(1e-6, (time.time() - t0) / 60)
        fases[fase] = {"frames": n, "frames_con_mano": con, "frames_pinza": pin,
                       "pinza_pct": 100.0 * pin / con if con else None,
                       "activaciones_min": activ / dur_min, "cortes_min": cortes / dur_min,
                       "anuncios_min": anuncios / dur_min, "duracion_s": dur_min * 60,
                       "pinza_px_pct": 100.0 * pin_px / con if con else None,
                       "activaciones_px_min": activ_px / dur_min, "cortes_px_min": cortes_px / dur_min,
                       "medida_pinza": {"p10": percentil(medidas, 10), "p50": percentil(medidas, 50),
                                        "p90": percentil(medidas, 90)}}
    if not fases:
        return None
    m = {"fases": fases, "umbral_cerrar_palma": float(sel.c("pinza_cerrar_palma")),
         "umbral_abrir_palma": float(sel.c("pinza_abrir_palma")), "gracia_pinza_ms": sel.c("gracia_pinza_ms")}
    if "sin_pinza" in fases:
        m["falsos_pct"] = fases["sin_pinza"]["pinza_pct"]
        m["activaciones_falsas_min"] = fases["sin_pinza"]["activaciones_min"]
        m["anuncios_falsos_min"] = fases["sin_pinza"]["anuncios_min"]
        m["falsos_px_pct"] = fases["sin_pinza"]["pinza_px_pct"]
        m["activaciones_falsas_px_min"] = fases["sin_pinza"]["activaciones_px_min"]
        # Dedos separados: la medida debería quedar por encima del umbral de abrir.
        m["medida_sin_pinza_p10"] = fases["sin_pinza"]["medida_pinza"]["p10"]
    if "con_pinza" in fases:
        m["sensibilidad_pct"] = fases["con_pinza"]["pinza_pct"]
        m["cortes_min"] = fases["con_pinza"]["cortes_min"]
        m["anuncios_min"] = fases["con_pinza"]["anuncios_min"]
        m["sensibilidad_px_pct"] = fases["con_pinza"]["pinza_px_pct"]
        m["cortes_px_min"] = fases["con_pinza"]["cortes_px_min"]
        # Dedos juntos: la medida debería quedar por debajo del umbral de cerrar.
        m["medida_con_pinza_p90"] = fases["con_pinza"]["medida_pinza"]["p90"]
    ctx.guardar("falsos", m)
    return m


# ============================================================
# TAREAS DE COMPRENSIÓN (con voz, como la aplicación) y SUS
# ============================================================
SUS = ["Creo que me gustaría usar este sistema con frecuencia.",
       "Encontré el sistema innecesariamente complejo.",
       "Pensé que el sistema era fácil de usar.",
       "Creo que necesitaría el apoyo de una persona técnica para poder usar este sistema.",
       "Encontré que las distintas funciones del sistema estaban bien integradas.",
       "Pensé que había demasiada inconsistencia en el sistema.",
       "Imagino que la mayoría de las personas aprendería a usar este sistema muy rápidamente.",
       "Encontré el sistema muy engorroso de usar.",
       "Me sentí muy seguro o segura usando el sistema.",
       "Necesité aprender muchas cosas antes de poder empezar con este sistema."]


def preguntas_automaticas(datos):
    """Preguntas genéricas a partir del CSV, con la respuesta esperada (solo la ve quien evalúa)."""
    q = []
    series = series_de(datos)
    q.append(("¿Cuántas líneas o series tiene la gráfica?", str(len(series))))
    for s in series[:3]:
        pts = [e for e in datos.elementos if e["tipo"] == "punto" and e["serie"] == s]
        if pts:
            alto = min(pts, key=lambda e: e["y1"])            # en la placa, y crece hacia abajo
            q.append((f"¿Dónde alcanza {s} su valor más alto?", alto["texto"]))
    cr = cruces(datos)
    if cr:
        c = cr[0]
        e = next(x for x in datos.elementos if x["id"] == c["curvas"][0])
        q.append((f"¿Dónde se cruzan {c['series'][0]} y {c['series'][1]}?", e["texto"]))
    hs = huecos(datos)
    if hs:
        q.append(("¿La línea tiene saltos o tramos sin datos? ¿Dónde?",
                  "; ".join(f"{h['clase']} entre {h['desde']} y {h['hasta']}" for h in hs)))
    if len(series) == 1:
        pts = [e for e in datos.elementos if e["tipo"] == "punto"]
        if len(pts) >= 2:
            q.append(("En general, ¿la línea sube o baja de izquierda a derecha?",
                      "sube" if pts[-1]["y1"] < pts[0]["y1"] else "baja"))
    return q


def leer_preguntas(ruta):
    res = []
    with open(ruta, encoding="utf-8-sig") as f:
        for linea in f:
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                p, _, r = linea.partition("|")
                res.append((p.strip(), r.strip()))
    return res


def prueba_tareas(ctx, preguntas, sus=False):
    ex = eg.Explorador(ctx.voz, ctx.cfg, avisar=ctx.decir)      # exploración real, con su voz
    ex.configurar(ctx.datos, ctx.registro())                   # el mismo registro que usa la aplicación
    tareas = []
    for i, (pregunta, esperada) in enumerate(preguntas, start=1):
        texto = f"Pregunta {i} de {len(preguntas)}: {pregunta}"
        ctx.decir(texto + " Cuando quieras empezar, quien evalúa pulsa espacio.")
        estado, t0, anuncios, distintos, ultimo = "espera", None, 0, set(), None
        resultado = None
        while resultado is None:
            _c, f = ctx.sesion.leer()
            if f is not None:
                f, _vis = ex.procesar(f)
                if estado == "midiendo" and ex.anterior is not None and ex.anterior != ultimo:
                    anuncios += 1
                    distintos.add(ex.anterior)
                ultimo = ex.anterior
                linea2 = (f"Esperada: {esperada}" if esperada else "") + \
                         ("  ·  Espacio: empezar" if estado == "espera" else
                          f"  ·  {time.time() - t0:.0f} s · 1 bien · 0 mal · 2 parcial")
                ctx.mostrar(f, [texto, linea2, "R: repetir pregunta · N: saltar · Esc: terminar"])
            k = ctx.pantalla.tecla(1)
            if k == 32 and estado == "espera":
                estado, t0 = "midiendo", time.time()
                ctx.decir("Adelante.")
            elif k in (ord("r"), ord("R")):
                ctx.decir(texto)
            elif k in (ord("1"), ord("0"), ord("2")) and estado == "midiendo":
                resultado = {"1": "correcta", "0": "incorrecta", "2": "parcial"}[chr(k)]
            elif k in (ord("n"), ord("N")):
                resultado = "saltada"
            elif k == 27:
                resultado = "fin"
        if resultado == "fin":
            break
        tareas.append({"pregunta": pregunta, "esperada": esperada, "resultado": resultado,
                       "tiempo_s": (time.time() - t0) if t0 else None, "anuncios": anuncios,
                       "elementos_distintos": len(distintos)})
        ctx.voz.detener()
    m = {}
    hechas = [t for t in tareas if t["resultado"] in ("correcta", "incorrecta", "parcial")]
    if hechas:
        puntos = sum(1.0 if t["resultado"] == "correcta" else 0.5 if t["resultado"] == "parcial" else 0 for t in hechas)
        m = {"tareas": len(hechas), "acierto_pct": 100.0 * puntos / len(hechas),
             "acierto": proporcion(sum(t["resultado"] == "correcta" for t in hechas), len(hechas)),
             "tiempo_mediana_s": mediana([t["tiempo_s"] for t in hechas]),
             "anuncios_mediana": mediana([t["anuncios"] for t in hechas])}
    if sus:
        resp = []
        for j, item in enumerate(SUS, start=1):
            texto = f"Afirmación {j} de 10: {item} Responde de 1, muy en desacuerdo, a 5, muy de acuerdo."
            ctx.decir(texto)
            r = None
            while r is None:
                _c, f = ctx.sesion.leer()
                if f is not None:
                    ctx.mostrar(f, [f"SUS {j}/10: {item}", "Tecla 1 a 5 · R: repetir · Esc: terminar"])
                k = ctx.pantalla.tecla(15)
                if k in tuple(ord(c) for c in "12345"):
                    r = int(chr(k))
                elif k in (ord("r"), ord("R")):
                    ctx.decir(texto)
                elif k == 27:
                    r = 0
            if r == 0:
                break
            resp.append(r)
        if len(resp) == 10:
            m["sus"] = 2.5 * sum((r - 1) if j % 2 == 0 else (5 - r) for j, r in enumerate(resp))
            m["sus_respuestas"] = resp
            ctx.decir(f"Puntuación SUS: {m['sus']:.0f}.")
    ctx.guardar("tareas", m, detalle=tareas)
    ex.cerrar()
    return m


# ============================================================
# REPORTE
# ============================================================
# (grupo, etiqueta, prueba, clave, objetivo en texto, función que dice si se cumple, formato)
def num(v, dec=1):
    """Número con coma decimal, como en el resto de la interfaz."""
    return f"{v:.{dec}f}".replace(".", ",")


def _pct(v):
    return f"{v:.0f} %"


def _mm(v):
    return f"{num(v)} mm"


FILAS = [
    ("Selección", "Acierto de selección", "objetivos", "acierto_pct", "≥ 90 %", lambda v: v >= 90, "prop:acierto"),
    ("Selección", "· en zonas aisladas", "objetivos", "acierto_aislado_pct", "≥ 90 %", lambda v: v >= 90, "prop:acierto_aislado"),
    ("Selección", "· junto a cruces", "objetivos", "acierto_cruce_pct", "≥ 80 %", lambda v: v >= 80, "prop:acierto_cruce"),
    ("Selección", "· ejes", "objetivos", "acierto_eje_pct", "≥ 90 %", lambda v: v >= 90, "prop:acierto_eje"),
    ("Selección", "· textos y leyendas", "objetivos", "acierto_texto_pct", "≥ 90 %", lambda v: v >= 90, "prop:acierto_texto"),
    ("Selección", "Acierto de serie", "objetivos", "acierto_serie_pct", "≥ 95 %", lambda v: v >= 95, _pct),
    ("Selección", "Error de la yema (mediana)", "objetivos", "error_mediana_mm", "≤ 4 mm", lambda v: v <= 4, _mm),
    ("Selección", "Sesgo de la yema", "objetivos", "sesgo_modulo_mm", "", None, _mm),
    ("Selección", "Temblor", "objetivos", "jitter_mm", "≤ 1,5 mm", lambda v: v <= 1.5, _mm),
    ("Selección", "Cambios de elemento por objetivo", "objetivos", "cambios_sel_mediana", "≤ 1", lambda v: v <= 1,
     lambda v: num(v)),
    ("Selección", "· sin suavizado ni estabilidad", "objetivos", "cambios_crudo_mediana", "", None, lambda v: num(v)),
    ("Selección", "Acierto sin suavizado ni estabilidad", "objetivos", "acierto_crudo_pct", "", None, _pct),
    ("Registro", "Error de registro (mediana)", "registro", "error_mediana_mm", "≤ 3 mm", lambda v: v <= 3, _mm),
    ("Registro", "Error máximo en el gráfico", "registro", "error_max_grafico_mm", "≤ 6 mm", lambda v: v <= 6, _mm),
    ("Registro", "Ídem con autodetección (A)", "registro", "error_auto_max_grafico_mm", "≤ 6 mm", lambda v: v <= 6, _mm),
    ("Registro", "Autodetección (A), mediana", "registro", "error_auto_mediana_grafico_mm", "≤ 3 mm", lambda v: v <= 3, _mm),
    ("Pinza", "Sensibilidad (pinza mantenida)", "falsos", "sensibilidad_pct", "≥ 95 %", lambda v: v >= 95, _pct),
    ("Pinza", "Pinza falsa (% frames)", "falsos", "falsos_pct", "≤ 2 %", lambda v: v <= 2, _pct),
    ("Pinza", "Activaciones falsas", "falsos", "activaciones_falsas_min", "≤ 1 /min", lambda v: v <= 1, lambda v: f"{num(v)} /min"),
    ("Pinza", "Cortes con pinza cerrada", "falsos", "cortes_min", "≤ 2 /min", lambda v: v <= 2, lambda v: f"{num(v)} /min"),
    ("Pinza", "· con el umbral antiguo (px, sin gracia)", "falsos", "cortes_px_min", "", None, lambda v: f"{num(v)} /min"),
    ("Pinza", "Sensibilidad con el umbral antiguo", "falsos", "sensibilidad_px_pct", "", None, _pct),
    ("Pinza", "Índice–medio / palma, dedos juntos (p90)", "falsos", "medida_con_pinza_p90", "", None, lambda v: num(v, 2)),
    ("Pinza", "Índice–medio / palma, dedos separados (p10)", "falsos", "medida_sin_pinza_p10", "", None,
     lambda v: num(v, 2)),
    ("Mano", "Detección de la mano", "mano", "deteccion_pct", "≥ 95 %", lambda v: v >= 95, "prop:deteccion"),
    ("Mano", "Contraste mano–placa", "mano", "delta_e_mano_placa", "", None, lambda v: f"ΔE {v:.0f}"),
    ("Mano", "fps procesados", "mano", "fps_proceso", "≥ 20", lambda v: v >= 20, lambda v: f"{v:.0f}"),
    ("Mano", "fps que entrega la cámara", "mano", "fps_captura", "≥ 25", lambda v: v >= 25, lambda v: f"{v:.0f}"),
    ("Mano", "Latencia de visión (p95)", "mano", "latencia_vision_p95_ms", "≤ 100 ms", lambda v: v <= 100, lambda v: f"{v:.0f} ms"),
    ("Mano", "Umbral de pinza / palma", "mano", "umbral_pinza_sobre_palma", "", None, lambda v: num(v, 2)),
    ("Placa", "Autodetección", "hoja", "exito_pct", "≥ 95 %", lambda v: v >= 95, "prop:exito"),
    ("Placa", "· con confianza alta", "hoja", "confianza_alta_pct", "≥ 90 %", lambda v: v >= 90, _pct),
    ("Placa", "Esquina recortada correcta", "hoja", "esquina_ok_pct", "≥ 95 %", lambda v: v >= 95, _pct),
    ("Placa", "Rotación de la placa", "hoja", "rotacion_med_grados", "≤ 2°", lambda v: v <= 2, lambda v: f"{num(v)}°"),
    ("Placa", "Placa cortada por el encuadre", "hoja", "toca_borde_pct", "0 %", lambda v: v == 0, _pct),
    ("Placa", "Contraste de gris placa–fondo", "hoja", "delta_gris", "≥ 40", lambda v: v >= 40, lambda v: f"{v:.0f}"),
    ("Placa", "Contraste de color placa–fondo", "hoja", "delta_e", "≥ 20", lambda v: v >= 20, lambda v: f"ΔE {v:.0f}"),
    ("Placa", "Desfase vs. calibración guardada", "hoja", "desfase_guardada_mm", "≤ 3 mm", lambda v: v <= 3, _mm),
    ("Placa", "Dispersión al recolocar", "hoja", "reposicion_mm", "≤ 2 mm", lambda v: v <= 2, _mm),
    ("Placa", "Anisotropía de la calibración", "hoja", "anisotropia_calibracion", "0,95–1,05", lambda v: 0.95 <= v <= 1.05, lambda v: num(v, 2)),
    ("QR", "Lectura del QR", "qr", "tasa_pct", "≥ 80 %", lambda v: v >= 80, "prop:tasa"),
    ("QR", "Tiempo hasta la 1.ª lectura", "qr", "t_primera_s", "≤ 3 s", lambda v: v <= 3, lambda v: f"{num(v)} s"),
    ("Geometría", "Área ambigua (2+ series)", "geometria", "ambigua_pct_banda", "≤ 10 %", lambda v: v <= 10, _pct),
    ("Geometría", "Conflicto de prioridad", "geometria", "conflicto_pct_banda", "≤ 5 %", lambda v: v <= 5, _pct),
    ("Geometría", "Acierto esperado, yema ±4 mm, puntos aislados", "geometria", "esperado_4mm_aislado_punto_pct", "≥ 90 %", lambda v: v >= 90, _pct),
    ("Geometría", "Acierto esperado, yema ±4 mm, líneas aisladas", "geometria", "esperado_4mm_aislado_linea_pct", "≥ 90 %", lambda v: v >= 90, _pct),
    ("Geometría", "Acierto esperado, yema ±4 mm, cruces", "geometria", "esperado_4mm_cruce_pct", "≥ 80 %", lambda v: v >= 80, _pct),
    ("Geometría", "Acierto esperado, yema ±4 mm, ejes", "geometria", "esperado_4mm_eje_pct", "≥ 90 %", lambda v: v >= 90, _pct),
    ("Geometría", "Trazo simulado: serie seguida en cruces", "geometria", "trazo_completa_cruce_serie_pct", "≥ 90 %",
     lambda v: v >= 90, _pct),
    ("Geometría", "· sin estabilidad", "geometria", "trazo_sin_estado_cruce_serie_pct", "", None, _pct),
    ("Geometría", "Trazo simulado: cambios de elemento", "geometria", "trazo_completa_resto_cambios_s", "", None,
     lambda v: f"{num(v)} /s"),
    ("Geometría", "· sin estabilidad", "geometria", "trazo_sin_estado_resto_cambios_s", "", None, lambda v: f"{num(v)} /s"),
    ("Geometría", "Separación mínima entre series", "geometria", "sep_min_puntos_series_mm", "", None, _mm),
    ("Geometría", "Cruces entre series", "geometria", "n_cruces", "", None, lambda v: f"{v:.0f}"),
    ("Usuario", "Acierto en tareas", "tareas", "acierto_pct", "≥ 80 %", lambda v: v >= 80, _pct),
    ("Usuario", "Tiempo por tarea (mediana)", "tareas", "tiempo_mediana_s", "", None, lambda v: f"{v:.0f} s"),
    ("Usuario", "SUS", "tareas", "sus", "≥ 68", lambda v: v >= 68, lambda v: f"{v:.0f}"),
]


def cargar_resultados(carpeta):
    por_placa = {}
    for ruta in sorted(glob.glob(os.path.join(carpeta, "*", "*.json"))):
        try:
            with open(ruta, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        por_placa.setdefault(d.get("placa") or os.path.basename(os.path.dirname(ruta)), []).append(d)
    return por_placa


def consolidar(resultados):
    """Última sesión de cada prueba; los intentos de 'objetivos' se juntan de todas las sesiones."""
    res, sesiones = {}, {}
    for d in sorted(resultados, key=lambda d: d.get("fecha", "")):
        p = d.get("prueba")
        sesiones[p] = sesiones.get(p, 0) + 1
        res[p] = d
    intentos = [t for d in resultados if d.get("prueba") == "objetivos" for t in d.get("detalle", [])]
    if intentos:
        res["objetivos"] = dict(res["objetivos"])
        res["objetivos"]["metricas"] = resumir_objetivos(intentos)
    return res, sesiones


def _grafica_de(res):
    cond = next((d.get("condiciones") for d in res.values() if d.get("condiciones")), {}) or {}
    return str(cond.get("grafica") or "").strip().lower() or None


def compartir_geometria(consol):
    """La prueba 'geometria' solo depende del CSV (y casi nada de la cámara): si una placa no la
    tiene y otra de la MISMA gráfica (--grafica) sí, se muestra esa. Devuelve {placa: de cuál}."""
    prestada = {}
    for p, (res, _ses) in consol.items():
        g = _grafica_de(res)
        if "geometria" in res or g is None:
            continue
        for q, (rq, _s) in consol.items():
            if q != p and "geometria" in rq and _grafica_de(rq) == g:
                res["geometria"] = rq["geometria"]
                prestada[p] = q
                break
    return prestada


def formatear(valor, fmt, metricas):
    if valor is None:
        return "—"
    if isinstance(fmt, str) and fmt.startswith("prop:"):
        pr = metricas.get(fmt[5:]) or {}
        lo, hi = pr.get("ic95", [None, None])
        txt = f"{valor:.0f} %"
        if pr.get("n"):
            txt += f" (n={pr['n']}" + (f"; IC95 {100 * lo:.0f}–{100 * hi:.0f}" if lo is not None else "") + ")"
        return txt
    return fmt(valor)


def generar_reporte(carpeta=CARPETA_METRICAS):
    por_placa = cargar_resultados(carpeta)
    if not por_placa:
        print("No hay resultados en", carpeta)
        return None
    placas = sorted(por_placa)
    consol = {p: consolidar(por_placa[p]) for p in placas}
    prestada = compartir_geometria(consol)
    lineas = [f"# Reporte de métricas del explorador de gráficas", "",
              f"Generado el {datetime.datetime.now():%Y-%m-%d %H:%M} con medir_metricas.py {VERSION}.", ""]
    lineas.append("| Grupo | Métrica | Objetivo | " + " | ".join(placas) + " |")
    lineas.append("| --- | --- | --- | " + " | ".join("---" for _ in placas) + " |")
    filas_csv = []
    for grupo, etiqueta, prueba, clave, objetivo, cumple, fmt in FILAS:
        celdas, alguna = [], False
        for p in placas:
            d = consol[p][0].get(prueba)
            met = d.get("metricas", {}) if d else {}
            v = met.get(clave)
            if v is not None:
                alguna = True
            marca = ""
            if v is not None and cumple is not None:
                ok = bool(cumple(v))
                marca = " ✓" if ok else " ✗"
            else:
                ok = None
            celdas.append(formatear(v, fmt, met) + marca)
            filas_csv.append([p, grupo, etiqueta, clave, "" if v is None else v, objetivo,
                              "" if ok is None else ("sí" if ok else "no")])
        if alguna:
            lineas.append(f"| {grupo} | {etiqueta} | {objetivo} | " + " | ".join(celdas) + " |")
    lineas += ["", "## Condiciones y sesiones", ""]
    for p in placas:
        res, ses = consol[p]
        cond = next((d.get("condiciones") for d in res.values() if d.get("condiciones")), {})
        graf = next((d.get("grafica") for d in res.values() if d.get("grafica")), {})
        partes = [", ".join(f"{k} = {v}" for k, v in cond.items())] if cond else []
        if graf:
            partes.append(f"gráfica: {graf.get('titulo')}")
        partes.append("sesiones: " + ", ".join(f"{k} ×{v}" for k, v in sorted(ses.items())))
        if p in prestada:
            partes.append(f"geometría tomada de {prestada[p]} (misma gráfica: solo depende del CSV)")
        lineas.append(f"- **{p}**: " + "; ".join(partes))
        conf = (res.get("objetivos") or {}).get("metricas", {}).get("confusiones")
        if conf:
            lineas.append("  - Confusiones más frecuentes: " + "; ".join(f"{c} (×{n})" for c, n in conf[:5]))
    lineas += ["", "✓ cumple el objetivo propuesto · ✗ no lo cumple · — sin medir. "
               "Las proporciones llevan su intervalo de confianza del 95 % (Wilson)."]
    ruta_md = os.path.join(carpeta, "reporte_metricas.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    ruta_csv = os.path.join(carpeta, "resumen_metricas.csv")
    with open(ruta_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["placa", "grupo", "metrica", "clave", "valor", "objetivo", "cumple"])
        w.writerows(filas_csv)
    print("\n".join(lineas))
    print("\nReporte:", ruta_md, "\nTabla:  ", ruta_csv)
    return ruta_md


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
def cargar_datos(entrada):
    if not entrada:
        return None
    try:
        d = eg.cargar_desde_enlace(entrada)
    except eg.ErrorDatos as e:
        sys.exit(f"No pude cargar la gráfica: {e}")
    print(f"Gráfica: {d.titulo} · {len(series_de(d))} series · {len(d.elementos)} elementos ({d.origen})")
    return d


def hoja_de_args(args, cfg, datos):
    if getattr(args, "hoja", None):
        return tuple(int(float(v)) for v in args.hoja.split(","))
    if getattr(args, "px_por_mm", None) and datos is not None:
        k = args.px_por_mm
        return (0, 0, int(round(datos.ancho_ref * k)), int(round(datos.alto_ref * k)))
    return tuple(int(v) for v in cfg["hoja"])


def modo_geometria(args):
    cfg = eg.cargar_config()
    datos = cargar_datos(args.datos)
    if datos is None:
        sys.exit("El modo geometria necesita --datos (ruta o enlace del CSV).")
    ctx = Contexto(args, cfg, datos, voz=None)
    ctx.hoja = hoja_de_args(args, cfg, datos)
    ctx.calibracion = "indicada con --hoja/--px-por-mm" if (args.hoja or args.px_por_mm) else ctx.calibracion
    r = analizar_geometria(cfg, datos, ctx.hoja, imagen=ctx.ruta(f"{marca_tiempo()}_mapa_seleccion.png"))
    ctx.guardar("geometria", r)
    rm = r["radios_mm"]
    pp = r["pinza_palma"]
    print(f"\nRadios en la placa: punto {rm['punto_config']:.1f} mm (efectivo mín. {rm['punto_efectivo_min']:.1f}), "
          f"línea {rm['linea']:.1f} mm, eje {rm['eje']:.1f} mm, texto {rm['texto']:.1f} mm · "
          f"pinza: cierra a {pp['cerrar']:.2f} y abre a {pp['abrir']:.2f} × palma")
    print(f"Cruces: {r['n_cruces']} · huecos: {r['n_huecos']} · separación mínima entre series: "
          f"{r['sep_min_puntos_series_mm']} mm")
    amb, con = r["ambigua_pct_banda"], r["conflicto_pct_banda"]
    print(f"Área ambigua: {amb or 0:.1f} % · conflicto de prioridad: {con or 0:.1f} % · "
          f"anisotropía de la calibración: {r['anisotropia_calibracion']:.2f}")
    print("Acierto esperado (estricto / por serie) si la yema cae con un error de:")
    for sigma, zonas in r["acierto_esperado"].items():
        print(f"  ±{sigma} mm: " + " · ".join(f"{z} {v['estricto_pct']:.0f} / {v['serie_pct']:.0f} %"
                                             for z, v in sorted(zonas.items())))
    tz = r.get("trazos", {}).get("modos", {})
    if tz:
        print("Yema simulada recorriendo cada serie (temblor ±%.1f mm, %.0f fps): serie correcta · cambios/s"
              % (r["trazos"]["ruido_mm"], r["trazos"]["fps"]))
        for modo, nombre in (("sin_estado", "sin estabilidad"), ("completa", "aplicación")):
            print(f"  {nombre:>16}: " + " · ".join(
                f"{z} {v['serie_pct'] or 0:.0f} % · {v['cambios_s']:.1f}/s" for z, v in sorted(tz.get(modo, {}).items())))
    return r


def modo_imagen(args):
    cfg = eg.cargar_config()
    datos = cargar_datos(args.datos)
    frame = cv2.imread(args.foto)
    if frame is None:
        sys.exit(f"No pude abrir la imagen {args.foto}")
    if args.orientacion:
        frame = eg.aplicar_orientacion(frame, args.orientacion)
    voz = eg.Voz(cfg["velocidad_voz"], False, cfg.get("voz", "")) if args.registro else None
    ctx = Contexto(args, cfg, datos, voz)
    ctx.hoja = hoja_de_args(args, cfg, datos)
    if args.hoja or args.px_por_mm:                    # se pidió un rectángulo: sin esquinas guardadas
        ctx.esquinas, ctx.origen_registro, ctx.calibracion = None, "rect", "indicada con --hoja/--px-por-mm"
    reg = ctx.registro()
    m = {"imagen": os.path.basename(args.foto), "tamano": [frame.shape[1], frame.shape[0]],
         "placa": analizar_placa(frame, datos, reg), "qr": analizar_qr(eg.LectorQR(), frame)}
    det = m["placa"].get("_det")
    if datos is not None:
        vista = dibujar_elementos(frame.copy(), escalar(ctx))
        cv2.polylines(vista, [poligono_px(reg)], True, (0, 255, 255), 1)
        if det is not None:
            cv2.polylines(vista, [poligono_px(det.esquinas)], True, (0, 255, 0), 1)
        cv2.imwrite(ctx.ruta(f"{marca_tiempo()}_superposicion.png"), vista)
    if args.registro and datos is not None:
        ctx.pantalla = Pantalla()
        refs = puntos_referencia(datos)
        clics = pedir_clics(ctx, frame, refs)
        m["registro"] = calcular_registro(datos, reg, refs, clics, det.registro(datos) if det else None)
        if "error" not in m["registro"]:
            cv2.imwrite(ctx.ruta(f"{marca_tiempo()}_registro.png"), imagen_registro(frame, m["registro"]))
        ctx.pantalla.cerrar()
    ctx.guardar("imagen", m)
    print(json.dumps(limpiar({k: v for k, v in m.items() if k != "registro"}), ensure_ascii=False, indent=1)[:3000])
    return m


PRUEBAS = ("hoja", "qr", "registro", "mano", "objetivos", "falsos")


def preparar_camara(args):
    cfg = eg.cargar_config()
    datos = cargar_datos(args.datos)
    voz = eg.Voz(cfg["velocidad_voz"], cfg.get("depurar_voz", False), cfg.get("voz", ""))
    try:
        sesion = SesionCamara(cfg, voz, args.orientacion, getattr(args, "camara", None))
    except ErrorCamara as e:
        print("[voz]", e)
        voz.decir(str(e), interrumpir=True, esencial=True)
        time.sleep(min(20.0, 2 + len(str(e)) * 0.07))         # que alcance a decirlo antes de salir
        voz.cerrar()
        sys.exit(1)
    ctx = Contexto(args, cfg, datos, voz, Pantalla(), sesion)
    print(f"Cámara: {sesion.motor.clave_camara} (número {sesion.motor.indice}) · orientación "
          f"{sesion.orientacion} · formato {eg.fourcc_texto(sesion.cap)} · calibración {ctx.calibracion}")
    if not confirmar_camara(ctx):
        ctx.sesion.cerrar()
        ctx.pantalla.cerrar()
        voz.cerrar()
        otras = "; ".join(f"{i}: {n}" for i, n in enumerate(sesion.nombres)) or "0, 1, 2…"
        sys.exit(f"Elige la cámara con --camara seguido de su número ({otras}).")
    return ctx


def confirmar_camara(ctx):
    """Antes de medir: la persona vidente comprueba que la imagen es la de la cámara sobre la placa."""
    nombre = ctx.sesion.motor.clave_camara
    cual = nombre if nombre.startswith("cámara") else f"cámara {nombre}"
    ok = ctx.esperar_espacio(f"Uso la {cual}. Si en la pantalla se ve la placa desde arriba, pulsa "
                             "espacio. Si no es la cámara correcta, pulsa Escape.")
    return ok is not False


def hoja_automatica(ctx, intentos=40):
    for _ in range(intentos):
        _c, f = ctx.sesion.leer()
        if f is None:
            continue
        det = eg.detectar_placa(f, ctx.datos)
        if det is not None:
            ctx.usar_deteccion(det, f"autodetectada al empezar (como la tecla A): {det.metodo}, "
                                    f"confianza {det.confianza}")
            print("Placa autodetectada:", np.round(det.esquinas, 1).tolist(), f"({det.metodo}, {det.confianza})")
            return True
    ctx.decir("No pude autodetectar la placa; uso la calibración guardada.")
    return False


def modo_camara(args):
    pruebas = [p.strip() for p in args.pruebas.split(",") if p.strip()]
    malas = [p for p in pruebas if p not in PRUEBAS]
    if malas:
        sys.exit("Pruebas desconocidas: " + ", ".join(malas) + ". Válidas: " + ", ".join(PRUEBAS))
    ctx = preparar_camara(args)
    resumen = {}
    try:
        if args.hoja_auto:
            hoja_automatica(ctx)
        if ctx.datos is None and "qr" not in pruebas:
            pruebas.insert(0, "qr")
        conds = [c.strip() for c in args.qr_condiciones.split(",") if c.strip()]
        for p in pruebas:
            print(f"\n=== Prueba: {p} ===")
            try:
                if p == "hoja":
                    resumen[p] = prueba_hoja(ctx, args.repeticiones)
                elif p == "qr":
                    resumen[p] = prueba_qr(ctx, conds, args.segundos_qr)
                elif p == "registro":
                    resumen[p] = prueba_registro(ctx)
                elif p == "mano":
                    resumen[p] = prueba_mano(ctx, args.segundos)
                elif p == "objetivos":
                    resumen[p] = prueba_objetivos(ctx, args.objetivos, args.ventana)
                elif p == "falsos":
                    resumen[p] = prueba_falsos(ctx, args.segundos_falsos)
            except eg.ErrorSeguimiento as e:
                ctx.decir(str(e))
                ctx.guardar(p, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                ctx.guardar(p, {"error": repr(e)})
            if p == "qr" and ctx.datos is None and any(q in pruebas for q in ("registro", "objetivos", "falsos")):
                ctx.decir("No tengo la gráfica porque el código QR no se leyó, así que las pruebas de registro, "
                          "objetivos y falsos no se pueden hacer. Vuelve a ejecutar el comando añadiendo "
                          "guion guion datos y el enlace de la gráfica.")
                print("  Ejemplo: py medir_metricas.py camara --placa salmon_g2 --color claro --grafica 2 "
                      "--pruebas objetivos,falsos --datos https://tactonauta.onrender.com/q/54D7E01D5C")
        ctx.decir("Pruebas terminadas. Para ver la comparación entre placas ejecuta: medir métricas, reporte.")
    finally:
        ctx.cerrar()
        ctx.sesion.cerrar()
        ctx.pantalla.cerrar()
        try:
            ctx.voz.cerrar()
        except Exception:
            pass
    return resumen


def _medir_fps(cap, frames=60, calentar=15, limite_s=12.0):
    """(fps, brillo medio, formato) leyendo 'frames' imágenes tras descartar 'calentar'."""
    for _ in range(calentar):
        cap.read()
    t0, n, brillo, ultimo = time.perf_counter(), 0, [], None
    while n < frames and time.perf_counter() - t0 < limite_s:
        ok, f = cap.read()
        if not ok or f is None or not f.size:
            continue
        n += 1
        if n % 10 == 0:
            brillo.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
        ultimo = f
    dt = time.perf_counter() - t0
    tam = f"{ultimo.shape[1]}x{ultimo.shape[0]}" if ultimo is not None else "?"
    return (n / dt if dt > 0 else 0.0), (float(np.mean(brillo)) if brillo else None), eg.fourcc_texto(cap), tam


def modo_fps(args):
    """Por qué la cámara entrega pocos fps: prueba formato (MJPG / sin comprimir) y exposición."""
    cfg = eg.cargar_config()
    nombres = nombres_camaras()
    idx, nombre, _ = elegir_camara(cfg, args.camara, nombres)
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    ancho, alto = cfg.get("resolucion", [640, 480])
    exps = []
    for v in args.exposiciones.split(","):
        v = v.strip().lower()
        if v:
            exps.append(None if v in ("auto", "automatica", "automática") else float(v))
    print(f"Cámara {idx}: {nombre} · {ancho}x{alto} · cierra el explorador y cualquier otro programa que la use.")
    if args.ajustes:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            print("Abro el panel de ajustes del driver: busca 'compensación de poca luz' (Low light "
                  "compensation) y desactívala; luego cierra el panel y pulsa Enter aquí.")
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
            try:
                input()
            except EOFError:
                pass
            cap.release()
    filas = []
    for mjpg in (True, False):
        for exp in exps:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                print("  no pude abrir la cámara (¿la tiene otro programa?)")
                return None
            try:
                eg.fijar_formato(cap, ancho, alto, mjpg)
                cap.set(cv2.CAP_PROP_FPS, 30.0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if exp is None:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # automática (V4L2); DirectShow:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # una de las dos según el driver
                else:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
                fps, brillo, formato, tam = _medir_fps(cap)
            finally:
                cap.release()
            fila = {"mjpg": mjpg, "exposicion": exp, "fps": fps, "brillo": brillo, "formato": formato, "tam": tam}
            filas.append(fila)
            print(f"  pedido {'MJPG' if mjpg else 'sin comprimir'} · exposición "
                  f"{'automática' if exp is None else f'{exp:g}'}: {fps:4.1f} fps · brillo "
                  f"{'?' if brillo is None else f'{brillo:3.0f}'} · entrega {formato} {tam}")
    # El driver recuerda la última exposición: se deja la de la configuración actual.
    cap = cv2.VideoCapture(idx, backend)
    if cap.isOpened():
        actual = cfg.get("exposicion")
        if actual is None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(actual))
        cap.read()
        cap.release()
    buenas = [f for f in filas if f["fps"] >= 25 and (f["brillo"] or 0) >= 60]
    print()
    if buenas:
        mejor = max(buenas, key=lambda f: ((f["brillo"] or 0) >= 90, f["exposicion"] is None, f["mjpg"], f["brillo"]))
        print("Recomendado en config_exploracion.json:")
        print(f'  "mjpg": {"true" if mejor["mjpg"] else "false"},')
        exp = "null" if mejor["exposicion"] is None else f"{mejor['exposicion']:g}"
        print(f'  "exposicion": {exp}')
    else:
        rapidas = [f for f in filas if f["fps"] >= 25]
        if rapidas:
            print("Con 25 fps o más la imagen queda muy oscura: hace falta más luz sobre la bandeja "
                  "(una lámpara LED encima), o aceptar 15 fps.")
        else:
            print("En ninguna combinación pasa de 25 fps. Suele ser la 'compensación de poca luz' del "
                  "driver: ejecuta  py medir_metricas.py fps --ajustes  y desactívala. Si sigue igual, "
                  "es el máximo de esta cámara a esa resolución.")
    return filas


def modo_tareas(args):
    ctx = preparar_camara(args)
    try:
        if args.hoja_auto:
            hoja_automatica(ctx)
        if ctx.datos is None:
            prueba_qr(ctx, ["en la bandeja"], args.segundos_qr)
        if ctx.datos is None:
            sys.exit("Sin gráfica: el QR no se leyó. Vuelve a ejecutarlo con --datos y el enlace de la gráfica, "
                     "p. ej.: py medir_metricas.py tareas --placa turquesa_g3 --sus --datos <enlace del QR>")
        preguntas = leer_preguntas(args.preguntas) if args.preguntas else preguntas_automaticas(ctx.datos)
        return prueba_tareas(ctx, preguntas, args.sus)
    finally:
        ctx.cerrar()
        ctx.sesion.cerrar()
        ctx.pantalla.cerrar()
        try:
            ctx.voz.cerrar()
        except Exception:
            pass


def construir_argumentos():
    p = argparse.ArgumentParser(description="Métricas del explorador de gráficas con textura.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="modo", required=True)

    def comunes(sp, datos_obligatorio=False):
        sp.add_argument("--placa", required=True, help="nombre corto de la placa impresa, p. ej. morado_g2")
        sp.add_argument("--datos", required=datos_obligatorio, help="CSV de la gráfica: ruta local o enlace (el del QR)")
        sp.add_argument("--color", help="color de la placa (oscuro, claro, medio...)")
        sp.add_argument("--grafica", help="qué gráfica es (p. ej. 2)")
        sp.add_argument("--fondo", help="sobre qué está la placa (p. ej. bandeja azul)")
        sp.add_argument("--luz", help="iluminación (p. ej. ventana, fluorescente)")
        sp.add_argument("--nota", help="cualquier comentario de la sesión")

    g = sub.add_parser("geometria", help="análisis del CSV sin cámara")
    comunes(g, datos_obligatorio=True)
    g.add_argument("--hoja", help="x,y,ancho,alto de la hoja en px (por defecto, la de config_exploracion.json)")
    g.add_argument("--px-por-mm", type=float, help="alternativa a --hoja: escala de la cámara sobre la placa")

    c = sub.add_parser("camara", help="batería de pruebas con la cámara")
    comunes(c)
    c.add_argument("--pruebas", default=",".join(PRUEBAS), help="lista separada por comas: " + ", ".join(PRUEBAS))
    c.add_argument("--objetivos", type=int, default=15, help="número de objetivos (15)")
    c.add_argument("--ventana", type=float, default=2.0, help="segundos de medición por objetivo (2)")
    c.add_argument("--repeticiones", type=int, default=3, help="veces que se recoloca la placa en la prueba hoja (3)")
    c.add_argument("--segundos", type=float, default=20.0, help="duración de la prueba mano (20)")
    c.add_argument("--segundos-falsos", type=float, default=15.0, help="duración de cada fase de falsos (15)")
    c.add_argument("--segundos-qr", type=float, default=15.0, help="tiempo máximo por condición de QR (15)")
    c.add_argument("--qr-condiciones", default="en la bandeja,en la mano a unos 20 cm",
                   help="condiciones del QR separadas por comas")
    c.add_argument("--hoja-auto", action="store_true", help="autodetectar la placa al empezar (como la tecla A)")
    c.add_argument("--orientacion", choices=sorted(eg.ORIENTACIONES) if eg else None,
                   help="forzar la orientación de la imagen")
    c.add_argument("--camara", help="qué cámara usar: su número (0, 1...) o parte de su nombre; "
                                    "por defecto, nombre_camara de config_exploracion.json")

    t = sub.add_parser("tareas", help="preguntas de comprensión con exploración real y SUS")
    comunes(t)
    t.add_argument("--preguntas", help="archivo de texto: una pregunta por línea, opcional '| respuesta esperada'")
    t.add_argument("--sus", action="store_true", help="aplicar el cuestionario SUS al final")
    t.add_argument("--hoja-auto", action="store_true")
    t.add_argument("--segundos-qr", type=float, default=15.0)
    t.add_argument("--orientacion", choices=sorted(eg.ORIENTACIONES) if eg else None)
    t.add_argument("--camara", help="número o parte del nombre de la cámara")

    i = sub.add_parser("imagen", help="analizar una foto o captura guardada")
    comunes(i)
    i.add_argument("--foto", required=True)
    i.add_argument("--registro", action="store_true", help="marcar puntos con clics para medir el error de calibración")
    i.add_argument("--hoja", help="x,y,ancho,alto de la hoja en px")
    i.add_argument("--px-por-mm", type=float)
    i.add_argument("--orientacion", choices=sorted(eg.ORIENTACIONES) if eg else None)

    fp = sub.add_parser("fps", help="probar formatos y exposición de la cámara (fps reales)")
    fp.add_argument("--camara", help="número o parte del nombre de la cámara")
    fp.add_argument("--ajustes", action="store_true", help="abrir el panel de ajustes del driver (Windows)")
    fp.add_argument("--exposiciones", default="auto,-4,-5,-6,-7,-8", help="valores a probar")

    r = sub.add_parser("reporte", help="juntar los resultados de todas las placas")
    r.add_argument("--carpeta", default=CARPETA_METRICAS)
    return p


def main(argv=None):
    importar_explorador()
    args = construir_argumentos().parse_args(argv)
    if args.modo == "geometria":
        return modo_geometria(args)
    if args.modo == "imagen":
        return modo_imagen(args)
    if args.modo == "camara":
        return modo_camara(args)
    if args.modo == "tareas":
        return modo_tareas(args)
    if args.modo == "fps":
        return modo_fps(args)
    return generar_reporte(args.carpeta)


if __name__ == "__main__":
    main()
