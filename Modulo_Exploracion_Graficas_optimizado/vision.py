#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISIÓN Y REGISTRO  ·  Explorador de gráficas impresas con textura  (Persona A)
=============================================================================

Todo lo que ocurre en píxeles de la cámara. Hacia fuera solo salen milímetros
de la placa (origen arriba a la izquierda, el mismo lienzo que DatosGrafica):

    Registro        px de la imagen  <->  mm de la placa (homografía)
    MuestraMano     posición de los dedos en mm, una por frame
    detectar_placa  las 4 esquinas de la placa en la imagen (gris, color y bordes)
    Superposicion   dibujo de la gráfica sobre el video, con el elemento resaltado
    DetectorMano    MediaPipe (API Tasks), opcionalmente recortado a la placa
    LectorQR        lectura del QR (con presupuesto de tiempo y variante "relieve")
    Captura         hilo que conserva solo el último frame de la cámara
    MotorVision     hilo de la cámara: QR, registro, mano y superposición

Lo que decide qué elemento se anuncia (pinza, radios, prioridades, voz) es de la
Persona B: MotorVision recibe un objeto con configurar(datos, registro) y
actualizar(muestra) -> Seleccion, y solo lee de la Seleccion .pinza y .elemento_id.

Este módulo no importa nada del explorador (no hay importaciones circulares).
"""

from __future__ import annotations

import base64
import http.client
import math
import os
import queue
import sys
import threading
import time
import traceback
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================
# CONSTANTES
# ============================================================
def _carpeta_base():
    """Junto a este script; si no se puede escribir ahí, en ~/ExploradorGraficas
    (la misma regla que el explorador)."""
    aqui = os.path.dirname(os.path.abspath(__file__))
    if os.access(aqui, os.W_OK):
        return aqui
    alt = os.path.join(os.path.expanduser("~"), "ExploradorGraficas")
    os.makedirs(alt, exist_ok=True)
    return alt


CARPETA_MODELOS = os.path.join(_carpeta_base(), "modelos")

# Modelo oficial de MediaPipe para el seguimiento de la mano (API Tasks).
URL_MODELO_MANO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/latest/hand_landmarker.task")
NOMBRE_MODELO_MANO = "hand_landmarker.task"

VISTA_DEFECTO = (800, 600)             # caja inicial del video (ancho, alto) en px

# Esquinas en el orden que usa todo este módulo: el de las agujas del reloj desde arriba a la izquierda.
NOMBRES_ESQUINAS = ("superior izquierda", "superior derecha", "inferior derecha", "inferior izquierda")
BORDES_PLACA = ("superior", "derecho", "inferior", "izquierdo")   # borde i va de la esquina i a la i+1


def _ascii(s):
    """Las fuentes de OpenCV no dibujan acentos ni flechas: versión ASCII."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("→", "->").replace("°", " ").replace("«", '"').replace("»", '"')
    return s.encode("ascii", "replace").decode()


# ============================================================
# REGISTRO: píxeles de la imagen <-> milímetros de la placa
# ============================================================
def _aplicar_h(H, pts):
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:3]


class Registro:
    """Dónde está cada milímetro de la placa en la imagen (ya orientada).

    Por dentro es una homografía H que lleva mm del lienzo a px. Con la placa vista
    desde arriba y algo girada o inclinada, una homografía es exacta; el antiguo
    rectángulo 'hoja' solo lo es si la placa está perfectamente alineada con la
    imagen (cada grado de giro sumaba unos 4 mm de error en el extremo del gráfico).

    Interfaz acordada con la Persona B (no cambiar sin acuerdo):
        imagen_a_placa(u_px, v_px) -> (x_mm, y_mm)
        placa_a_imagen(x_mm, y_mm) -> (u_px, v_px) enteros, para dibujar
        error_estimado_mm          float | None  (None = no se sabe)
    """

    def __init__(self, H, ancho_mm, alto_mm, origen="rect", error_estimado_mm=None, esquinas_px=None):
        H = np.asarray(H, dtype=float)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
            raise ValueError("homografía no válida")
        self.H = H / H[2, 2]
        self.Hinv = np.linalg.inv(self.H)
        self.ancho_mm, self.alto_mm = float(ancho_mm), float(alto_mm)
        self.origen = origen                       # "rect" | "autodeteccion" | "clics" | "qr" | ...
        self.error_estimado_mm = error_estimado_mm
        # Esquinas físicas de la placa (px) si se conocen; en el formato clásico el lienzo no es la placa.
        self._esquinas = None if esquinas_px is None else np.asarray(esquinas_px, float).reshape(4, 2)
        # Coeficientes sueltos: la conversión de un solo punto se hace en cada frame.
        self._d = [float(v) for v in self.H.ravel()]
        self._i = [float(v) for v in self.Hinv.ravel()]

    # ---------- contrato ----------
    def imagen_a_placa(self, u_px, v_px):
        a = self._i
        w = a[6] * u_px + a[7] * v_px + a[8]
        return (a[0] * u_px + a[1] * v_px + a[2]) / w, (a[3] * u_px + a[4] * v_px + a[5]) / w

    def placa_a_imagen(self, x_mm, y_mm):
        u, v = self.placa_a_imagen_f(x_mm, y_mm)
        return int(round(u)), int(round(v))

    # ---------- extras para A (y para quien los quiera) ----------
    def placa_a_imagen_f(self, x_mm, y_mm):
        a = self._d
        w = a[6] * x_mm + a[7] * y_mm + a[8]
        return (a[0] * x_mm + a[1] * y_mm + a[2]) / w, (a[3] * x_mm + a[4] * y_mm + a[5]) / w

    def placa_a_imagen_arr(self, pts_mm):
        return _aplicar_h(self.H, pts_mm)

    def imagen_a_placa_arr(self, pts_px):
        return _aplicar_h(self.Hinv, pts_px)

    def esquinas_px(self):
        """Esquinas de la placa en la imagen (sup-izq, sup-der, inf-der, inf-izq), float."""
        if self._esquinas is not None:
            return self._esquinas.copy()
        W, H = self.ancho_mm, self.alto_mm
        return self.placa_a_imagen_arr([[0, 0], [W, 0], [W, H], [0, H]])

    def caja_px(self):
        """Rectángulo (x, y, ancho, alto) que envuelve la placa: es la antigua 'hoja'."""
        e = self.esquinas_px()
        x0, y0 = np.floor(e.min(axis=0))
        x1, y1 = np.ceil(e.max(axis=0))
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    def px_por_mm(self):
        """Escala media en el centro de la placa (informativa)."""
        cx, cy = self.ancho_mm / 2, self.alto_mm / 2
        p = self.placa_a_imagen_arr([[cx - 5, cy], [cx + 5, cy], [cx, cy - 5], [cx, cy + 5]])
        return float((np.hypot(*(p[1] - p[0])) + np.hypot(*(p[3] - p[2]))) / 20.0)

    def dentro(self, x_mm, y_mm, margen_mm=0.0):
        return (-margen_mm <= x_mm <= self.ancho_mm + margen_mm and
                -margen_mm <= y_mm <= self.alto_mm + margen_mm)

    def bordes_fuera(self, ancho_img, alto_img, margen_px=2.0):
        """Bordes de la placa ("superior", "derecho", ...) que se salen de la imagen o la
        tocan. Si un borde se sale entero, sus vecinos también asoman un poco: se nombran
        solo los que se salen en más de la mitad (o, si ninguno, el que más se sale)."""
        e = self.esquinas_px()
        fraccion = []
        for i in range(4):
            a, b = e[i], e[(i + 1) % 4]
            t = np.linspace(0.0, 1.0, 21)[:, None]
            p = a + t * (b - a)
            fuera = ((p[:, 0] < margen_px) | (p[:, 1] < margen_px) |
                     (p[:, 0] > ancho_img - 1 - margen_px) | (p[:, 1] > alto_img - 1 - margen_px))
            fraccion.append(float(fuera.mean()))
        if max(fraccion) == 0:
            return []
        if max(fraccion) >= 0.5:
            return [BORDES_PLACA[i] for i in range(4) if fraccion[i] >= 0.5]
        return [BORDES_PLACA[i] for i in range(4) if fraccion[i] == max(fraccion)]

    def transformado(self, M, origen=None):
        """El mismo registro tras una transformación de la imagen (M: 3×3, px → px).
        Sirve para girar la imagen (tecla O) o para corregir un desplazamiento."""
        M = np.asarray(M, float)
        esq = None if self._esquinas is None else _aplicar_h(M, self._esquinas)
        return Registro(M @ self.H, self.ancho_mm, self.alto_mm, origen or self.origen,
                        self.error_estimado_mm, esq)

    def diferencia_mm(self, otro, x0=None, y0=None, x1=None, y1=None, n=15):
        """Cuánto discrepan dos registros: para cada punto de la placa, dónde lo
        pondría el otro. Devuelve (mediana, máximo) en mm sobre una rejilla."""
        x0 = 0.0 if x0 is None else x0
        y0 = 0.0 if y0 is None else y0
        x1 = self.ancho_mm if x1 is None else x1
        y1 = self.alto_mm if y1 is None else y1
        gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
        mm = np.c_[gx.ravel(), gy.ravel()]
        otra = otro.imagen_a_placa_arr(self.placa_a_imagen_arr(mm))
        d = np.hypot(*(otra - mm).T)
        return float(np.median(d)), float(d.max())

    def __repr__(self):
        e = self.esquinas_px().round(1).tolist()
        err = "?" if self.error_estimado_mm is None else f"{self.error_estimado_mm:.1f} mm"
        return f"Registro({self.origen}, esquinas={e}, error≈{err})"

    # ---------- construcción ----------
    @classmethod
    def desde_rect(cls, rect, ancho_mm, alto_mm, estirar=True, origen="rect"):
        """El antiguo rectángulo 'hoja' (x, y, ancho, alto), con la misma fórmula que
        transformacion(): estirar=True ocupa la hoja entera (formato handtracking);
        estirar=False usa un solo factor, centrado (formato clásico)."""
        hx, hy, hw, hh = (float(v) for v in rect)
        if estirar:
            H = [[hw / ancho_mm, 0, hx], [0, hh / alto_mm, hy], [0, 0, 1]]
            esq = None
        else:
            e = min(hw / ancho_mm, hh / alto_mm)
            mx, my = hx + (hw - ancho_mm * e) / 2, hy + (hh - alto_mm * e) / 2
            H = [[e, 0, mx], [0, e, my], [0, 0, 1]]
            esq = [[hx, hy], [hx + hw, hy], [hx + hw, hy + hh], [hx, hy + hh]]
        return cls(H, ancho_mm, alto_mm, origen, None, esq)

    @classmethod
    def desde_esquinas(cls, esquinas_px, ancho_mm, alto_mm, estirar=True, origen="esquinas",
                       error_estimado_mm=None):
        """Las 4 esquinas de la placa en la imagen, en el orden sup-izq, sup-der,
        inf-der, inf-izq DE LA PLACA (la esquina recortada, donde se juntarían sus
        dos bordes)."""
        dst = np.asarray(esquinas_px, dtype=np.float32).reshape(4, 2)
        if estirar:
            src = np.array([[0, 0], [ancho_mm, 0], [ancho_mm, alto_mm], [0, alto_mm]], np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
        else:
            # Formato clásico: el lienzo se encaja, sin deformar y centrado, dentro de la placa.
            wq = (np.hypot(*(dst[1] - dst[0])) + np.hypot(*(dst[2] - dst[3]))) / 2
            hq = (np.hypot(*(dst[3] - dst[0])) + np.hypot(*(dst[2] - dst[1]))) / 2
            e = min(wq / ancho_mm, hq / alto_mm)
            mx, my = (wq - ancho_mm * e) / 2, (hq - alto_mm * e) / 2
            unidad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
            Hq = cv2.getPerspectiveTransform(unidad, dst)
            A = np.array([[e / wq, 0, mx / wq], [0, e / hq, my / hq], [0, 0, 1]])
            H = Hq @ A
        return cls(H, ancho_mm, alto_mm, origen, error_estimado_mm, None if estirar else dst)

    @classmethod
    def desde_puntos(cls, mm, px, ancho_mm, alto_mm, origen="puntos"):
        """Ajuste por mínimos cuadrados con 4 o más pares (mm, px); con 3, afín."""
        mm = np.asarray(mm, np.float64).reshape(-1, 2)
        px = np.asarray(px, np.float64).reshape(-1, 2)
        if len(mm) >= 4:
            H, _ = cv2.findHomography(mm, px, 0)
        elif len(mm) == 3:
            A, _ = cv2.estimateAffine2D(mm, px)
            H = None if A is None else np.vstack([A, [0, 0, 1]])
        else:
            H = None
        if H is None:
            raise ValueError("los puntos no permiten calcular el registro")
        r = cls(H, ancho_mm, alto_mm, origen)
        r.error_estimado_mm = float(np.sqrt(np.mean(np.sum((r.imagen_a_placa_arr(px) - mm) ** 2, axis=1)))) \
            if len(mm) > 4 else None
        return r


def registro_para(datos, esquinas=None, hoja=None, origen=None, error_estimado_mm=None):
    """El registro de una gráfica: con esquinas si las hay; si no, con el rectángulo 'hoja'."""
    if datos is None:
        return None
    if esquinas is not None:
        return Registro.desde_esquinas(esquinas, datos.ancho_ref, datos.alto_ref, datos.estirar,
                                       origen or "esquinas", error_estimado_mm)
    if hoja is not None:
        return Registro.desde_rect(hoja, datos.ancho_ref, datos.alto_ref, datos.estirar, origen or "rect")
    return None


def ordenar_esquinas(pts):
    """4 puntos en cualquier orden → sup-izq, sup-der, inf-der, inf-izq de la imagen
    (en el sentido de las agujas del reloj). Vale para giros de menos de 45°."""
    p = np.asarray(pts, float).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]                       # en la imagen (y hacia abajo) esto es horario
    i0 = int(np.argmin(p[:, 0] + p[:, 1]))
    return np.roll(p, -i0, axis=0)


def esquinas_validas(valor):
    """Esquinas guardadas en la configuración → array 4×2, o None si no sirven."""
    try:
        e = np.asarray(valor, float).reshape(4, 2)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(e)) or abs(cv2.contourArea(e.astype(np.float32))) < 100:
        return None
    return e


# ============================================================
# MUESTRA DE LA MANO (contrato con B)
# ============================================================
@dataclass
class MuestraMano:            # una por frame
    t: float                  # time.monotonic()
    indice_mm: tuple[float, float] | None
    medio_mm: tuple[float, float] | None
    palma_mm: float | None    # para normalizar la pinza
    dentro_placa: bool


def dedos_px(puntos, ancho, alto):
    """(índice, medio, muñeca, base del dedo medio) en px de la imagen."""
    def p(i):
        return puntos[i].x * ancho, puntos[i].y * alto
    return p(8), p(12), p(0), p(9)


def muestra_de_mano(puntos, ancho, alto, registro, t=None):
    """Una mano de MediaPipe (21 puntos normalizados a la imagen completa) → MuestraMano en mm.
    Sin mano o sin registro, los campos van en None (pero la muestra existe: lleva su t)."""
    t = time.monotonic() if t is None else t
    if puntos is None or registro is None:
        return MuestraMano(t, None, None, None, False)
    ind, med, muneca, base = dedos_px(puntos, ancho, alto)
    i_mm = registro.imagen_a_placa(*ind)
    m_mm = registro.imagen_a_placa(*med)
    a, b = registro.imagen_a_placa(*muneca), registro.imagen_a_placa(*base)
    palma = math.hypot(b[0] - a[0], b[1] - a[1])
    return MuestraMano(t, (float(i_mm[0]), float(i_mm[1])), (float(m_mm[0]), float(m_mm[1])),
                       float(palma), bool(registro.dentro(*i_mm)))


# ============================================================
# AUTODETECCIÓN DE LA PLACA (4 esquinas)
# ============================================================
@dataclass
class DeteccionPlaca:
    esquinas: np.ndarray          # 4×2 px, en el orden de la PLACA (sup-izq, sup-der, inf-der, inf-izq)
    rect: tuple                   # rectángulo envolvente (x, y, ancho, alto), recortado a la imagen
    esquina_vista: str | None     # posición EN LA IMAGEN de la esquina recortada que se ve
    giro: int                     # 0..3: cuartos de vuelta que se corrigieron con la esquina recortada
    metodo: str                   # qué segmentación la encontró
    confianza: str                # "alta" | "media"
    puntuacion: float
    residuo_px: float             # RMS del ajuste de los lados
    reconstruida: bool            # un lado estaba fuera de la imagen y se reconstruyó
    bordes_imagen: list           # lados de la IMAGEN que corta la placa ("arriba", ...)
    area_pct: float
    rotacion_grados: float
    error_estimado_mm: float | None
    contorno: np.ndarray | None = None

    def registro(self, datos, origen="autodeteccion"):
        return Registro.desde_esquinas(self.esquinas, datos.ancho_ref, datos.alto_ref, datos.estirar,
                                       origen, self.error_estimado_mm)


def _cerrar(m, k):
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _abrir(m, k):
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _rellenar(m):
    """Rellena los huecos de cada región (las texturas de la placa dejan agujeros)."""
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, c, -1, 255, -1)
    return out


def _mascaras(frame, metodos, solo=None):
    """Segmentaciones candidatas: (clave, máscara 0/255). Cada una puede fallar con
    algún fondo; detectar_placa se queda con el mejor cuadrilátero de todas.
      gris    umbral de Otsu sobre el gris (el método antiguo): falla si el fondo
              tiene el mismo gris que la placa (bandeja azul, inserto claro).
      color   distancia de color (Lab) al fondo, estimado en el borde de la imagen.
      kmeans  grupos de color (Lab): separa placa, bandeja e inserto aunque el gris coincida.
      bordes  bordes de Canny en L y en color: cada región que encierran es candidata.
    Con 'solo' (una clave ya vista) se calcula solo esa máscara: sirve para repetir el
    método ganador a resolución completa (k-means da los mismos grupos: se agrupa
    siempre sobre una copia de 160 px, con la misma semilla)."""
    h, w = frame.shape[:2]
    k = max(3, int(round(min(h, w) / 160)) * 2 + 1)
    cache = {}

    def lab():
        if "lab" not in cache:
            l = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2Lab).astype(np.float32)
            l[..., 0] *= 0.6               # la luz cambia sobre todo L: se le da menos peso
            cache["lab"] = l
        return cache["lab"]

    def quiere(clave):
        return solo is None or solo == clave or (solo[0] == "bordes" and clave[0] == "bordes")

    if "gris" in metodos and (solo is None or solo[0] == "gris"):
        gris = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        _, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for inv in (False, True):
            if quiere(("gris", inv)):
                yield ("gris", inv), (255 - bw if inv else bw)
    if "color" in metodos and quiere(("color",)):
        L = lab()
        b = max(4, int(0.04 * min(h, w)))
        anillo = np.concatenate([L[:b].reshape(-1, 3), L[-b:].reshape(-1, 3),
                                 L[:, :b].reshape(-1, 3), L[:, -b:].reshape(-1, 3)])
        fondo = np.median(anillo, axis=0)
        dist = np.sqrt(((L - fondo) ** 2).sum(axis=2))
        tope = float(np.percentile(dist, 99.5)) or 1.0
        d8 = np.clip(dist * (255.0 / tope), 0, 255).astype(np.uint8)
        _, bw = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield ("color",), _rellenar(_cerrar(_abrir(bw, k), k))
    if "kmeans" in metodos and (solo is None or solo[0] == "kmeans"):
        L = lab()
        peq = cv2.resize(L, (160, max(1, int(160 * h / w))), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        plano = L.reshape(-1, 3)
        for n in (3, 4):
            if solo is not None and solo[1] != n:
                continue
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
            cv2.setRNGSeed(n)
            _, _, centros = cv2.kmeans(np.ascontiguousarray(peq), n, None, crit, 2, cv2.KMEANS_PP_CENTERS)
            d = ((plano[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
            etiquetas = d.argmin(axis=1).reshape(h, w)
            for c in range(n):
                if quiere(("kmeans", n, c)):
                    m = (etiquetas == c).astype(np.uint8) * 255
                    yield ("kmeans", n, c), _rellenar(_cerrar(_abrir(m, k), k))
    if "bordes" in metodos and (solo is None or solo[0] == "bordes"):
        l8 = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2Lab)
        e = cv2.Canny(l8[..., 0], 30, 90) | cv2.Canny(l8[..., 1], 12, 36) | cv2.Canny(l8[..., 2], 12, 36)
        e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2)
        e[:2, :] = e[-2:, :] = 255             # el marco cierra las placas que tocan el borde
        e[:, :2] = e[:, -2:] = 255
        # Cada región sin bordes es una candidata. La de la placa es el anillo entre su
        # contorno y lo que tiene dibujado dentro: su contorno exterior es el de la placa.
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(255 - e, connectivity=4)
        grandes = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.03 * w * h]
        for j, i in enumerate(sorted(grandes, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:4]):
            m = (etiquetas == i).astype(np.uint8) * 255
            # Devuelve los 2 px que se comió la dilatación de los bordes.
            yield ("bordes", j), cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)


def _interseccion(l1, l2):
    """Intersección de dos rectas (vx, vy, x0, y0)."""
    (a1, b1, x1, y1), (a2, b2, x2, y2) = l1, l2
    den = a1 * b2 - b1 * a2
    if abs(den) < 1e-9:
        return None
    t = ((x2 - x1) * b2 - (y2 - y1) * a2) / den
    return np.array([x1 + t * a1, y1 + t * b1])


def _ajustar_lado(pts, a, b, usar):
    """Recta de un lado: ajuste robusto a los puntos del contorno cercanos al tramo a→b,
    sin los extremos (allí están la esquina recortada y el redondeo de las esquinas)."""
    d = b - a
    L = float(np.hypot(*d))
    if L < 1:
        return None, None, None
    u = d / L
    rel = pts - a
    t = rel @ u / L
    dist = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    sel = usar & (t > 0.12) & (t < 0.88) & (dist < max(3.0, 0.02 * L))
    q = pts[sel]
    if len(q) < 8:
        q = np.array([a, b], float)
    vx, vy, x0, y0 = cv2.fitLine(q.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    res = np.abs((q[:, 0] - x0) * vy - (q[:, 1] - y0) * vx)
    return (float(vx), float(vy), float(x0), float(y0)), float(np.sqrt(np.mean(res ** 2))), q.mean(axis=0)


def _cuadrilatero(contorno, w, h, aspecto):
    """Del contorno de una región, las 4 esquinas (virtuales) de la placa: se ajusta una
    recta a cada lado largo y se cortan. La esquina recortada queda donde se juntarían
    los dos bordes. Si un lado entero queda fuera de la imagen y se conoce el aspecto de
    la placa, se reconstruye. Devuelve (esquinas 4×2 en orden horario, residuo_px, reconstruida)."""
    pts = contorno.reshape(-1, 2).astype(float)
    en_borde = (pts[:, 0] <= 2) | (pts[:, 1] <= 2) | (pts[:, 0] >= w - 3) | (pts[:, 1] >= h - 3)
    hull = cv2.convexHull(contorno)
    aprox = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(float)
    n = len(aprox)
    if n < 3 or n > 12:
        return None

    def borde(p):
        return p[0] <= 3 or p[1] <= 3 or p[0] >= w - 4 or p[1] >= h - 4

    tramos, en_marco = [], set()
    for i in range(n):
        a, b = aprox[i], aprox[(i + 1) % n]
        if borde(a) and borde(b) and borde((a + b) / 2):
            en_marco.add(i)                            # tramo sobre el marco de la imagen
            continue
        tramos.append((i, a, b, float(np.hypot(*(b - a)))))
    if not tramos:
        return None
    largo = max(t[3] for t in tramos)
    lados = [t for t in tramos if t[3] >= 0.2 * largo]
    if len(lados) > 4:
        lados = sorted(sorted(lados, key=lambda t: -t[3])[:4], key=lambda t: t[0])
    if len(lados) < 3:
        return None
    if len(lados) == 3:
        # El lado que falta está donde hay un tramo sobre el marco: se ordena para que
        # quede entre el último lado y el primero.
        for k in range(3):
            a, b = lados[k][0], lados[(k + 1) % 3][0]
            entre = [(a + j) % n for j in range(1, (b - a) % n)]
            if any(j in en_marco for j in entre):
                lados = lados[k + 1:] + lados[:k + 1]
                break
        else:
            return None
    rectas, residuos, centros = [], [], []
    for _i, a, b, _L in lados:
        r, res, c = _ajustar_lado(pts, a, b, ~en_borde)
        if r is None:
            return None
        rectas.append(r)
        residuos.append(res)
        centros.append(c)
    reconstruida = False
    if len(rectas) == 4:
        esq = [_interseccion(rectas[i], rectas[(i + 1) % 4]) for i in range(4)]
        if any(e is None for e in esq):
            return None
        esq = np.array(esq)
    else:
        if aspecto is None:
            return None
        # Falta un lado (fuera de la imagen): entre los lados 2 y 0, opuesto al lado 1.
        # Asume una perspectiva suave (cámara cenital), como el resto del sistema.
        c01, c12 = _interseccion(rectas[0], rectas[1]), _interseccion(rectas[1], rectas[2])
        if c01 is None or c12 is None:
            return None
        u0 = centros[0] - c01
        u2 = centros[2] - c12
        if np.hypot(*u0) < 1 or np.hypot(*u2) < 1:
            return None
        u0, u2 = u0 / np.hypot(*u0), u2 / np.hypot(*u2)
        L1 = float(np.hypot(*(c12 - c01)))
        horizontal = abs(rectas[1][0]) >= abs(rectas[1][1])
        d1, d0 = (aspecto, 1.0) if horizontal else (1.0, aspecto)   # en unidades del alto de la placa
        s = L1 / d1
        esq = np.array([c01, c12, c12 + u2 * d0 * s, c01 + u0 * d0 * s])
        reconstruida = True
    area = 0.5 * float(np.sum(esq[:, 0] * np.roll(esq[:, 1], -1) - np.roll(esq[:, 0], -1) * esq[:, 1]))
    if area < 0:
        esq = esq[::-1]
    i0 = int(np.argmin(esq[:, 0] + esq[:, 1]))
    esq = np.roll(esq, -i0, axis=0)
    return esq, float(np.sqrt(np.mean(np.square(residuos)))), reconstruida


def _angulos(esq):
    res = []
    for i in range(4):
        a, b, c = esq[i - 1], esq[i], esq[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosv = float(v1 @ v2 / (np.hypot(*v1) * np.hypot(*v2) + 1e-9))
        res.append(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))))
    return res


def _esquina_recortada_vista(esq, pts_utiles, w, h, recorte_px):
    """Índice (en la imagen) de la esquina que queda lejos del contorno, o None."""
    if len(pts_utiles) < 20:
        return None
    dist = []
    for e in esq:
        if e[0] < 4 or e[1] < 4 or e[0] > w - 5 or e[1] > h - 5:
            dist.append(-1.0)                        # fuera de la imagen: no se puede saber
            continue
        dist.append(float(np.min(np.hypot(pts_utiles[:, 0] - e[0], pts_utiles[:, 1] - e[1]))))
    orden = sorted(range(4), key=lambda i: -dist[i])
    d1, d2 = dist[orden[0]], max(dist[orden[1]], 1.0)
    lado = min(np.hypot(*(esq[1] - esq[0])), np.hypot(*(esq[3] - esq[0])))
    if recorte_px:
        if 0.4 * recorte_px <= d1 <= 2.2 * recorte_px and d1 >= 2.0 * d2:
            return orden[0]
        return None
    if d1 >= 0.02 * lado and d1 >= 2.5 * d2:
        return orden[0]
    return None


def _evaluar(contorno, w, h, aspecto, esperada, recorte_mm, dims_mm):
    """Puntuación de un contorno como placa (o None si no lo es)."""
    area_c = float(cv2.contourArea(contorno))
    if area_c < 0.06 * w * h or area_c > 0.985 * w * h:
        return None
    r = _cuadrilatero(contorno, w, h, aspecto)
    if r is None:
        return None
    esq, residuo, reconstruida = r
    q = esq.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(q):
        return None
    if any(abs(a - 90) > 25 for a in _angulos(esq)):
        return None
    lados = [float(np.hypot(*(esq[(i + 1) % 4] - esq[i]))) for i in range(4)]
    ancho_px, alto_px = (lados[0] + lados[2]) / 2, (lados[1] + lados[3]) / 2
    if min(lados) < 0.15 * min(w, h):
        return None
    if np.max(np.abs(esq)) > 3 * max(w, h):
        return None
    marco = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32).reshape(-1, 1, 2)
    area_q, _ = cv2.intersectConvexConvex(q, marco)
    if area_q <= 0:
        return None
    relleno = area_c / area_q
    if not 0.9 <= relleno <= 1.04:
        return None
    puntos = 1.0 - 3.0 * abs(1.0 - relleno) - 0.08 * residuo
    if aspecto is not None:
        r_asp = ancho_px / alto_px
        err = min(abs(math.log(r_asp / aspecto)), abs(math.log(r_asp * aspecto)))
        if err > 0.2:
            return None
        puntos -= 2.0 * err
    pts = contorno.reshape(-1, 2).astype(float)
    utiles = pts[~((pts[:, 0] <= 2) | (pts[:, 1] <= 2) | (pts[:, 0] >= w - 3) | (pts[:, 1] >= h - 3))]
    recorte_px = None
    if recorte_mm and dims_mm:
        s = (ancho_px / dims_mm[0] + alto_px / dims_mm[1]) / 2
        recorte_px = recorte_mm / math.sqrt(2) * s
    i_rec = _esquina_recortada_vista(esq, utiles, w, h, recorte_px)
    if esperada:
        puntos += 0.6 if i_rec is not None else -0.2
    elif i_rec is not None:
        puntos -= 0.3
    if reconstruida:
        puntos -= 0.4
    fuera = bool((esq[:, 0] < 0).any() or (esq[:, 1] < 0).any() or (esq[:, 0] > w - 1).any() or (esq[:, 1] > h - 1).any())
    if fuera:
        puntos -= 0.1
    bordes = []
    for nombre, sel in (("arriba", pts[:, 1] <= 2), ("abajo", pts[:, 1] >= h - 3),
                        ("izquierda", pts[:, 0] <= 2), ("derecha", pts[:, 0] >= w - 3)):
        if sel.sum() >= 5:
            bordes.append(nombre)
    return {"esq": esq, "residuo": residuo, "reconstruida": reconstruida, "puntos": puntos,
            "i_rec": i_rec, "relleno": relleno, "bordes": bordes, "contorno": contorno,
            "s": (ancho_px + alto_px) / 2}


def detectar_placa(frame, datos=None, metodos=("gris", "color", "kmeans", "bordes"), preferido=None):
    """Busca la placa y devuelve una DeteccionPlaca (o None).

    Prueba varias segmentaciones (ver _mascaras) y se queda con el cuadrilátero que
    mejor encaja: lados rectos, ángulos casi rectos, el aspecto de la placa (si se
    conoce 'datos') y la esquina recortada donde la placa dice. Con 'preferido' se
    prueba primero ese método y, si da confianza alta, no se prueban los demás.

    Las esquinas salen en el orden DE LA PLACA: si la esquina recortada se ve en otra
    posición (placa girada), se reordenan para que el registro siga siendo correcto, y
    'giro' dice cuántos cuartos de vuelta se corrigieron."""
    h, w = frame.shape[:2]
    aspecto = dims = None
    esperada, recorte_mm = "", 0.0
    if datos is not None:
        dims = (float(datos.ancho_ref), float(datos.alto_ref))
        if getattr(datos, "estirar", True):
            aspecto = dims[0] / dims[1]
        esperada = getattr(datos, "esquina_recortada", "") or ""
        recorte_mm = float(getattr(datos, "recorte_mm", 0.0) or 0.0)
    orden = list(metodos)
    preferido = preferido.rstrip("0123456789") if preferido else None     # "kmeans3" → "kmeans"
    if preferido in orden:
        orden.remove(preferido)
        orden.insert(0, preferido)

    def buscar(img, met, solo=None):
        hh, ww = img.shape[:2]
        res = []
        for clave, m in _mascaras(img, (met,), solo):
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in sorted(cs, key=cv2.contourArea, reverse=True)[:3]:
                r = _evaluar(c, ww, hh, aspecto, esperada, recorte_mm, dims)
                if r is not None:
                    res.append((r, clave))
        return res

    # 1) Búsqueda a baja resolución (rápida) con todos los métodos.
    f = min(1.0, 320.0 / w)
    peq = frame if f == 1.0 else cv2.resize(frame, (int(round(w * f)), int(round(h * f))),
                                            interpolation=cv2.INTER_AREA)
    mejor, clave_mejor = None, None
    for i, met in enumerate(orden):
        for r, clave in buscar(peq, met):
            if mejor is None or r["puntos"] > mejor["puntos"]:
                mejor, clave_mejor = r, clave
        if i == 0 and preferido and mejor is not None and _confianza(mejor, esperada) == "alta":
            break
    if mejor is None or mejor["puntos"] < 0:
        return None
    # 2) El método ganador, otra vez a resolución completa: esquinas con precisión de px.
    if f < 1.0:
        esq_peq = mejor["esq"] / f
        fino = [r for r, _c in buscar(frame, clave_mejor[0], clave_mejor)
                if np.max(np.hypot(*(r["esq"] - esq_peq).T)) < 4.0 / f]
        if fino:
            mejor = max(fino, key=lambda r: r["puntos"])
        else:                                  # no se repitió: se usa la de baja resolución, escalada
            mejor = dict(mejor, esq=esq_peq, residuo=mejor["residuo"] / f, s=mejor["s"] / f,
                         contorno=(mejor["contorno"] / f).astype(np.int32))
    metodo_mejor = clave_mejor[0] + (str(clave_mejor[1]) if clave_mejor[0] == "kmeans" else "")

    esq_img = mejor["esq"]
    giro, vista = 0, None
    if mejor["i_rec"] is not None:
        vista = NOMBRES_ESQUINAS[mejor["i_rec"]]
        if esperada in NOMBRES_ESQUINAS:
            giro = (mejor["i_rec"] - NOMBRES_ESQUINAS.index(esperada)) % 4
    esq_placa = np.array([esq_img[(p + giro) % 4] for p in range(4)])
    x0, y0 = np.clip(np.floor(esq_img.min(axis=0)), 0, None)
    x1, y1 = np.ceil(esq_img.max(axis=0))
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    rot = math.degrees(math.atan2(*(esq_placa[1] - esq_placa[0])[::-1]))
    err = None
    if dims:
        err = mejor["residuo"] / max(mejor["s"] / max(dims), 1e-6) * 1.5
        err += 3.0 if mejor["reconstruida"] else 0.0
    area_q = abs(cv2.contourArea(esq_img.astype(np.float32)))
    return DeteccionPlaca(
        esquinas=esq_placa, rect=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)), esquina_vista=vista,
        giro=giro, metodo=metodo_mejor, confianza=_confianza(mejor, esperada), puntuacion=float(mejor["puntos"]),
        residuo_px=float(mejor["residuo"]), reconstruida=bool(mejor["reconstruida"]),
        bordes_imagen=list(mejor["bordes"]), area_pct=100.0 * area_q / (w * h), rotacion_grados=float(rot),
        error_estimado_mm=err, contorno=mejor["contorno"])


def _confianza(r, esperada):
    ok_esquina = (r["i_rec"] is not None) if esperada else True
    if ok_esquina and not r["reconstruida"] and r["residuo"] < 2.0 and abs(1 - r["relleno"]) < 0.03:
        return "alta"
    return "media"


def detectar_hoja(frame, datos=None):
    """Compatibilidad con el código anterior: ((x, y, ancho, alto), esquina_recortada_vista)
    o (None, None). Ahora usa detectar_placa (varias segmentaciones)."""
    d = detectar_placa(frame, datos)
    if d is None:
        return None, None
    return d.rect, d.esquina_vista


def corregir_con_qr(registro, qr_px, qr_mm):
    """Corrección rígida (giro + desplazamiento) de un registro con el QR: lleva el centro
    y la dirección del QR que predice el registro a donde se ve. No usa el tamaño del QR,
    así no depende de si el recuadro del CSV incluye o no el margen blanco del código.
    qr_px: 4 esquinas vistas (orden del propio QR); qr_mm: las mismas en la placa."""
    pred = registro.placa_a_imagen_arr(qr_mm)
    vis = np.asarray(qr_px, float).reshape(4, 2)
    cp, cv_ = pred.mean(axis=0), vis.mean(axis=0)
    ap = math.atan2(*((pred[1] - pred[0]) + (pred[2] - pred[3]))[::-1])
    av = math.atan2(*((vis[1] - vis[0]) + (vis[2] - vis[3]))[::-1])
    da = av - ap
    c, s = math.cos(da), math.sin(da)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T1 = np.array([[1, 0, -cp[0]], [0, 1, -cp[1]], [0, 0, 1]])
    T2 = np.array([[1, 0, cv_[0]], [0, 1, cv_[1]], [0, 0, 1]])
    return T2 @ R @ T1, math.degrees(da), float(np.hypot(*(cv_ - cp)))


def residuo_qr_mm(registro, qr_px, qr_mm):
    """Dónde cae (en mm) el centro del QR visto frente al del CSV, y su tamaño visto en mm.
    Sirve para estimar el error del registro y para revisar el CSV."""
    vis_mm = registro.imagen_a_placa_arr(qr_px)
    qm = np.asarray(qr_mm, float)
    d_centro = float(np.hypot(*(vis_mm.mean(axis=0) - qm.mean(axis=0))))
    lado_visto = float(np.mean([np.hypot(*(vis_mm[(i + 1) % 4] - vis_mm[i])) for i in range(4)]))
    lado_csv = float(np.mean([np.hypot(*(qm[(i + 1) % 4] - qm[i])) for i in range(4)]))
    return d_centro, lado_visto, lado_csv


def esquinas_qr_mm(datos):
    """Las 4 esquinas (sup-izq, sup-der, inf-der, inf-izq) del elemento 'qr' del CSV, o None."""
    if datos is None:
        return None
    e = next((e for e in datos.elementos if str(e.get("id", "")).lower() == "qr"), None)
    if e is None or e["forma"] != "rect":
        return None
    x0, x1 = sorted((e["x1"], e["x2"]))
    y0, y1 = sorted((e["y1"], e["y2"]))
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


# ============================================================
# CÁMARA
# ============================================================
def listar_camaras():
    """Nombres de las cámaras en el orden de DirectShow (Windows, con pygrabber); si no, []."""
    if not sys.platform.startswith("win"):
        return []
    def leer():
        # pygrabber usa COM, que hay que iniciar en cada hilo. Si otro módulo (pyttsx3) importó
        # comtypes desde otro hilo, en este no se inició: "No se ha llamado a CoInitialize".
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass                                   # ya iniciado (en este u otro modo): se sigue
        from pygrabber.dshow_graph import FilterGraph
        return list(FilterGraph().get_input_devices())

    try:
        dispositivos = leer()
    except Exception as e:
        # Si en este hilo COM ya se inició en un modo incompatible, en un hilo nuevo empieza de cero.
        res = {}

        def en_hilo():
            try:
                res["d"] = leer()
            except Exception as e2:
                res["e"] = e2
        t = threading.Thread(target=en_hilo, daemon=True)
        t.start()
        t.join(5)
        if "d" not in res:
            print("Aviso: no pude leer los nombres de las cámaras:", repr(e), "/", repr(res.get("e")))
            return []
        dispositivos = res["d"]
    print("Cámaras detectadas:")
    for i, n in enumerate(dispositivos):
        print(f"  [{i}] {n}")
    return dispositivos


# ------------------------------------------------------------
# Orientación de la imagen de la cámara
# ------------------------------------------------------------
# Cada cámara puede estar montada de una forma (cenital, del revés, de lado) y
# algunos drivers entregan la imagen en espejo. La orientación se guarda POR
# CÁMARA y se corrige sola al leer el QR (giros), o a mano con la tecla O.
ORIENTACIONES = {
    "ninguno": "sin girar",
    "180": "girada 180 grados",
    "90": "girada 90 grados a la derecha",
    "270": "girada 90 grados a la izquierda",
    "horizontal": "en espejo de izquierda a derecha",
    "vertical": "en espejo de arriba abajo",
    "transpuesta": "girada y en espejo, por la diagonal",
    "antitranspuesta": "girada y en espejo, por la otra diagonal",
}
GIROS = ("ninguno", "90", "180", "270")


def aplicar_orientacion(frame, modo):
    if modo == "horizontal":
        return cv2.flip(frame, 1)
    if modo == "vertical":
        return cv2.flip(frame, 0)
    if modo == "180":
        return cv2.flip(frame, -1)
    if modo == "90":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if modo == "270":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if modo == "transpuesta":
        return cv2.transpose(frame)
    if modo == "antitranspuesta":
        return cv2.flip(cv2.transpose(frame), -1)
    return frame


def matriz_orientacion(modo, w, h):
    """Matriz 3×3 que lleva un punto de la imagen original (w × h) a la imagen orientada."""
    a, b = w - 1, h - 1
    m = {"ninguno": [[1, 0, 0], [0, 1, 0]],
         "horizontal": [[-1, 0, a], [0, 1, 0]],
         "vertical": [[1, 0, 0], [0, -1, b]],
         "180": [[-1, 0, a], [0, -1, b]],
         "90": [[0, -1, b], [1, 0, 0]],
         "270": [[0, 1, 0], [-1, 0, a]],
         "transpuesta": [[0, 1, 0], [1, 0, 0]],
         "antitranspuesta": [[0, -1, b], [-1, 0, a]]}.get(modo, [[1, 0, 0], [0, 1, 0]])
    return np.array(m + [[0, 0, 1]], dtype=float)


def cambio_orientacion(modo_antes, modo_despues, w, h):
    """Matriz que lleva un punto de la imagen orientada 'antes' a la orientada 'después'."""
    return matriz_orientacion(modo_despues, w, h) @ np.linalg.inv(matriz_orientacion(modo_antes, w, h))


def convertir_rect(rect, modo_antes, modo_despues, w, h):
    """La hoja calibrada (x, y, ancho, alto) en la orientación 'antes' → la misma zona
    de la imagen en la orientación 'después'. Así no hace falta recalibrar al girar."""
    x, y, rw, rh = rect
    esquinas = np.array([[x, y, 1], [x + rw, y, 1], [x, y + rh, 1], [x + rw, y + rh, 1]], dtype=float).T
    p = (cambio_orientacion(modo_antes, modo_despues, w, h) @ esquinas)[:2]
    x0, y0 = p.min(axis=1)
    x1, y1 = p.max(axis=1)
    return (int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0)))


def orientacion_desde_qr(pts):
    """Con las esquinas del QR en la imagen original (sup-izq, sup-der, inf-der, inf-izq
    del propio código), el giro que deja el QR derecho. Solo giros: con una imagen
    en espejo OpenCV devuelve las esquinas como si estuviera girada, así que el
    espejo no se puede deducir del QR (para eso está la tecla O)."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(p) != 4:
        return None
    der, abajo = p[1] - p[0], p[3] - p[0]
    if np.linalg.norm(der) < 1 or np.linalg.norm(abajo) < 1:
        return None
    der, abajo = der / np.linalg.norm(der), abajo / np.linalg.norm(abajo)
    mejor, puntos = None, -9.0
    for modo in GIROS:
        lin = matriz_orientacion(modo, 2, 2)[:2, :2]
        valor = (lin @ der)[0] + (lin @ abajo)[1]
        if valor > puntos:
            mejor, puntos = modo, valor
    return mejor if puntos > 1.4 else None          # QR muy inclinado: mejor no decidir


# ============================================================
# LECTOR DE QR (robusto a distancia, inclinación y desenfoque leve)
# ============================================================
class LectorQR:
    """Prueba varias preparaciones de la imagen y varios lectores hasta leer el QR.

    · Cada intento es una lista de pasos (preparación, lector), ordenada por lo que
      mejor funcionó en las pruebas (zbar sobre "relieve" o "nitidez" lee casi siempre
      que hay resolución; OpenCV sobre la imagen directa, después), con las dos
      polaridades intercaladas: un QR claro sobre fondo oscuro está INVERTIDO.
    · Primero va el paso con el que se leyó la última vez: en una misma placa casi
      siempre vuelve a servir, y así cada intento tarda poco.
    · 'presupuesto_ms' corta el intento; el siguiente sigue por donde se quedó, así en
      pocos frames se prueban todos los pasos sin que ninguno tarde mucho.
    · 'roi' (x0, y0, x1, y1) lee solo esa zona, ampliada (el QR de una placa ya registrada).
    · "relieve" (paso alto + CLAHE fuerte) es para un QR en relieve del mismo color que
      la placa: ahí no hay tinta, solo sombras.
    · "saturacion" usa la saturación del color en vez del gris: un QR pintado de blanco
      sobre una placa de color (morada, turquesa) se separa mucho mejor así, y además
      queda como módulos oscuros sobre fondo claro, que es lo que esperan los lectores.
    Límite físico: con menos de ~2 px por módulo no se puede leer. En la bandeja, a
    640×480, el QR mide unos 43 px (≈1,3 px por módulo, contando su margen): para leerlo
    ahí hace falta más resolución ("resolucion_qr" en config_exploracion.json)."""

    VARIANTES = ("directa", "clahe", "nitidez", "x1.6", "umbral", "relieve", "saturacion")
    PASOS = (("saturacion", "zbar"), ("relieve", "zbar"), ("nitidez", "zbar"), ("directa", "cv"),
             ("saturacion", "cv"), ("umbral", "zbar"),
             ("directa", "zbar"), ("nitidez", "aruco"), ("directa", "aruco"), ("clahe", "aruco"),
             ("x1.6", "zbar"), ("x1.6", "cv"), ("nitidez", "cv"), ("relieve", "cv"), ("clahe", "zbar"),
             ("umbral", "cv"), ("clahe", "cv"), ("relieve", "aruco"), ("x1.6", "aruco"), ("umbral", "aruco"),
             ("saturacion", "aruco"))
    LADO_ZONA = 360                         # una zona (roi) pequeña se amplía hasta este lado, en px

    def __init__(self):
        self.cv = cv2.QRCodeDetector()
        self.aruco = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_fuerte = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        self.invertido = False              # polaridad con la que se vio el QR la última vez
        self.orientacion = None             # giro de la imagen deducido del último QR leído
        self.esquinas = None                # esquinas del último QR leído (orden del código), o None
        self.variante = None                # paso (polaridad, preparación, lector) que leyó la última vez
        self._cursor = 0                    # por dónde seguir en el próximo intento
        self._sat = None                    # canal de saturación de la imagen (o zona) en curso
        try:
            from pyzbar import pyzbar
            self.zbar = pyzbar
        except Exception:
            self.zbar = None

    def _preparar(self, g, nombre):
        if nombre == "directa":
            return g, 1.0
        if nombre == "clahe":
            return self.clahe.apply(g), 1.0
        if nombre == "nitidez":
            return cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0, 0), 3), -0.8, 0), 1.0
        if nombre == "x1.6":
            return cv2.resize(g, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC), 1.6
        if nombre == "umbral":
            return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10), 1.0
        if nombre == "saturacion":
            return (self._sat if self._sat is not None else g), 1.0
        if nombre == "relieve":
            f = g.astype(np.float32)
            alto = f - cv2.GaussianBlur(f, (0, 0), 6)
            alto = cv2.normalize(alto, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return self.clahe_fuerte.apply(alto), 1.0
        return g, 1.0

    def _pasos(self):
        """Orden de este intento: el que leyó la última vez y luego los demás, desde donde
        se cortó el intento anterior."""
        lectores = {"zbar": self.zbar is not None, "cv": True, "aruco": self.aruco is not None}
        polaridades = (True, False) if self.invertido else (False, True)
        todos = [(inv, v, d) for v, d in self.PASOS if lectores[d] for inv in polaridades]
        pref = self.variante if self.variante in todos else None
        if pref:
            todos.remove(pref)
        k = self._cursor % len(todos)
        return pref, ([pref] if pref else []) + todos[k:] + todos[:k]

    @staticmethod
    def poligono_plausible(pts, w, h):
        """¿Puede ser un QR? Un QR visto por la cámara es un cuadrilátero convexo
        casi cuadrado (con algo de perspectiva), dentro de la imagen y de tamaño
        razonable. El detector de OpenCV a veces 'encuentra' cuadriláteros sobre
        las texturas de la gráfica (líneas de puntos, recuadros): esos se descartan."""
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

    def leer(self, frame, roi=None, presupuesto_ms=None):
        """Devuelve (texto|None, polígono|None). Si solo se localiza el QR pero
        no se decodifica, devuelve (None, polígono) para poder guiar al usuario;
        ese polígono solo se acepta si tiene forma de QR. Tras leer, self.esquinas
        tiene las 4 esquinas en el orden del código (si el lector las dio)."""
        t0 = time.perf_counter()
        H_img, W_img = frame.shape[:2]
        ox = oy = 0
        s0 = 1.0
        zona = frame
        if roi is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in roi)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W_img, x1), min(H_img, y1)
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                ox, oy = x0, y0
                zona = frame[y0:y1, x0:x1]
                lado = max(zona.shape[:2])
                if lado < self.LADO_ZONA:               # la zona del QR, ampliada
                    s0 = float(self.LADO_ZONA) / lado
                    zona = cv2.resize(zona, None, fx=s0, fy=s0, interpolation=cv2.INTER_CUBIC)
        gris = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY) if zona.ndim == 3 else zona
        # Saturación invertida: lo blanco (pintura) queda oscuro y la placa de color, clara.
        self._sat = (255 - cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)[..., 1]) if zona.ndim == 3 else None

        def a_frame(p, f):
            p = np.asarray(p, dtype=float).reshape(-1, 2) / f / s0
            return p + [ox, oy]

        def agotado():
            return presupuesto_ms is not None and 1000 * (time.perf_counter() - t0) > presupuesto_ms

        preparadas = {}
        poligono, poli_inv = None, self.invertido
        self.orientacion, self.esquinas = None, None
        pref, pasos = self._pasos()
        for paso in pasos:
            if agotado():
                break
            if paso != pref:
                self._cursor += 1                       # probado: la próxima vez se sigue con el siguiente
            inv, var, lector = paso
            if (inv, var) not in preparadas:
                if var == "saturacion":
                    if self._sat is None:
                        continue
                    preparadas[(inv, var)] = ((255 - self._sat) if inv else self._sat), 1.0
                else:
                    preparadas[(inv, var)] = self._preparar(255 - gris if inv else gris, var)
            img, f = preparadas[(inv, var)]
            if lector == "zbar":
                try:
                    for o in self.zbar.decode(img, symbols=[self.zbar.ZBarSymbol.QRCODE]):
                        txt = o.data.decode("utf-8", "replace").strip()
                        if not txt:
                            continue
                        self.invertido, self.variante = inv, paso
                        try:                    # zbar no da las esquinas en orden: se las pido a OpenCV,
                            poly = np.array([(p.x, p.y) for p in o.polygon], float)   # solo alrededor del código
                            m = 0.3 * float(np.ptp(poly, axis=0).max())
                            cx0, cy0 = np.maximum(poly.min(axis=0) - m, 0).astype(int)
                            cx1, cy1 = (poly.max(axis=0) + m).astype(int)
                            ok, pp = self.cv.detect(img[cy0:cy1, cx0:cx1])
                            if ok and pp is not None:
                                esq = a_frame(pp.reshape(-1, 2) + [cx0, cy0], f)
                                if self.poligono_plausible(esq, W_img, H_img):
                                    self.esquinas = esq
                                    self.orientacion = orientacion_desde_qr(esq)
                        except (cv2.error, ValueError):
                            pass
                        pts = a_frame([(p.x, p.y) for p in o.polygon], f)
                        return txt, (cv2.convexHull(self._arr(pts)) if len(pts) >= 3 else None)
                except Exception:
                    pass
                continue
            det = self.cv if lector == "cv" else self.aruco
            try:
                txt, pts, _ = det.detectAndDecode(img)
            except cv2.error:
                continue
            cand = None
            if pts is not None and len(pts):
                cand = a_frame(pts, f)
                if poligono is None and self.poligono_plausible(cand, W_img, H_img):
                    poligono, poli_inv = cand, inv
            if txt:
                self.invertido, self.variante = inv, paso
                ok = cand is not None and self.poligono_plausible(cand, W_img, H_img)
                if ok:
                    self.esquinas = cand
                    self.orientacion = orientacion_desde_qr(cand)
                return txt.strip(), (cand if ok else None)
        if poligono is None and not agotado():  # último recurso: solo localizar
            for inv in ((True, False) if self.invertido else (False, True)):
                try:
                    ok, pts = self.cv.detect(255 - gris if inv else gris)
                except cv2.error:
                    continue
                if ok and pts is not None:
                    cand = a_frame(pts, 1.0)
                    if self.poligono_plausible(cand, W_img, H_img):
                        poligono, poli_inv = cand, inv
                        break
        if poligono is not None:
            self.invertido = poli_inv             # la próxima vez se empieza por esa polaridad
        return None, poligono

    @staticmethod
    def _arr(pts):
        return np.array(pts, dtype="float32").reshape(-1, 1, 2)

# ============================================================
# ZOOM AL QR (modo F3): leer solo la esquina de la placa donde va el código
# ============================================================
# En la bandeja, el QR ocupa menos del 2 % de la imagen. Leer la imagen entera es lento
# (con más resolución, mucho más) y el lector se distrae con las texturas y el braille.
# Aquí se busca primero la placa (con su forma: cuadrada y con una esquina recortada) y
# se lee solo la zona de su esquina superior izquierda, ampliada: es el "acercamiento".
# El detalle que se gana depende de la resolución de la cámara: ampliar una imagen de
# 640×480 no inventa detalle (el QR queda en ~1,7 px por módulo y hace falta ~2,5); por
# eso, además, MotorVision sube la resolución de la cámara solo mientras busca el QR,
# eligiendo una en la que la placa entera siga dentro de la imagen (ver _sondear_modo_qr).
# Solo 4:3, como la normal (640×480): los modos panorámicos (1280×720, 1920×1080) de casi
# todas las webcams RECORTAN el sensor (se pierde la parte de arriba de la bandeja, donde
# está el QR). Aunque la configuración traiga otros, se descartan (ver misma_proporcion).
RESOLUCIONES_QR = ((1280, 960), (1600, 1200), (2048, 1536), (2592, 1944))
VERSION_MODO_QR = 4        # lo guardado en "modos_qr" con una versión menor se vuelve a probar


def misma_proporcion(a, b, tol=0.03):
    """¿Las resoluciones a y b tienen la misma proporción (ancho/alto)? Si no, la cámara
    recorta o estira la imagen al cambiar de una a otra."""
    try:
        ra, rb = float(a[0]) / float(a[1]), float(b[0]) / float(b[1])
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        return False
    return abs(ra - rb) <= tol * rb


class PlantillaPlaca:
    """Lo que la autodetección necesita saber de la placa ANTES de tener el CSV (en F3 el CSV
    llega justamente por el QR): tamaño, esquina recortada y dónde va el QR. Por defecto, las
    placas del proyecto: 220 × 220 mm, esquina superior derecha recortada 10 mm, QR arriba a
    la izquierda. Se cambia con "plantilla_placa" y "qr_zona_placa" en la configuración."""

    def __init__(self, cfg):
        p = cfg.get("plantilla_placa") or {}
        mm = p.get("mm") or [220.0, 220.0]
        self.ancho_ref, self.alto_ref = float(mm[0]), float(mm[1])
        self.estirar = True
        self.esquina_recortada = p.get("esquina_recortada", "superior derecha") or ""
        self.recorte_mm = float(p.get("recorte_mm", 10.0) or 0.0)
        self.elementos = []


class ZoomQR:
    """Lee el QR en la esquina de la placa. Si no ve la placa (p. ej. está en la mano),
    lee la imagen completa, como antes.

    leer(frame, orientacion, presupuesto_ms) -> (texto|None, polígono|None), y deja en
    self.ultima lo necesario para la vista ampliada y la guía por voz."""

    def __init__(self, cfg, lector=None):
        self.cfg = cfg
        self.lector = lector or LectorQR()
        self.plantilla = PlantillaPlaca(cfg)
        z = cfg.get("qr_zona_placa") or [0.02, 0.02, 0.21, 0.21]
        self.zona = [float(v) for v in z]             # fracción de la placa: x0, y0, x1, y1
        self.modulo_mm = float(cfg.get("qr_modulo_mm", 1.03) or 1.03)
        self.cada_s = 1.0                             # cada cuánto se vuelve a buscar la placa
        self.loc, self._t_loc = None, 0.0
        self._n, self._k_ok = 0, 0
        self.ultima = {"placa": False}

    def localizar(self, frame, orientacion="ninguno"):
        """La placa en la imagen tal cual llega de la cámara. Devuelve un dict o None."""
        h, w = frame.shape[:2]
        f = min(1.0, 800.0 / max(w, h))              # a 800 px basta y es rápido
        peq = frame if f == 1.0 else cv2.resize(frame, (int(round(w * f)), int(round(h * f))),
                                                interpolation=cv2.INTER_AREA)
        det = detectar_placa(peq, self.plantilla)
        if det is None:
            return None
        esq = np.asarray(det.esquinas, float) / f
        if det.esquina_vista is None:
            # Sin ver la esquina recortada no se sabe cuál es la de arriba a la izquierda de la
            # placa: se supone la que queda arriba a la izquierda en la imagen ya orientada.
            vis = _aplicar_h(matriz_orientacion(orientacion, w, h), esq)
            esq = np.roll(esq, -int(np.argmin(vis[:, 0] + vis[:, 1])), axis=0)
        reg = Registro.desde_esquinas(esq, self.plantilla.ancho_ref, self.plantilla.alto_ref, True, "plantilla")
        return {"registro": reg, "forma": (h, w), "segura": det.esquina_vista is not None,
                "px_por_mm": reg.px_por_mm(), "confianza": det.confianza}

    def zonas(self, loc):
        """La zona del QR en cada esquina de la placa (la de arriba a la izquierda primero):
        [(polígono 4×2 px, caja (x0, y0, x1, y1), dentro de la imagen)]."""
        W, H = self.plantilla.ancho_ref, self.plantilla.alto_ref
        fx0, fy0, fx1, fy1 = self.zona
        h, w = loc["forma"]
        res = []
        for espejo_x, espejo_y in ((False, False), (True, False), (True, True), (False, True)):
            xs = [W - fx1 * W, W - fx0 * W] if espejo_x else [fx0 * W, fx1 * W]
            ys = [H - fy1 * H, H - fy0 * H] if espejo_y else [fy0 * H, fy1 * H]
            pts = np.array([[xs[0], ys[0]], [xs[1], ys[0]], [xs[1], ys[1]], [xs[0], ys[1]]])
            poli = loc["registro"].placa_a_imagen_arr(pts)
            x0, y0 = poli.min(axis=0)
            x1, y1 = poli.max(axis=0)
            m = 0.02 * max(w, h)
            dentro = bool(x0 >= -m and y0 >= -m and x1 <= w - 1 + m and y1 <= h - 1 + m)
            res.append((poli, (max(0.0, x0), max(0.0, y0), min(w - 1.0, x1), min(h - 1.0, y1)), dentro))
        return res

    def leer(self, frame, orientacion="ninguno", presupuesto_ms=None):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        ahora = time.time()
        if (self.loc is None and ahora - self._t_loc > 0.5) or ahora - self._t_loc > self.cada_s \
                or (self.loc is not None and self.loc["forma"] != (h, w)):
            self._t_loc = ahora
            try:
                self.loc = self.localizar(frame, orientacion)
            except Exception:
                traceback.print_exc()
                self.loc = None
        resto = None if presupuesto_ms is None else max(40.0, presupuesto_ms - 1000 * (time.perf_counter() - t0))
        if self.loc is None:                         # sin placa a la vista: la imagen entera
            self.ultima = {"placa": False}
            return self.lector.leer(frame, presupuesto_ms=resto)
        zonas = self.zonas(self.loc)
        # Con la esquina recortada a la vista se sabe dónde está el QR: se lee siempre esa esquina.
        # Si no se vio, se suponen la de arriba a la izquierda y, de vez en cuando, las otras tres.
        # La vista ampliada se queda siempre en la esquina principal (no salta entre esquinas).
        self._n += 1
        k = self._k_ok
        if not self.loc["segura"] and self._n % 4 == 0:
            k = (self._k_ok + self._n // 4) % 4
        poli, caja, dentro = zonas[k]
        texto, pol = self.lector.leer(frame, roi=caja, presupuesto_ms=resto)
        if texto:
            self._k_ok = k
        pv, cv_, dv = zonas[self._k_ok]
        self.ultima = {"placa": True, "zona": pv, "caja": cv_, "esquina": self._k_ok, "dentro": dv,
                       "px_por_mm": self.loc["px_por_mm"], "segura": self.loc["segura"],
                       "px_por_modulo": self.loc["px_por_mm"] * self.modulo_mm}
        return texto, pol

    def nitidez(self, frame, loc, mm_px=8.0):
        """Detalle real en la esquina del QR, medido igual para cualquier resolución: la zona se
        lleva a una cuadrícula fija de 'mm_px' px por mm de la placa y se mide su energía de
        bordes. Una resolución "inflada" (la cámara amplía por dentro una imagen menor) da el
        mismo detalle que la menor, aunque tenga más píxeles."""
        W, H = self.plantilla.ancho_ref, self.plantilla.alto_ref
        fx0, fy0, fx1, fy1 = self.zona
        x0, y0, x1, y1 = fx0 * W, fy0 * H, fx1 * W, fy1 * H
        ancho, alto = int((x1 - x0) * mm_px), int((y1 - y0) * mm_px)
        # cuadrícula (px de salida) -> mm -> px de la imagen
        a_mm = np.array([[1.0 / mm_px, 0, x0], [0, 1.0 / mm_px, y0], [0, 0, 1.0]])
        M = loc["registro"].H @ a_mm
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        parche = cv2.warpPerspective(gris, M, (ancho, alto), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        parche = cv2.GaussianBlur(parche.astype(np.float32), (0, 0), 0.7)
        return float(np.mean(np.abs(cv2.Laplacian(parche, cv2.CV_32F))))


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


class _Punto:
    """Un punto de la mano normalizado a la IMAGEN COMPLETA (como los de MediaPipe)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class DetectorMano:
    """Devuelve, para cada mano vista, su lista de 21 puntos (con .x e .y entre 0 y 1,
    relativos a la imagen completa aunque se haya recortado).
    Punto 8 = punta del índice · punto 12 = punta del dedo medio (igual en ambas APIs).

    detectar(frame, roi=None): con roi (x0, y0, x1, y1) solo se mira esa zona (la placa
    con margen): la mano ocupa más píxeles del recorte y el modelo la sitúa mejor.
    "mano_lado_max_px" (configuración) reduce la imagen antes de pasarla al modelo."""

    def __init__(self, cfg, avisar=None):
        import mediapipe as mp                     # puede lanzar ImportError
        self.mp = mp
        self.version = getattr(mp, "__version__", "?")
        conf = float(cfg.get("confianza_mano", 0.5))
        self.lado_max = int(cfg.get("mano_lado_max_px") or 0)
        vision, BaseOptions = _api_tasks(mp)
        self._ultimo_ts = -1
        self._t0 = time.monotonic()
        self.ms_ultimo = 0.0                       # lo que tardó el modelo en el último frame

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
            except Exception:
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
              f"{'Tasks (HandLandmarker)' if self.api == 'tasks' else 'clásica (mp.solutions)'}"
              + (f", imagen reducida a {self.lado_max} px." if self.lado_max else "."))

    def _marca_tiempo(self):
        """El modo VIDEO exige milisegundos estrictamente crecientes."""
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._ultimo_ts:
            ts = self._ultimo_ts + 1
        self._ultimo_ts = ts
        return ts

    def detectar(self, frame_bgr, roi=None):
        t = time.perf_counter()
        H, W = frame_bgr.shape[:2]
        x0, y0, x1, y1 = 0, 0, W, H
        if roi is not None:
            x0, y0, x1, y1 = (int(v) for v in roi)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 32 or y1 - y0 < 32:
                x0, y0, x1, y1 = 0, 0, W, H
        sub = frame_bgr if (x0, y0, x1, y1) == (0, 0, W, H) else frame_bgr[y0:y1, x0:x1]
        if self.lado_max and max(sub.shape[:2]) > self.lado_max:
            f = self.lado_max / max(sub.shape[:2])
            sub = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
        if self.api == "tasks":
            imagen = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self.motor.detect_for_video(imagen, self._marca_tiempo())
            manos = list(res.hand_landmarks or [])
        else:
            res = self.motor.process(rgb)
            manos = [hl.landmark for hl in (res.multi_hand_landmarks or [])]
        self.ms_ultimo = 1000 * (time.perf_counter() - t)
        if (x0, y0, x1, y1) == (0, 0, W, H):
            return manos
        cw, ch = x1 - x0, y1 - y0
        return [[_Punto((x0 + p.x * cw) / W, (y0 + p.y * ch) / H, getattr(p, "z", 0.0)) for p in m]
                for m in manos]

    def cerrar(self):
        try:
            self.motor.close()
        except Exception:
            pass


def roi_mano(registro, ancho, alto, margen=0.4):
    """Zona de la imagen donde buscar la mano: la placa con un margen (fracción de su
    lado), para que quepan también la palma y la muñeca. None si es casi toda la imagen."""
    if registro is None:
        return None
    x, y, w, h = registro.caja_px()
    m = margen * max(w, h)
    x0, y0 = max(0, int(x - m)), max(0, int(y - m))
    x1, y1 = min(ancho, int(x + w + m)), min(alto, int(y + h + m))
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    if (x1 - x0) * (y1 - y0) > 0.85 * ancho * alto:
        return None
    return x0, y0, x1, y1


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
# SUPERPOSICIÓN (lo que se dibuja sobre el video)
# ============================================================
# Colores BGR para las series (el cian queda para los ejes y el rojo para los puntos).
COLORES_SERIE = [(0, 200, 0), (0, 140, 255), (255, 0, 255), (255, 128, 0), (0, 220, 220), (180, 105, 255)]
COLOR_EJE, COLOR_PUNTO, COLOR_TEXTO = (255, 255, 0), (0, 0, 255), (170, 170, 170)
COLOR_PLACA = (0, 255, 255)


def contorno_placa_mm(datos):
    """Contorno de la placa en mm, con la esquina recortada (para dibujarlo)."""
    W, H = datos.ancho_ref, datos.alto_ref
    esq = [[0, 0], [W, 0], [W, H], [0, H]]
    nombre = getattr(datos, "esquina_recortada", "")
    r = float(getattr(datos, "recorte_mm", 0.0) or 0.0)
    if not getattr(datos, "estirar", True) or nombre not in NOMBRES_ESQUINAS or r <= 0:
        return np.array(esq, float)
    i = NOMBRES_ESQUINAS.index(nombre)
    p, a, b = np.array(esq[i], float), np.array(esq[i - 1], float), np.array(esq[(i + 1) % 4], float)
    pa, pb = p + (a - p) / np.hypot(*(a - p)) * r, p + (b - p) / np.hypot(*(b - p)) * r
    return np.array(esq[:i] + [pa.tolist(), pb.tolist()] + esq[i + 1:], float)


class Superposicion:
    """Dibuja la gráfica sobre el video con el registro. Las coordenadas en px se
    calculan una vez por cambio de registro (no en cada frame)."""

    def __init__(self):
        self.registro = self.datos = None
        self.geo = {}            # id → ("punto", (x, y)) | ("segmento", p1, p2) | ("rect", polígono)
        self.elementos = []
        self.por_id = {}
        self.colores = {}
        self.placa = None

    def configurar(self, datos, registro):
        self.datos, self.registro = datos, registro
        self.geo, self.elementos, self.placa = {}, [], None
        self.por_id = {e["id"]: e for e in datos.elementos} if datos is not None else {}
        if datos is None or registro is None:
            return
        series = list(datos.series) or sorted({e["serie"] for e in datos.elementos})
        self.colores = {s: COLORES_SERIE[i % len(COLORES_SERIE)] for i, s in enumerate(series)}
        for e in datos.elementos:
            if e["forma"] == "punto":
                g = ("punto", registro.placa_a_imagen(e["x1"], e["y1"]))
            elif e["forma"] == "rect":
                x0, x1 = sorted((e["x1"], e["x2"]))
                y0, y1 = sorted((e["y1"], e["y2"]))
                p = registro.placa_a_imagen_arr([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                g = ("rect", np.round(p).astype(np.int32).reshape(-1, 1, 2))
            else:
                g = ("segmento", registro.placa_a_imagen(e["x1"], e["y1"]), registro.placa_a_imagen(e["x2"], e["y2"]))
            self.geo[e["id"]] = g
            self.elementos.append(e)
        self.placa = np.round(registro.placa_a_imagen_arr(contorno_placa_mm(datos))).astype(np.int32).reshape(-1, 1, 2)

    def _trazar(self, frame, g, color, grosor):
        if g[0] == "punto":
            cv2.circle(frame, g[1], 5 if grosor <= 2 else 10, color, -1 if grosor <= 2 else 2)
        elif g[0] == "rect":
            cv2.polylines(frame, [g[1]], True, color, grosor)
        else:
            cv2.line(frame, g[1], g[2], color, grosor)

    def dibujar(self, frame, elemento_id=None, dedos=None, pinza=False, texto=None):
        """dedos: (índice_px, medio_px) o None. texto: lo que se muestra arriba (el 'corto')."""
        h, _w = frame.shape[:2]
        if self.placa is not None:
            cv2.polylines(frame, [self.placa], True, COLOR_PLACA, 1)
        for e in self.elementos:                                  # textos y leyendas: gris fino
            if e["tipo"] in ("punto", "curva", "tramo", "eje"):
                continue
            g = self.geo[e["id"]]
            if g[0] == "punto":
                cv2.circle(frame, g[1], 3, COLOR_TEXTO, -1)
            else:
                self._trazar(frame, g, COLOR_TEXTO, 1)
        for e in self.elementos:                                  # ejes en cian
            if e["tipo"] == "eje":
                g = self.geo[e["id"]]
                self._trazar(frame, g, COLOR_EJE, 2)
                if e["corto"] in ("Eje horizontal", "Eje vertical") and g[0] == "segmento":
                    cv2.putText(frame, "X" if e["corto"] == "Eje horizontal" else "Y",
                                (g[2][0] + 6, g[2][1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EJE, 2)
        for e in self.elementos:                                  # curvas: un color por serie
            if e["tipo"] in ("curva", "tramo"):
                self._trazar(frame, self.geo[e["id"]], self.colores.get(e["serie"], COLORES_SERIE[0]), 2)
        for e in self.elementos:                                  # puntos en rojo
            if e["tipo"] == "punto":
                cv2.circle(frame, self.geo[e["id"]][1], 5, COLOR_PUNTO, -1)
        g = self.geo.get(elemento_id) if elemento_id is not None else None
        if g is not None:                                         # elemento señalado: resaltado
            self._trazar(frame, g, (255, 255, 255), 4 if g[0] == "segmento" else 2)
        if dedos is not None:                                     # los dedos encima de todo
            (ix, iy), (mx, my) = ((int(round(a)), int(round(b))) for a, b in dedos)
            cv2.circle(frame, (mx, my), 8, (255, 0, 0), -1)
            cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
            cv2.line(frame, (mx, my), (ix, iy), (255, 0, 0), 2)
        cv2.putText(frame, "AUDIO ACTIVO" if pinza else "Junta medio + indice",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if pinza else (255, 255, 255), 2)
        if texto:
            corto = _ascii(texto)
            if len(corto) > 55:
                corto = corto[:52] + "..."
            cv2.putText(frame, corto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame


# ============================================================
# CAPTURA (solo el último frame)
# ============================================================
def fijar_formato(cap, ancho, alto, mjpg=True):
    """Pide a la cámara un tamaño y, si 'mjpg', el formato comprimido MJPG.

    Por qué importa: sin comprimir (YUY2) una imagen de 1920×1080 pesa 4 MB y por USB 2.0
    no caben más de unas 5 por segundo; la cámara entrega entonces imágenes lentas y a
    veces mezcladas a medias (las rayas horizontales de colores). Con MJPG caben a 30 fps.
    Algunos drivers de DirectShow vuelven a YUY2 al cambiar el tamaño, así que el formato
    se pide antes y, si se perdió, otra vez después. Devuelve (ancho, alto, formato)."""
    mj = cv2.VideoWriter_fourcc(*"MJPG")
    try:
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, mj)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        if mjpg and fourcc_texto(cap) != "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, mj)
    except Exception as e:
        print("Aviso: no se pudo cambiar el formato de la cámara:", e)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fourcc_texto(cap)
    except Exception:
        return int(ancho), int(alto), "?"


class Captura(threading.Thread):
    """Lee la cámara sin parar y conserva SOLO el último frame. Así el
    procesamiento (mano, QR) nunca trabaja con imágenes viejas acumuladas en
    el búfer del driver: es la causa principal del retraso al mover la mano.
    pedir_resolucion() cambia la resolución desde este mismo hilo (entre dos
    lecturas), para no tocar la cámara mientras otro hilo lee."""

    def __init__(self, cap, mjpg=True):
        super().__init__(daemon=True)
        self.cap, self.frame, self.n, self.ok = cap, None, 0, True
        self.cond = threading.Condition()
        self._fin = threading.Event()
        self._resolucion = None
        self.mjpg = mjpg
        self.cambios = 0                  # cuántos cambios de resolución se aplicaron
        self.n_cambio = 0                 # número del último frame anterior al último cambio
        self.formato = fourcc_texto(cap)

    def pedir_resolucion(self, ancho, alto):
        self._resolucion = (int(ancho), int(alto))

    def run(self):
        fallos = 0
        while not self._fin.is_set():
            if self._resolucion is not None:
                (rw, rh), self._resolucion = self._resolucion, None
                aw, ah, self.formato = fijar_formato(self.cap, rw, rh, self.mjpg)
                print("Resolución de la cámara: %dx%d (formato %s)" % (aw, ah, self.formato))
                with self.cond:
                    self.cambios, self.n_cambio = self.cambios + 1, self.n
            ok, f = self.cap.read()
            ok = ok and f is not None and f.size > 0
            fallos = 0 if ok else fallos + 1
            with self.cond:
                if ok:
                    self.frame, self.n = f, self.n + 1
                elif fallos > 30:                   # un fallo suelto (habitual en algunos drivers) no corta
                    self.ok = False
                self.cond.notify_all()
            if not self.ok:
                break
            if not ok:
                time.sleep(0.03)

    def esperar(self, n_visto, timeout=1.0):
        """Bloquea hasta que haya un frame más nuevo que 'n_visto'."""
        with self.cond:
            self.cond.wait_for(lambda: self.n != n_visto or not self.ok, timeout)
            if self.n == n_visto:
                return None, n_visto
            return self.frame, self.n

    def parar(self):
        self._fin.set()


def fourcc_texto(cap):
    try:
        v = int(cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00") or "?"
    except Exception:
        return "?"


class MedidorVision:
    """fps de la cámara y del procesamiento, y ms del modelo de la mano; lo imprime
    en la consola cada 'cada' segundos (clave "depurar_vision")."""

    def __init__(self, cada=10.0):
        self.cada, self.t0, self.n, self.n_cam0, self.ms = cada, time.time(), 0, None, []
        self.ultimo = {}

    def contar(self, n_camara, ms_mano=None):
        if self.n_cam0 is None:
            self.n_cam0 = n_camara
        self.n += 1
        if ms_mano is not None:
            self.ms.append(ms_mano)
        dt = time.time() - self.t0
        if dt >= self.cada:
            self.ultimo = {"fps_camara": (n_camara - self.n_cam0) / dt, "fps_proceso": self.n / dt,
                           "ms_mano": float(np.median(self.ms)) if self.ms else None}
            self.t0, self.n, self.n_cam0, self.ms = time.time(), 0, n_camara, []
            return self.ultimo
        return None


# ============================================================
# HILO DE VISIÓN (cámara + QR + registro + exploración)
# ============================================================
class MotorVision(threading.Thread):
    """Lee la cámara en un hilo propio y le pasa a la GUI:
         · self.cola_frames  → (datos_imagen, (ancho, alto), escala_vista, formato)
         · cola de eventos   → ("estado"|"error_camara"|"qr"|"elemento"|"placa"|"orientacion", ...)
    Modos: "espera"/"vista" (solo muestra), "qr" (busca QR), "explorar".

    En "explorar", por cada frame: mano → MuestraMano (mm, con el registro) →
    interaccion.actualizar(muestra) → Seleccion → superposición. 'crear_interaccion'
    es una función (detector, avisar) → objeto de la Persona B con configurar(datos,
    registro) y actualizar(muestra)."""

    def __init__(self, cola_eventos, voz, cfg, crear_interaccion=None):
        super().__init__(daemon=True)
        self.cola, self.voz, self.cfg = cola_eventos, voz, cfg
        self.cola_frames = queue.Queue(maxsize=1)
        self._parar = threading.Event()
        self.modo = "vista"
        self.datos = None
        self.hoja = tuple(cfg["hoja"])
        self.esquinas = esquinas_validas(cfg.get("esquinas_placa_px"))
        self.origen_registro = cfg.get("registro_origen") or ("esquinas" if self.esquinas is not None else "rect")
        self.error_registro = cfg.get("registro_error_mm")
        self.registro = None
        self.vista = VISTA_DEFECTO
        self._reconfigurar = False
        self._autodetectar = False
        self._comprobar_placa = True               # al empezar a explorar se comprueba dónde está la placa
        self._esquinas_pedidas = None
        self._clics = []
        self.indice = 0
        self.clave_camara = "cámara 0"            # nombre de la cámara (clave para guardar su orientación)
        self.orientacion = "ninguno"
        self._orient_pedida = None
        self._dim_raw = None                      # (ancho, alto) de la imagen tal cual llega de la cámara
        self._lector_ajuste = None                # lector QR para la tecla A durante la exploración
        self._crear_interaccion = crear_interaccion
        self.detector = None
        self.interaccion = None
        self.superposicion = Superposicion()
        self.t_mano = time.time()                 # última vez que se vio la mano
        self.lector = None
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
        # Seguimiento de la placa (en segundo plano, solo cuando no hay mano encima).
        self._hilo_placa = None
        self._placa_nueva = threading.Event()
        self._placa_frame, self._placa_res = None, None
        self._t_placa = 0.0
        self._candidata = None
        self._metodo_placa = None
        self._t_aviso_borde = 0.0
        self.captura = None
        self._res_normal = None                   # (ancho, alto) sin orientar con que se abrió la cámara
        self._res_pedida, self._t_res, self._intentos_res = None, 0.0, {}
        self._n_explorar = 0                      # frames desde que empezó la exploración
        self.medidor = MedidorVision()
        # Lectura del QR con zoom (modo F3): resolución elegida para el QR y estado de la guía.
        self._mjpg = bool(cfg.get("mjpg", True))
        self._modo_qr = None                      # (ancho, alto) para leer el QR, o None = la normal
        self._sondeado = False
        self._ignorar_fija = False                # la fija de la configuración resultó recortar
        self.zoom = None
        self._t_zoom, self._zoom_avisado = None, False

    # ---- control desde la GUI ----
    def set_modo(self, modo):
        self.modo = modo
        self._t_visto = time.time()
        if modo == "qr":
            self._t_zoom, self._zoom_avisado = None, False
        if modo == "explorar":
            self._comprobar_placa, self._n_explorar = True, 0

    def set_datos(self, datos):
        self.datos, self._reconfigurar = datos, True

    def set_hoja(self, hoja):
        """Compatibilidad: calibrar con un rectángulo (borra las esquinas)."""
        self.hoja, self.esquinas, self.origen_registro = tuple(int(v) for v in hoja), None, "rect"
        self._reconfigurar = True

    def set_esquinas(self, esquinas, origen="clics"):
        """Esquinas de la placa (sup-izq, sup-der, inf-der, inf-izq) en la imagen orientada."""
        self._esquinas_pedidas = (np.asarray(esquinas, float).reshape(4, 2), origen)

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

    def _probar_camara(self, idx, backend, mjpg):
        """Abre la cámara con un backend y comprueba que de verdad entrega imágenes:
        en algunos equipos Windows DirectShow abre pero da frames vacíos, o el
        formato MJPG no está soportado. Devuelve el VideoCapture o None.
        "exposicion" (configuración): con poca luz muchas webcams alargan la exposición
        y bajan solas a 15 fps; un valor fijo (p. ej. -6 en DirectShow ≈ 1/64 s) lo evita."""
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception:
            return None
        if not cap.isOpened():
            cap.release()
            return None
        ancho, alto = self.cfg.get("resolucion", [640, 480])
        fijar_formato(cap, ancho, alto, mjpg)
        try:
            cap.set(cv2.CAP_PROP_FPS, float(self.cfg.get("fps_camara", 30) or 30))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # menos cola = menos retraso
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)      # el QR llega a distintas alturas
            exp = self.cfg.get("exposicion")
            if exp is not None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual (DirectShow / V4L2)
                cap.set(cv2.CAP_PROP_EXPOSURE, float(exp))
        except Exception:
            pass
        for _ in range(15):                         # algunas cámaras tardan en dar el primer frame
            ok, f = cap.read()
            if ok and f is not None and f.size:
                return cap
            time.sleep(0.05)
        cap.release()
        return None

    def _abrir(self):
        nombres = listar_camaras()
        buscada = str(self.cfg.get("nombre_camara") or "").strip()
        idx, encontrada = self.cfg.get("indice_camara"), False
        if idx is None:
            idx = next((i for i, n in enumerate(nombres) if buscada and buscada.lower() in n.lower()), None)
            encontrada = idx is not None
            if idx is None:
                idx = 0
        if sys.platform.startswith("win"):
            backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")]
        else:
            backends = [(cv2.CAP_ANY, "predeterminado")]
        formatos = [True, False] if self.cfg.get("mjpg", True) else [False]
        cap, idx_buscado = None, idx
        nombre_b, mjpg = "", False
        for i in ([idx, 0] if idx != 0 else [0]):
            for backend, nombre_b in backends:
                for mjpg in formatos:
                    cap = self._probar_camara(i, backend, mjpg)
                    if cap is not None:
                        break
                if cap is not None:
                    break
            if cap is not None:
                idx = i
                break
        if cap is None:
            return None
        if idx != idx_buscado:                      # la buscada no abrió: se usa la 0
            encontrada = False
        self._mjpg = mjpg
        self.indice = idx
        self.clave_camara = nombres[idx] if idx < len(nombres) else f"cámara {idx}"
        if (buscada and not encontrada and self.cfg.get("indice_camara") is None) or idx != idx_buscado:
            self._emitir("estado", f"No encontré la cámara {buscada or idx_buscado}; uso {self.clave_camara}.", True)
        self.orientacion = self._orientacion_inicial(encontrada)
        print("Cámara abierta: %s (índice %d, %s%s): %dx%d a %.0f fps (formato %s, exposición %s) · orientación: %s" % (
            self.clave_camara, idx, nombre_b, ", MJPG pedido" if mjpg else "",
            cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FPS), fourcc_texto(cap),
            "automática" if self.cfg.get("exposicion") is None else self.cfg.get("exposicion"), self.orientacion))
        return cap

    def _orientacion_inicial(self, camara_configurada):
        """1) la guardada para esta cámara; 2) la clave antigua 'voltear' si se fijó a mano;
        3) la cámara habitual del proyecto (montada del revés) → 180; 4) sin girar.
        Ya no se decide por el número de la cámara: en otro equipo la misma cámara
        puede tener otro número y salía en espejo o del revés."""
        guardadas = self.cfg.get("orientaciones") or {}
        if guardadas.get(self.clave_camara) in ORIENTACIONES:
            return guardadas[self.clave_camara]
        antigua = self.cfg.get("voltear", "auto")
        if antigua in ORIENTACIONES:
            return antigua
        return "180" if camara_configurada else "ninguno"

    def set_orientacion(self, modo):
        """Desde la GUI (tecla O): se aplica en el hilo de la cámara."""
        if modo in ORIENTACIONES:
            self._orient_pedida = (modo, "manual")

    def _fijar_orientacion(self, modo, origen):
        if modo not in ORIENTACIONES:
            return
        cambia = modo != self.orientacion
        dims = self._res_normal or self._dim_raw      # el registro está en px de la resolución normal
        if cambia and dims is not None:               # la calibración sigue en el mismo sitio de la placa
            w, h = dims
            self.hoja = convertir_rect(self.hoja, self.orientacion, modo, w, h)
            if self.esquinas is not None:
                M = cambio_orientacion(self.orientacion, modo, w, h)
                # Las esquinas siguen siendo las mismas de la placa (no se reordenan).
                self.esquinas = _aplicar_h(M, self.esquinas)
            self._clics = []
        self.orientacion = modo
        if cambia:
            self._reconfigurar = True
        if cambia or origen == "manual":
            self._emitir("orientacion", modo, origen, self.clave_camara, tuple(self.hoja),
                         cambia, None if self.esquinas is None else self.esquinas.round(2).tolist())

    def _voltear(self, frame):
        return aplicar_orientacion(frame, self.orientacion)

    def _a_vista(self, pts):
        """Puntos de la imagen original → imagen orientada (para dibujar el QR)."""
        w, h = self._dim_raw
        return _aplicar_h(matriz_orientacion(self.orientacion, w, h), pts)

    def _hablar_guia(self, texto, cada=4.0):
        ahora = time.time()
        if ahora - self._t_aviso >= cada and not self.voz.hablando.is_set():
            self._t_aviso = ahora
            self.voz.decir(texto, esencial=True)
            return True
        return False

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

    # ---- resolución distinta para leer el QR ("resolucion_qr": "auto" o [ancho, alto]) ----
    def _ajustar_resolucion(self, frame):
        """En el modo QR pide a la cámara la resolución elegida para el QR (si hay una);
        en los demás, la de siempre (la que dio al abrirse: el registro está en esos px).
        Devuelve False mientras en "explorar" lleguen frames de otro tamaño. Si la cámara
        no acepta el cambio tras 3 intentos, se deja como está."""
        tiene = (frame.shape[1], frame.shape[0])
        if self._res_normal is None:
            self._res_normal = tiene
        if self.captura is None:
            return True
        if self.modo == "qr":
            if (self._modo_qr is not None and tiene == self._modo_qr
                    and not misma_proporcion(tiene, self._res_normal)):
                # La cámara entregó un formato que recorta la imagen (p. ej. pidió 1280×960 y
                # dio 1280×720): la esquina del QR queda fuera. Se vuelve a la normal.
                print("QR: la cámara entregó %dx%d, que recorta la imagen respecto a %dx%d; "
                      "vuelvo a la normal." % (tiene + tuple(self._res_normal)))
                self._modo_qr = None
            elif (self._modo_qr is not None and tiene != self._modo_qr
                    and self._res_pedida == self._modo_qr and time.time() - self._t_res > 1.0
                    and not misma_proporcion(tiene, self._res_normal)):
                print("QR: pedí %dx%d y la cámara dio %dx%d (recorta la imagen); vuelvo a la normal."
                      % (tuple(self._modo_qr) + tiene))
                self._modo_qr = None
                if not self._ignorar_fija:      # una vez: probar sola las demás resoluciones
                    self._ignorar_fija, self._sondeado = True, False
            quiere = self._modo_qr or self._res_normal
        else:
            quiere = self._res_normal
        if tiene == quiere:
            return True
        n = self._intentos_res.get(quiere, 0)
        if n < 3 and (self._res_pedida != quiere or time.time() - self._t_res > 3.0):
            self._intentos_res[quiere] = n + 1
            self._res_pedida, self._t_res = quiere, time.time()
            self.captura.pedir_resolucion(*quiere)
        if self.modo == "explorar" and n >= 3:
            print("Aviso: la cámara no volvió a su resolución normal; el registro puede no coincidir.")
        return self.modo != "explorar" or n >= 3


    def _config_resolucion_qr(self):
        """"auto" (o vacío): elegir sola · [ancho, alto]: esa · "normal": no cambiarla."""
        v = self.cfg.get("resolucion_qr", "auto")
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return tuple(int(x) for x in v)
        if isinstance(v, str) and v.strip().lower() in ("normal", "no", "ninguna"):
            return "normal"
        return "auto"

    def _zoom_qr(self):
        if self.zoom is None:
            self.zoom = ZoomQR(self.cfg, self.lector)
        return self.zoom

    def _frames_nuevos(self, cambios_antes, segundos=0.8, saltar=3, timeout=6.0):
        """[(t, frame)] durante 'segundos', tras aplicarse un cambio de resolución
        (cambios_antes = captura.cambios antes de pedirlo; None = sin cambio)."""
        cap = self.captura
        t0 = time.time()
        if cambios_antes is not None:
            while cap.cambios == cambios_antes and time.time() - t0 < timeout and not self._parar.is_set():
                time.sleep(0.02)
            if cap.cambios == cambios_antes:
                return []
            n = cap.n_cambio
        else:
            n = cap.n
        res, vistos, t1 = [], 0, time.time()
        while not self._parar.is_set() and time.time() - t1 < segundos + 4.0:
            f, n = cap.esperar(n, 1.0)
            if f is None:
                if not cap.ok:
                    break
                continue
            vistos += 1
            self._enviar_frame(self._voltear(f))       # que el video no se congele mientras tanto
            if vistos <= saltar:                       # los primeros, con exposición y enfoque aún moviéndose
                continue
            res.append((time.time(), f))
            if res[-1][0] - res[0][0] >= segundos:
                break
        return res

    def _sondear_modo_qr(self):
        """Elige con qué resolución leer el QR (una sola vez por cámara; queda guardada en
        "modos_qr"). Prueba las de "resoluciones_qr_candidatas" y se queda con la que da más
        píxeles por milímetro en la esquina del QR, siempre que la placa entera y esa
        esquina sigan dentro de la imagen y lleguen al menos 3 imágenes por segundo.

        Por qué no basta con pedir la más alta: muchas webcams, en 1920×1080, recortan el
        sensor (la imagen muestra solo una parte de la placa y el QR queda fuera) y, si no
        aceptan MJPG, envían a pocos fps y con rayas."""
        self._sondeado = True
        modo = self._config_resolucion_qr()
        if self._ignorar_fija and modo != "normal":
            modo = "auto"
        if modo == "normal" or not self.cfg.get("qr_zoom", True):
            self._modo_qr = None
            return
        normal = tuple(self._res_normal) if self._res_normal else tuple(self.cfg.get("resolucion", [640, 480]))
        if modo != "auto":
            if misma_proporcion(modo, normal):
                self._modo_qr = modo
                print("QR: resolución fija de la configuración: %dx%d" % modo)
                return
            print("QR: la resolución fija %dx%d tiene otra proporción que %dx%d (recorta la imagen); "
                  "la ignoro y pruebo sola." % (modo + normal))
        guardado = (self.cfg.get("modos_qr") or {}).get(self.clave_camara)
        # Solo vale lo elegido con la versión actual (misma proporción y campo de visión).
        if isinstance(guardado, dict) and guardado.get("v", 0) >= VERSION_MODO_QR:
            res = guardado.get("res")
            res = tuple(int(v) for v in res) if res else None
            if (res is None or misma_proporcion(res, normal)) and not self._ignorar_fija:
                self._modo_qr = res
                print("QR: resolución guardada para %s: %s" % (self.clave_camara,
                      "%dx%d" % res if res else "la normal"))
                return
        self._emitir("estado", "Preparando la cámara para leer el código de cerca. "
                               "Solo la primera vez tarda unos segundos.", True)
        zoom = ZoomQR(self.cfg, LectorQR())
        candidatas = [normal] + [tuple(int(v) for v in c)
                                 for c in (self.cfg.get("resoluciones_qr_candidatas") or RESOLUCIONES_QR)]
        otras = [c for c in candidatas if not misma_proporcion(c, normal)]
        if otras:
            print("QR: no pruebo %s: tienen otra proporción y recortan la imagen."
                  % ", ".join("%dx%d" % c for c in otras))
        candidatas = [c for c in candidatas if misma_proporcion(c, normal)]
        probadas, vistas = [], set()
        actual = normal
        ref_campo = None                         # esquinas de la placa (relativas) en la resolución normal
        for pedida in candidatas:
            if self._parar.is_set() or self.modo != "qr":
                break
            if pedida in vistas:
                continue
            antes = None
            if pedida != actual:
                antes = self.captura.cambios
                self.captura.pedir_resolucion(*pedida)
            fr = self._frames_nuevos(antes)
            if not fr:
                print("QR: %dx%d · la cámara no respondió" % pedida)
                continue
            f = fr[-1][1]
            real = (f.shape[1], f.shape[0])
            actual = real
            vistas.update({pedida, real})
            if any(p["res"] == real for p in probadas):
                continue
            if not misma_proporcion(real, normal):
                print("QR: pedí %dx%d y la cámara dio %dx%d: DESCARTADA, otra proporción (recorta la imagen)"
                      % (pedida + real))
                continue
            fps = (len(fr) - 1) / max(1e-6, fr[-1][0] - fr[0][0]) if len(fr) > 1 else 0.0
            loc = zoom.localizar(f, self.orientacion)
            dentro = bool(loc) and zoom.zonas(loc)[0][2]          # la esquina del QR, dentro de la imagen
            entera = bool(loc) and not loc["registro"].bordes_fuera(*real)
            segura = bool(loc) and loc.get("confianza") == "alta"
            # Mismo campo de visión que la resolución normal: las esquinas de la placa deben caer en
            # el mismo sitio RELATIVO de la imagen. Los modos panorámicos (1280×720, 1920×1080) de
            # muchas webcams recortan el sensor: la placa sale cortada y en otro sitio. Con la placa
            # cortada, la autodetección puede tomar la parte visible por la placa entera y "ver" la
            # esquina del QR donde solo hay braille: por eso esos modos se descartan.
            esq = loc["registro"].placa_a_imagen_arr(
                np.array([[0, 0], [zoom.plantilla.ancho_ref, 0], [zoom.plantilla.ancho_ref, zoom.plantilla.alto_ref],
                          [0, zoom.plantilla.alto_ref]], float)) / np.array(real, float) if loc else None
            if ref_campo is None and loc is not None and entera and segura and real == normal:
                ref_campo = esq
            desvio = None if (esq is None or ref_campo is None) else float(np.abs(esq - ref_campo).max())
            mismo_campo = (desvio <= 0.05) if desvio is not None else True
            formato = self.captura.formato
            detalle = None
            if dentro and entera and segura and mismo_campo:   # mediana de 3 imágenes: el ruido varía
                detalle = float(np.median([zoom.nitidez(x[1], loc) for x in fr[-3:]]))
            p = {"res": real, "fps": fps, "formato": formato, "placa": bool(loc), "dentro": dentro,
                 "entera": entera, "segura": segura, "mismo_campo": mismo_campo,
                 "px_por_mm": loc["px_por_mm"] if loc else None, "detalle": detalle}
            probadas.append(p)
            if not loc:
                estado = "sin placa a la vista"
            else:
                motivos = [m for m, ok in (("placa cortada", entera), ("esquina del QR fuera", dentro),
                                           ("detección dudosa", segura),
                                           ("otro campo de visión (recorta la imagen)", mismo_campo)) if not ok]
                estado = ("%.1f px/mm, " % loc["px_por_mm"]) + (
                    "DESCARTADA: " + ", ".join(motivos) if motivos else "placa entera, detalle real %.1f" % detalle)
            print("QR: %dx%d (%s, %.0f fps): %s" % (real[0], real[1], formato, fps, estado))
        validas = [p for p in probadas if p["fps"] >= 3.0 and p["detalle"] is not None]
        if validas:
            # Más píxeles no siempre es más detalle: muchas cámaras amplían por dentro (la imagen
            # sale más grande pero igual de borrosa). Se elige la resolución MENOR cuyo detalle real
            # llegue al 85 % del mejor: igual de legible, más rápida y con menos ruido.
            def peso(p):
                comprimido = p["formato"] == "MJPG" or p["res"][0] * p["res"][1] <= 1.2 * 640 * 480
                return p["detalle"] * (1.0 if comprimido else 0.85)
            tope = max(peso(p) for p in validas)
            buenas = [p for p in validas if peso(p) >= 0.85 * tope]
            mejor = min(buenas, key=lambda p: p["res"][0] * p["res"][1])["res"]
            self._modo_qr = None if mejor == normal else mejor
            self._emitir("modo_qr", self.clave_camara, {"v": VERSION_MODO_QR, "res": list(mejor) if self._modo_qr else None})
            print("QR: elijo %dx%d para leer el código." % mejor)
        else:
            self._modo_qr = None                    # no se vio la placa: se reintenta la próxima vez
            print("QR: en ninguna resolución vi la esquina del QR dentro de la imagen; uso la normal.")
        self._emitir("estado", "Listo. Leyendo el código de la esquina de la placa"
                     + (" con %d por %d píxeles." % self._modo_qr if self._modo_qr else "."), False)
        objetivo = self._modo_qr or normal
        if actual != objetivo:
            self.captura.pedir_resolucion(*objetivo)
            self._res_pedida, self._t_res = objetivo, time.time()   # que _ajustar_resolucion no lo repita

    # ---- QR ----
    def _trabajo_qr(self):
        """Decodifica QR en segundo plano: tarda cientos de ms y no debe frenar el video."""
        presupuesto = self.cfg.get("qr_presupuesto_ms")
        while not self._parar.is_set():
            if not self._qr_nuevo.wait(0.2):
                continue
            info = {"placa": False}
            try:
                if self.cfg.get("qr_zoom", True):
                    zoom = self._zoom_qr()
                    texto, poli = zoom.leer(self._qr_frame, self.orientacion, presupuesto)
                    info = dict(zoom.ultima)
                else:
                    texto, poli = self.lector.leer(self._qr_frame, presupuesto_ms=presupuesto)
                orient = self.lector.orientacion
            except Exception:
                traceback.print_exc()
                texto, poli, orient = None, None, None
            self._qr_res = (texto, poli, time.time(), orient, self._qr_frame.shape[:2], info)
            self._qr_nuevo.clear()

    def _paso_qr(self, crudo):
        """El QR se lee sobre la imagen tal cual llega de la cámara (en otro hilo, con zoom a
        la esquina de la placa). Lo que se muestra es la imagen ya orientada; cuando la placa
        está localizada, ampliada en la esquina que se está leyendo."""
        if self.lector is None:
            self.lector = LectorQR()
            self._hilo_qr = threading.Thread(target=self._trabajo_qr, daemon=True)
            self._hilo_qr.start()
        h, w = crudo.shape[:2]
        if not self._qr_nuevo.is_set():              # trabajador libre: dale el frame actual
            self._qr_frame = crudo.copy()
            self._qr_nuevo.set()
        ahora = time.time()
        res = self._qr_res
        if res is not None and res[4] != (h, w):     # resultado de otra resolución: ya no vale
            res = None
        info = res[5] if res is not None and len(res) > 5 and ahora - res[2] < 2.0 else {}
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
        frame, a_vista = self._vista_qr(crudo, info)
        if texto:
            if poli is not None:
                cv2.polylines(frame, [a_vista(poli).astype("int32").reshape(-1, 1, 2)], True, (0, 255, 0), 4)
            self._t_visto = ahora
            if texto != self._ultimo_qr or ahora - self._t_qr > 6:
                self._ultimo_qr, self._t_qr = texto, ahora
                if res[3] and self.cfg.get("orientacion_auto_qr", True):
                    self._fijar_orientacion(res[3], "qr")
                self._emitir("qr", texto)
        elif info.get("placa"):
            # Placa localizada: se está leyendo su esquina. La guía depende del detalle disponible.
            self._t_visto = ahora
            if self._t_zoom is None:
                self._t_zoom = ahora
            if poli is not None:
                cv2.polylines(frame, [a_vista(poli).astype("int32").reshape(-1, 1, 2)], True, (0, 255, 255), 3)
            ppm = info.get("px_por_modulo")
            if ppm is not None and ppm < float(self.cfg.get("qr_px_por_modulo_min", 2.2)):
                self._hablar_guia("Veo la placa, pero desde esta altura la cámara no tiene detalle suficiente "
                                  "para leer el código. Acerca a la cámara la esquina del código, o pulsa "
                                  "Escape y usa F2 para elegir la gráfica.", cada=15)
            elif not self._zoom_avisado:
                self._zoom_avisado = self._hablar_guia("Encontré la placa. Estoy leyendo el código de su "
                                                       "esquina; mantenla quieta.", cada=0)
            elif ahora - self._t_zoom > 15:
                self._hablar_guia("Todavía no puedo leer el código. Puede tener poco contraste o estar "
                                  "dañado. Puedes acercarlo a la cámara, o pulsar Escape y usar F2 para "
                                  "elegir la gráfica.", cada=20)
        elif poli is not None:
            cv2.polylines(frame, [a_vista(poli).astype("int32").reshape(-1, 1, 2)], True, (0, 255, 255), 3)
            self._t_visto = ahora
            lado = math.sqrt(abs(cv2.contourArea(np.asarray(poli, np.float32).reshape(-1, 1, 2))))
            if lado < 70:                       # px: con menos, los módulos no se distinguen
                self._hablar_guia("Veo el código pero se ve muy pequeño. Acerca la gráfica a la cámara.")
            elif lado > 0.7 * min(w, h):
                self._hablar_guia("El código está muy cerca. Aleja un poco la gráfica.")
            else:
                self._hablar_guia("Código detectado. Mantén la gráfica quieta un momento.")
        elif ahora - self._t_visto > 10:
            self._hablar_guia("No veo el código QR. Muévelo despacio frente a la cámara, "
                              "con el código mirando hacia la cámara.", cada=10)
        cv2.putText(frame, "Buscando codigo QR..." if not info.get("placa") else "Zoom: esquina del codigo QR",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame

    def _vista_qr(self, crudo, info):
        """(imagen para mostrar, función px de la imagen original → px de esa imagen).
        Con la placa localizada se muestra ampliada la esquina que se lee: así una persona
        que ayude ve exactamente lo que la cámara intenta leer."""
        caja = info.get("caja")
        if not info.get("placa") or caja is None or not self.cfg.get("qr_zoom_vista", True):
            return self._voltear(crudo), self._a_vista
        h, w = crudo.shape[:2]
        x0, y0, x1, y1 = caja
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        lado = max(x1 - x0, y1 - y0) * 1.6
        rx0, ry0 = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
        rx1, ry1 = int(min(w, cx + lado / 2)), int(min(h, cy + lado / 2))
        if rx1 - rx0 < 16 or ry1 - ry0 < 16:
            return self._voltear(crudo), self._a_vista
        recorte = crudo[ry0:ry1, rx0:rx1].copy()
        # Que la vista ampliada tenga al menos el tamaño de la imagen normal (se ve como un zoom).
        k = max(1.0, min(4.0, 480.0 / recorte.shape[0]))
        if k > 1.0:
            recorte = cv2.resize(recorte, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        rh, rw = recorte.shape[:2]
        M = matriz_orientacion(self.orientacion, rw, rh) @ np.array([[k, 0, -k * rx0], [0, k, -k * ry0], [0, 0, 1.0]])
        return aplicar_orientacion(recorte, self.orientacion), (lambda pts: _aplicar_h(M, pts))

    # ---- registro ----
    def _construir_registro(self):
        """Registro actual a partir de las esquinas (o del rectángulo 'hoja') y de la gráfica."""
        try:
            self.registro = registro_para(self.datos, self.esquinas, self.hoja,
                                          self.origen_registro, self.error_registro)
        except ValueError:
            self.registro = registro_para(self.datos, None, self.hoja, "rect")
        self.superposicion.configurar(self.datos, self.registro)
        if self.interaccion is not None and self.datos is not None:
            self.interaccion.configurar(self.datos, self.registro)
        self._reconfigurar = False

    def _info_placa(self, ok, origen, det=None, w=0, h=0, **extra):
        """Lo que la GUI necesita para hablar y guardar la calibración."""
        info = {"ok": ok, "origen": origen, "camara": self.clave_camara, "resolucion": [w, h],
                "esquina_esperada": getattr(self.datos, "esquina_recortada", "") if self.datos else "",
                "hoja": list(self.hoja), "bordes": [], "error_mm": None}
        if self.esquinas is not None and ok:
            info["esquinas"] = self.esquinas.round(2).tolist()
        if self.registro is not None:
            info["bordes"] = self.registro.bordes_fuera(w, h)
            info["error_mm"] = self.registro.error_estimado_mm
        if det is not None:
            info.update({"esquina_vista": det.esquina_vista, "giro": det.giro, "metodo": det.metodo,
                         "confianza": det.confianza, "reconstruida": det.reconstruida, "pct": det.area_pct,
                         "rotacion_grados": det.rotacion_grados})
        info.update(extra)
        return info

    def _qr_en_placa(self, crudo):
        """Lee el QR de la placa (imagen sin orientar). Devuelve sus esquinas en la imagen
        orientada (orden del propio código) o None. De paso endereza la imagen con él."""
        if self._lector_ajuste is None:
            self._lector_ajuste = LectorQR()
        roi = None
        qr_mm = esquinas_qr_mm(self.datos)
        if self.registro is not None and qr_mm is not None and self._dim_raw is not None:
            # Zona donde debería estar el QR según el registro actual (con margen amplio).
            w, h = self._dim_raw
            M = np.linalg.inv(matriz_orientacion(self.orientacion, w, h))
            c = qr_mm.mean(axis=0)
            zona = c + (qr_mm - c) * 3.0
            p = _aplicar_h(M, self.registro.placa_a_imagen_arr(zona))
            roi = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
        try:
            texto, _p = self._lector_ajuste.leer(crudo, roi=roi, presupuesto_ms=250)
            if not texto and roi is not None:
                texto, _p = self._lector_ajuste.leer(crudo, presupuesto_ms=250)
        except Exception:
            traceback.print_exc()
            return None
        if not texto:
            return None
        if self._lector_ajuste.orientacion and self.cfg.get("orientacion_auto_qr", True):
            self._fijar_orientacion(self._lector_ajuste.orientacion, "qr")
        esq = self._lector_ajuste.esquinas
        return None if esq is None else self._a_vista(esq)

    def _aceptar(self, esquinas, origen, error_mm):
        self.esquinas = np.asarray(esquinas, float).reshape(4, 2)
        self.origen_registro, self.error_registro = origen, error_mm
        self._construir_registro()
        self.hoja = self.registro.caja_px() if self.registro is not None else self.hoja

    def _buscar_placa(self, frame, origen, qr_px=None):
        """Autodetección (tecla A o al empezar a explorar). Emite un evento "placa"."""
        h, w = frame.shape[:2]
        previo = self.registro
        det = detectar_placa(frame, self.datos, preferido=self._metodo_placa)
        estricto = origen == "inicio" and previo is not None and previo.origen != "rect"
        if det is not None and (det.confianza == "alta" or not estricto):
            self._metodo_placa = det.metodo
            nuevo = det.registro(self.datos)
            extra = {}
            qr_mm = esquinas_qr_mm(self.datos)
            if qr_px is not None and qr_mm is not None:
                d_c, lado_v, lado_csv = residuo_qr_mm(nuevo, qr_px, qr_mm)
                extra.update(qr_centro_mm=d_c, qr_lado_visto_mm=lado_v, qr_lado_csv_mm=lado_csv)
                print(f"QR como control: su centro cae a {d_c:.1f} mm de donde dice el CSV; "
                      f"lado visto {lado_v:.1f} mm, en el CSV {lado_csv:.1f} mm.")
                if nuevo.error_estimado_mm is not None and d_c < 15:
                    nuevo.error_estimado_mm = max(nuevo.error_estimado_mm, d_c)
            if previo is not None:
                med, mx = previo.diferencia_mm(nuevo)
                extra["desplazamiento_mm"] = mx
            self._aceptar(det.esquinas, "autodeteccion", nuevo.error_estimado_mm)
            self._emitir("placa", self._info_placa(True, origen, det, w, h, **extra))
            return True
        # Sin placa fiable: si se ve el QR, al menos se corrige el desplazamiento y el giro.
        qr_mm = esquinas_qr_mm(self.datos)
        if previo is not None and qr_px is not None and qr_mm is not None and self.cfg.get("qr_ancla", True):
            M, giro, despl = corregir_con_qr(previo, qr_px, qr_mm)
            if despl > 2 or abs(giro) > 1:
                corregido = previo.transformado(M, "qr")
                corregido.error_estimado_mm = None
                self._aceptar(corregido.esquinas_px(), "qr", None)
                self._emitir("placa", self._info_placa(True, "qr", None, w, h, desplazamiento_mm=despl,
                                                       giro_qr_grados=giro, motivo_original=origen))
                return True
        self._emitir("placa", self._info_placa(False, origen, det, w, h,
                                               guardada_otra_camara=self._guardada_dudosa(w, h)))
        return False

    def _guardada_dudosa(self, w, h):
        cam = self.cfg.get("registro_camara")
        res = self.cfg.get("registro_resolucion")
        return bool((cam and cam != self.clave_camara) or (res and list(res) != [w, h]))

    def _trabajo_placa(self):
        """Seguimiento de la placa en segundo plano (no frena el video)."""
        while not self._parar.is_set():
            if not self._placa_nueva.wait(0.3):
                continue
            frame, datos, orient = self._placa_frame
            try:
                det = detectar_placa(frame, datos, preferido=self._metodo_placa)
            except Exception:
                traceback.print_exc()
                det = None
            self._placa_res = (det, orient, frame.shape[:2])
            self._placa_nueva.clear()

    def _toca_seguir(self):
        """¿Hay que mandar ahora un frame al seguimiento de la placa? (cada pocos segundos,
        sin mano a la vista desde hace 1 s y con el trabajador libre)."""
        return (bool(self.cfg.get("seguir_placa", True)) and self.datos is not None and self.registro is not None
                and time.time() - self._t_placa >= float(self.cfg.get("seguir_placa_cada_s", 2.5))
                and time.time() - self.t_mano >= 1.0 and not self._placa_nueva.is_set())

    def _seguir_placa(self, limpio, w, h):
        """¿Se movió la placa? Se acepta el cambio solo si dos detecciones seguidas, de
        confianza alta, coinciden. 'limpio' es la imagen sin la superposición (o None si
        no toca mandar una nueva al trabajador)."""
        if self.datos is None or self.registro is None:
            return
        res, self._placa_res = self._placa_res, None
        if res is not None:
            det, orient, forma = res
            if det is not None and det.confianza == "alta" and orient == self.orientacion and forma == (h, w):
                nuevo = det.registro(self.datos)
                _med, mx = self.registro.diferencia_mm(nuevo)
                if mx > float(self.cfg.get("umbral_movimiento_placa_mm", 3.0)):
                    if self._candidata is not None and self._candidata.diferencia_mm(nuevo)[1] < 1.5:
                        self._aceptar(det.esquinas, "autodeteccion", nuevo.error_estimado_mm)
                        self._emitir("placa", self._info_placa(True, "seguimiento", det, w, h, desplazamiento_mm=mx))
                        self._candidata = None
                    else:
                        self._candidata = nuevo
                else:
                    self._candidata = None
        if limpio is None:
            return
        self._t_placa = time.time()
        if self._hilo_placa is None:
            self._hilo_placa = threading.Thread(target=self._trabajo_placa, daemon=True)
            self._hilo_placa.start()
        self._placa_frame = (limpio, self.datos, self.orientacion)
        self._placa_nueva.set()

    def _avisar_bordes(self, w, h):
        """Si la placa se sale de la imagen, lo dice (como mucho cada 20 s)."""
        if self.registro is None or time.time() - self._t_aviso_borde < 20:
            return
        if self.registro.bordes_fuera(w, h):
            self._t_aviso_borde = time.time()
            self._emitir("borde", self.registro.bordes_fuera(w, h))

    def _crear_motores(self):
        """Detector de la mano (A) e interacción (B). Devuelve False si falló."""
        try:
            if self.detector is None:
                self._emitir("estado", "Cargando el seguimiento de la mano. Un momento.", True)
                self.detector = DetectorMano(self.cfg, avisar=lambda t: self._emitir("estado", t, True))
            if self.interaccion is None and self._crear_interaccion is not None:
                self.interaccion = self._crear_interaccion(self.detector, lambda t: self._emitir("estado", t, True))
        except ImportError:
            traceback.print_exc()
            self._emitir("error_camara", "Falta la biblioteca mediapipe, o no se pudo cargar. "
                                         "Instálala con: pip install --upgrade mediapipe")
            self.modo = "vista"
            return False
        except ErrorSeguimiento as e:
            self._emitir("error_camara", str(e))
            self.modo = "vista"
            return False
        self._reconfigurar = True
        return True

    def _paso_explorar(self, crudo):
        qr_px = None
        self._n_explorar += 1
        # La comprobación al empezar espera ~1 s: la cámara aún ajusta exposición y enfoque.
        comprobar = self._comprobar_placa and self.datos is not None and self._n_explorar >= 20
        if self._autodetectar or (comprobar and self.cfg.get("autodetectar_al_empezar", True)):
            # Tecla A (o al empezar): si el QR de la placa se ve, primero se endereza la
            # imagen con él; sus esquinas sirven además para comprobar el registro.
            if self._reconfigurar and self.datos is not None:
                self._construir_registro()
            qr_px = self._qr_en_placa(crudo)
        frame = self._voltear(crudo)
        h, w = frame.shape[:2]
        if (self.detector is None or (self.interaccion is None and self._crear_interaccion is not None)) \
                and not self._crear_motores():
            return frame
        if self._esquinas_pedidas is not None:
            (esq, origen), self._esquinas_pedidas = self._esquinas_pedidas, None
            self._aceptar(esq, origen, None)
            self._emitir("placa", self._info_placa(True, origen, None, w, h))
        if self._reconfigurar and self.datos is not None:
            self._construir_registro()
        if self._autodetectar:
            self._autodetectar = False
            self._comprobar_placa = False
            self._buscar_placa(frame, "autodeteccion", qr_px)
        elif comprobar:
            self._comprobar_placa = False
            if self.cfg.get("autodetectar_al_empezar", True):
                self._buscar_placa(frame, "inicio", qr_px)
            else:
                self._avisar_bordes(w, h)
        if self.datos is None or self.registro is None:
            return frame

        roi = roi_mano(self.registro, w, h, float(self.cfg.get("mano_margen_recorte", 0.4))) \
            if self.cfg.get("mano_recorte_placa", True) else None
        # Copia limpia para el seguimiento de la placa (la superposición se dibuja encima).
        limpio = frame.copy() if self._toca_seguir() else None
        manos = self.detector.detectar(frame, roi)
        puntos = manos[0] if manos else None
        if puntos is not None:
            self.t_mano = time.time()
        muestra = muestra_de_mano(puntos, w, h, self.registro)
        sel = self.interaccion.actualizar(muestra) if self.interaccion is not None else None
        ident = getattr(sel, "elemento_id", None)
        elem = self.superposicion.por_id.get(ident) if ident else None
        visible = elem["corto"] if elem else None
        dedos = dedos_px(puntos, w, h)[:2] if puntos is not None else None
        self.superposicion.dibujar(frame, ident, dedos, bool(getattr(sel, "pinza", False)), visible)

        r = self.medidor.contar(self.captura.n if self.captura else 0, self.detector.ms_ultimo)
        if r and self.cfg.get("depurar_vision", True):
            print("Visión: cámara %.1f fps · proceso %.1f fps · modelo de la mano %s ms%s" % (
                r["fps_camara"], r["fps_proceso"], "?" if r["ms_mano"] is None else "%.0f" % r["ms_mano"],
                " (recortado a la placa)" if roi else ""))
        self._seguir_placa(limpio, w, h)
        if time.time() - self.t_mano > 10:
            self._hablar_guia("No veo tu mano. Colócala frente a la cámara, sobre la gráfica.", cada=15)
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
        captura = Captura(cap, self._mjpg)
        self.captura = captura
        self._res_pedida, self._t_res = None, 0.0
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
                    if not self._ajustar_resolucion(frame):
                        self._enviar_frame(self._voltear(frame))      # aún llega la resolución del QR
                        continue
                    self._dim_raw = (frame.shape[1], frame.shape[0])
                    if self._orient_pedida is not None:
                        pedida, self._orient_pedida = self._orient_pedida, None
                        self._fijar_orientacion(*pedida)
                    if self.modo == "qr" and not self._sondeado:
                        self._sondear_modo_qr()            # elige la resolución para el QR (una vez)
                        continue
                    if self.modo == "qr":
                        frame = self._paso_qr(frame)
                    elif self.modo == "explorar":
                        frame = self._paso_explorar(frame)
                    else:
                        frame = self._voltear(frame)
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
            if self.detector is not None:
                self.detector.cerrar()
