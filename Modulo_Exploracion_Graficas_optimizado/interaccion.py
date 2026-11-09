#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTERACCIÓN  ·  Explorador de gráficas impresas con textura  (Persona B)
=======================================================================

Decide qué elemento de la gráfica se anuncia. Trabaja directamente en milímetros
de la placa: recibe una vision.MuestraMano por frame y devuelve una Seleccion
(contrato con la Persona A: configurar(datos, registro) y actualizar(muestra)).

Cada frame pasa por estos pasos:

    1. Pinza       distancia índice–medio dividida por el largo de la palma (no depende
                   de lo lejos que esté la cámara), con dos umbrales (histéresis: uno
                   para cerrar y otro, más alto, para abrir) y un tiempo de gracia: un
                   instante con los dedos separados o sin ver la mano no corta el audio.
    2. Suavizado   filtro "1 euro" sobre la yema del índice: quieto quita el temblor,
                   en movimiento casi no retrasa.
    3. Búsqueda    elementos al alcance (cada uno con su radio en mm):
                     · datos: punto > curva/tramo/barra, y a igualdad el más cercano;
                     · ejes, textos y leyendas compiten por distancia y solo le ganan a
                       un dato si están claramente más cerca (margen_resto_mm). Así un
                       eje deja de quedar tapado por la curva o el punto que pasa cerca;
                     · cruces: se conserva la serie que se venía siguiendo;
                     · el elemento actual tiene un pequeño margen extra (histéresis).
    4. Estabilidad un cambio de elemento se acepta solo si se mantiene unos ms.
    5. Narración   con la pinza cerrada, al cambiar de elemento se interrumpe lo que se
                   estaba diciendo y se lee el nuevo; se avisa el cruce con otra serie y
                   cuando la yema sale de la placa. Al soltar la pinza se calla.

Todas las constantes se pueden cambiar en config_exploracion.json (ver DEFECTO).
Este módulo solo importa vision.py (nada del explorador: no hay importaciones circulares).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from vision import Registro, MuestraMano  # noqa: F401  (MuestraMano: tipo de la entrada)


# ============================================================
# CONSTANTES
# ============================================================
# Datos: punto > curva/tramo/barra. Ejes, textos y leyendas (prioridad >= PRIORIDAD_RESTO)
# compiten entre sí por distancia y solo ganan a un dato si están claramente más cerca.
PRIORIDAD = {"punto": 0, "curva": 1, "tramo": 1, "barra": 1, "texto": 2, "leyenda": 2, "eje": 3}
PRIORIDAD_RESTO = 2
TIPOS_DATOS = ("punto", "curva", "tramo", "barra")
TIPOS_LINEA = ("curva", "tramo")
PX_POR_MM_CLASICO = 1.5   # formato clásico (lienzo sin mm): escala supuesta de la hoja de siempre

DEFECTO = {
    # --- radios de alcance, en mm de la placa (formato handtracking) ---
    "radio_punto_mm": 7.0,          # como mucho; con puntos de la misma serie muy juntos se reduce
    "radio_punto_min_mm": 3.0,
    "radio_linea_mm": 8.0,
    "radio_eje_mm": 7.0,            # compite por distancia con los textos de las marcas, que van al lado
    "radio_texto_mm": 7.0,
    # (valores elegidos con el análisis sin cámara de medir_metricas.py, modo geometria,
    #  sobre las gráficas 1 y 2: ver respaldo_antes_parte_B/ para la versión anterior)
    "margen_resto_mm": 3.0,         # cuánto más cerca debe estar un eje/texto para ganarle a un dato
    "ventaja_punto_mm": 3.0,        # un punto gana a una curva si no está más lejos que ella + esto
                                    # (None = el punto gana siempre que esté a su alcance)
    # --- estabilidad ---
    "histeresis_mm": 2.0,           # margen extra del elemento actual (para no saltar en el borde)
    "seguir_serie_s": 1.5,          # cuánto se recuerda la serie que se venía siguiendo
    "ventaja_serie_mm": 3.0,        # en un cruce, ventaja de esa serie frente a la otra
    "entrada_ms": 60,               # de nada a un elemento
    "cambio_ms": 120,               # de un elemento a otro
    "salida_ms": 150,               # de un elemento a nada
    "suavizado": True,
    "suavizado_min_hz": 1.5,        # filtro 1 euro: menos = más suave en reposo
    "suavizado_beta": 0.05,         # más = menos retraso al moverse rápido
    # --- pinza ---
    "pinza_cerrar_palma": 0.28,     # dedos juntos si índice–medio <= 0,28 × palma
    "pinza_abrir_palma": 0.36,      # y separados solo si >= 0,36 × palma (histéresis)
    "pinza_cerrar_mm": 22.0,        # lo mismo en mm, si la palma no es fiable
    "pinza_abrir_mm": 28.0,
    "palma_min_mm": 45.0,           # palma (muñeca–base del medio) plausible en la placa
    "palma_max_mm": 160.0,
    "gracia_pinza_ms": 250,         # dedos separados durante menos de esto: la pinza sigue cerrada
    "gracia_mano_ms": 300,          # mano perdida durante menos de esto: tampoco se corta
    # --- narración ---
    "interrumpir_narracion": True,  # un elemento nuevo corta la frase anterior
    "callar_al_soltar": True,       # al soltar la pinza se corta el audio
    "anunciar_cruces": True,        # "Cruce con ..." al pasar por un cruce de series
    "cruce_mm": 3.0,                # distancia al punto de cruce para avisarlo
    "avisar_fuera_placa": True,     # "Fuera de la placa" con la pinza cerrada fuera de ella
    "fuera_placa_ms": 400,
    "fuera_placa_cada_s": 5.0,
}
ADMITEN_NONE = {"ventaja_punto_mm"}      # en estas claves, null en el JSON tiene significado


# ============================================================
# GEOMETRÍA (misma fórmula para mm o px)
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


def interseccion_segmentos(a, b):
    """Punto donde se cortan los segmentos a y b (dicts con x1..y2), o None."""
    rx, ry = a["x2"] - a["x1"], a["y2"] - a["y1"]
    sx, sy = b["x2"] - b["x1"], b["y2"] - b["y1"]
    rxs = rx * sy - ry * sx
    if abs(rxs) < 1e-9:
        return None
    qx, qy = b["x1"] - a["x1"], b["y1"] - a["y1"]
    t = (qx * sy - qy * sx) / rxs
    u = (qx * ry - qy * rx) / rxs
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return a["x1"] + t * rx, a["y1"] + t * ry
    return None


def es_dato(e):
    return PRIORIDAD.get(e["tipo"], PRIORIDAD_RESTO) < PRIORIDAD_RESTO


# ============================================================
# CONTRATO CON LA PERSONA A
# ============================================================
@dataclass
class Seleccion:
    """Lo que decide la interacción en cada frame (contrato con la Persona A)."""
    pinza: bool
    elemento_id: str | None
    candidatos: list = field(default_factory=list)    # lo que estaba al alcance, para depurar cruces


class _Nada:
    """Marca 'pendiente de quedarse sin elemento' (distinta de 'nada pendiente')."""


_NADA = _Nada()


class _VozMuda:
    """Para usar Interaccion sin audio (medir_metricas.py, pruebas)."""

    class _Evento:
        def is_set(self):
            return False

    hablando = _Evento()

    def decir(self, texto, interrumpir=False, esencial=False):
        pass

    def detener(self):
        pass


class FiltroUnEuro:
    """Filtro "1 euro" (Casiez, Roussel y Vogel, 2012) para una posición 2D.

    La frecuencia de corte crece con la velocidad: en reposo filtra mucho (quita el
    temblor) y al moverse rápido filtra poco (no deja la yema atrás)."""

    def __init__(self, min_hz=1.5, beta=0.05, d_hz=1.0):
        self.min_hz, self.beta, self.d_hz = float(min_hz), float(beta), float(d_hz)
        self.reiniciar()

    def reiniciar(self):
        self.t = None
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0

    @staticmethod
    def _alfa(dt, hz):
        tau = 1.0 / (2 * math.pi * hz)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x, y):
        if self.t is None:
            self.t, self.x, self.y = t, x, y
            return x, y
        dt = t - self.t
        if dt <= 0:
            return self.x, self.y
        a_d = self._alfa(dt, self.d_hz)
        self.vx = a_d * (x - self.x) / dt + (1 - a_d) * self.vx
        self.vy = a_d * (y - self.y) / dt + (1 - a_d) * self.vy
        a = self._alfa(dt, self.min_hz + self.beta * math.hypot(self.vx, self.vy))
        self.x = a * x + (1 - a) * self.x
        self.y = a * y + (1 - a) * self.y
        self.t = t
        return self.x, self.y


# ============================================================
# INTERACCIÓN
# ============================================================
class Interaccion:
    """Selección con la pinza y narración, en mm de la placa.

    Interfaz para vision.MotorVision:
        configurar(datos, registro)       al cargar la gráfica o cambiar la calibración
        actualizar(muestra) -> Seleccion  una vez por frame
    'registro' es un vision.Registro o, como antes, el rectángulo 'hoja' (x, y, ancho, alto).
    """

    def __init__(self, voz, cfg, avisar=None):
        self.voz = voz if voz is not None else _VozMuda()
        self.cfg = cfg if cfg is not None else {}
        self.avisar = avisar
        self.datos = None
        self.registro = None
        self.elementos = []
        self.por_id = {}
        self.cruces = []
        self._escala = 1.0               # unidades del lienzo por px del frame (solo formato clásico)
        self.reiniciar()

    # ---------- configuración ----------
    def c(self, clave):
        """Valor de configuración, con el valor por defecto de este módulo."""
        v = self.cfg.get(clave, DEFECTO[clave])
        return DEFECTO[clave] if v is None and clave not in ADMITEN_NONE else v

    def clave_dato(self, e, d):
        """Orden entre datos (menor gana): (prioridad, distancia efectiva).
        Con 'ventaja_punto_mm' (prioridad suave) todos los datos comparten prioridad y un punto
        cuenta como si estuviera esa distancia más cerca: gana a la curva que pasa por él, pero
        no a la línea de otra serie que el dedo está tocando a unos mm de él."""
        ventaja = self.c("ventaja_punto_mm")
        if ventaja is None:
            return PRIORIDAD.get(e["tipo"], PRIORIDAD_RESTO), d
        return 0, (d - self._mm("ventaja_punto_mm") if e["tipo"] == "punto" else d)

    @property
    def _en_mm(self):
        return self.datos is None or self.datos.formato == "handtracking"

    def _mm(self, clave):
        """Una distancia de la configuración (en mm) en unidades del lienzo. En el formato
        clásico no se sabe cuánto mide el lienzo: se supone la escala de la hoja de siempre."""
        v = float(self.c(clave))
        return v if self._en_mm else v * PX_POR_MM_CLASICO * self._escala

    def _px(self, clave, defecto):
        """Una distancia en px del frame (configuración antigua) en unidades del lienzo."""
        return float(self.cfg.get(clave, defecto)) * self._escala

    def configurar(self, datos, registro):
        if registro is not None and not isinstance(registro, Registro):
            registro = Registro.desde_rect(registro, datos.ancho_ref, datos.alto_ref, datos.estirar)
        self.datos, self.registro = datos, registro
        self._escala = self._calcular_escala()
        self.elementos = [dict(e) for e in datos.elementos] if datos is not None else []
        self.por_id = {e["id"]: e for e in self.elementos}
        self._asignar_radios()
        self.cruces = self._buscar_cruces()
        self.reiniciar()

    def _calcular_escala(self):
        """Unidades del lienzo por px del frame. En el formato handtracking no se usa (el
        lienzo ya está en mm). En el formato clásico el lienzo tiene unidades propias: los
        radios en px de siempre se pasan a esas unidades con el registro."""
        if self._en_mm or self.registro is None:
            return 1.0
        try:
            ppu = self.registro.px_por_mm()          # px por unidad del lienzo
        except Exception:
            return 1.0
        return 1.0 / ppu if ppu and math.isfinite(ppu) and ppu > 0 else 1.0

    def _asignar_radios(self):
        """Radio de alcance de cada elemento. Un punto usa como mucho el 40 % de la distancia
        al punto más cercano DE SU SERIE: así el tramo central de cada curva se puede tocar,
        y un punto no se encoge porque pase cerca otra serie (antes, en los cruces, los
        puntos quedaban de 2-3 mm y se perdían)."""
        if self._en_mm:
            r_pt, r_min = self._mm("radio_punto_mm"), self._mm("radio_punto_min_mm")
            r_lin, r_eje, r_txt = self._mm("radio_linea_mm"), self._mm("radio_eje_mm"), self._mm("radio_texto_mm")
        else:                                            # formato clásico: los radios en px de siempre
            r_pt, r_min = self._px("radio_punto", 20), 4.0 * self._escala
            r_lin, r_eje, r_txt = self._px("radio_linea", 15), self._px("radio_eje", 20), self._px("radio_texto", 12)
        for e in self.elementos:
            if e["tipo"] == "punto":
                e["radio"] = r_pt
            elif e["tipo"] in TIPOS_DATOS:
                e["radio"] = r_lin
            elif e["tipo"] == "eje":
                e["radio"] = r_eje
            else:
                e["radio"] = r_txt
        por_serie = {}
        for e in self.elementos:
            if e["tipo"] == "punto":
                por_serie.setdefault(e.get("serie", ""), []).append(e)
        for pts in por_serie.values():
            for e in pts:
                vecino = min((distancia(e["x1"], e["y1"], o["x1"], o["y1"]) for o in pts if o is not e),
                             default=math.inf)
                e["radio"] = max(min(r_min, r_pt), min(r_pt, 0.4 * vecino))

    def _buscar_cruces(self):
        """Puntos donde se cortan curvas de series distintas: (x, y, id_a, id_b)."""
        curvas = [e for e in self.elementos if e["tipo"] in TIPOS_LINEA and e["forma"] == "segmento"
                  and e.get("serie")]
        if len(curvas) > 800:            # gráfica enorme: no merece la pena (O(n²))
            return []
        res = []
        for i, a in enumerate(curvas):
            for b in curvas[i + 1:]:
                if a["serie"] == b["serie"]:
                    continue
                p = interseccion_segmentos(a, b)
                if p is not None:
                    res.append((p[0], p[1], a["id"], b["id"]))
        return res

    def reiniciar(self):
        """Olvida la mano: pinza abierta, sin elemento, sin serie seguida."""
        self.pinza = False
        self.audio_activado = False      # nombre antiguo (Explorador), mismo valor que 'pinza'
        self.pinza_antes = False
        self.medida_pinza = None         # última proporción índice–medio / palma (o distancia)
        self._t_abre = None
        self._t_ultima_mano = None
        self.filtro = FiltroUnEuro(self.c("suavizado_min_hz"), self.c("suavizado_beta"))
        self.posicion = None             # yema del índice suavizada (unidades del lienzo)
        self.estable = None              # elemento elegido (dict) tras el filtro de estabilidad
        self._pendiente, self._t_pendiente = None, 0.0
        self._serie, self._t_serie = "", -math.inf
        self.anterior = None             # último id anunciado (medir_metricas lo lee)
        self._cruces_dichos = set()
        self._t_fuera, self._t_aviso_fuera = None, -math.inf
        self.t_mano = time.time()

    def cerrar(self):
        pass

    # ---------- búsqueda ----------
    def _radio(self, e):
        return e["radio"] if "radio" in e else self._mm("radio_texto_mm")

    def _buscar(self, x, y, candidatos=None, actual=None, serie=""):
        """Elemento bajo la yema (x, y en unidades del lienzo).

        · Datos (punto, curva, tramo, barra): gana el más cercano, con ventaja para los
          puntos (clave_dato).
        · Ejes, textos y leyendas: gana el más cercano; le quitan el sitio a un dato solo
          si están más cerca que él por más de 'margen_resto_mm'.
        · 'serie': la que se venía siguiendo. Si alguno de sus elementos está al alcance,
          gana salvo que el otro esté más cerca por más de 'ventaja_serie_mm'.
        · 'actual': el elemento elegido ahora; alcanza 'histeresis_mm' más y cuenta como
          si estuviera esa distancia más cerca.
        Sin 'actual' ni 'serie' el resultado depende solo de (x, y): es lo que reproduce
        el mapa de medir_metricas.py. Si se pasa una lista en 'candidatos', se llena con
        los ids al alcance."""
        hist = self._mm("histeresis_mm") if actual is not None else 0.0
        dato = kd = resto = kr = de_serie = ks = None
        for e in self.elementos:
            d = distancia_elemento(x, y, e)
            es_actual = e is actual
            if d > e["radio"] + (hist if es_actual else 0.0):
                continue
            if candidatos is not None:
                candidatos.append(e["id"])
            if es_actual:
                d -= hist
            p = PRIORIDAD.get(e["tipo"], PRIORIDAD_RESTO)
            if p < PRIORIDAD_RESTO:
                k = self.clave_dato(e, d)
                if kd is None or k < kd:
                    dato, kd = e, k
                if serie and e.get("serie") == serie and (ks is None or k < ks):
                    de_serie, ks = e, k
            else:
                k = (d, p)
                if kr is None or k < kr:
                    resto, kr = e, k
        if de_serie is not None and dato is not de_serie and ks[1] - self._mm("ventaja_serie_mm") <= kd[1]:
            dato, kd = de_serie, ks
        if dato is None:
            return resto
        if resto is None:
            return dato
        return resto if kr[0] + self._mm("margen_resto_mm") < kd[1] else dato

    # ---------- pinza ----------
    def _medir_pinza(self, m):
        """(medida, umbral_cerrar, umbral_abrir): con la palma si es fiable; si no, en distancia."""
        dist = distancia(m.indice_mm[0], m.indice_mm[1], m.medio_mm[0], m.medio_mm[1])
        palma = m.palma_mm
        if self._en_mm:
            fiable = palma is not None and self.c("palma_min_mm") <= palma <= self.c("palma_max_mm")
        else:                                            # lienzo sin mm: basta con que exista
            fiable = palma is not None and palma > 1e-6
        if fiable:
            return dist / palma, float(self.c("pinza_cerrar_palma")), float(self.c("pinza_abrir_palma"))
        if not self._en_mm:                              # el umbral en px de siempre
            u = self._px("distancia_pinza", 35)
            return dist, u, u * 1.25
        return dist, float(self.c("pinza_cerrar_mm")), float(self.c("pinza_abrir_mm"))

    def _actualizar_pinza(self, m, t):
        if m is None:                                   # sin mano: se mantiene durante la gracia
            if self.pinza and (self._t_ultima_mano is None
                               or (t - self._t_ultima_mano) * 1000 >= self.c("gracia_mano_ms")):
                self.pinza = False
            self._t_abre = None
            return
        self._t_ultima_mano = t
        medida, cerrar, abrir = self._medir_pinza(m)
        self.medida_pinza = medida
        if not self.pinza:
            if medida <= cerrar:
                self.pinza, self._t_abre = True, None
        elif medida >= abrir:
            if self._t_abre is None:
                self._t_abre = t
            if (t - self._t_abre) * 1000 >= self.c("gracia_pinza_ms"):
                self.pinza, self._t_abre = False, None
        else:
            self._t_abre = None

    # ---------- estabilidad ----------
    def _serie_seguida(self, t):
        return self._serie if t - self._t_serie <= float(self.c("seguir_serie_s")) else ""

    def _estabilizar(self, crudo, t):
        """El elemento elegido cambia solo si el nuevo (o 'nada') se mantiene unos ms."""
        objetivo = _NADA if crudo is None else crudo
        if objetivo is (_NADA if self.estable is None else self.estable):
            self._pendiente = None
            return
        if self._pendiente is not objetivo:
            self._pendiente, self._t_pendiente = objetivo, t
        espera = (self.c("entrada_ms") if self.estable is None else
                  self.c("salida_ms") if crudo is None else self.c("cambio_ms"))
        if (t - self._t_pendiente) * 1000 >= espera:
            self.estable, self._pendiente = crudo, None

    # ---------- frame a frame ----------
    def actualizar(self, muestra):
        """Una MuestraMano (mm) → Seleccion. Narra solo con la pinza cerrada."""
        t = muestra.t if muestra is not None and muestra.t is not None else time.monotonic()
        candidatos = []
        hay = (muestra is not None and muestra.indice_mm is not None and muestra.medio_mm is not None
               and self.datos is not None)
        self._actualizar_pinza(muestra if hay else None, t)
        if hay:
            self.t_mano = time.time()
            x, y = muestra.indice_mm
            if self.filtro.t is not None and t - self.filtro.t > self.c("gracia_mano_ms") / 1000.0:
                self.filtro.reiniciar()                  # la mano volvió: no arrastrar la posición vieja
            if self.c("suavizado"):
                x, y = self.filtro(t, x, y)
            self.posicion = (x, y)
            crudo = self._buscar(x, y, candidatos, self.estable, self._serie_seguida(t))
            self._estabilizar(crudo, t)
            e = self.estable
            if e is not None and es_dato(e) and e.get("serie"):
                self._serie, self._t_serie = e["serie"], t
        elif not self.pinza:
            # Mano perdida más allá de la gracia: se olvida el elemento (la serie caduca sola).
            self.estable, self._pendiente, self.posicion = None, None, None
            self.filtro.reiniciar()
        self.audio_activado = self.pinza
        self._narrar(muestra if hay else None, t)
        ident = self.estable["id"] if self.estable is not None else None
        return Seleccion(self.pinza, ident, candidatos)

    # ---------- narración ----------
    def _decir(self, texto, interrumpir):
        self.voz.decir(texto, interrumpir=interrumpir, esencial=True)

    def _narrar(self, m, t):
        if not self.pinza:
            if self.pinza_antes and self.c("callar_al_soltar"):
                self.voz.detener()
            self.anterior = None
            self._cruces_dichos.clear()
            self._t_fuera = None
            self.pinza_antes = False
            return
        e = self.estable
        if e is not None and e["id"] != self.anterior:
            interrumpir = bool(self.c("interrumpir_narracion"))
            # Sin interrumpir (como la versión anterior) se espera a que termine la frase.
            if interrumpir or not self.voz.hablando.is_set():
                self._decir(e["texto"], interrumpir)
                self.anterior = e["id"]
        if e is not None and self.c("anunciar_cruces"):
            self._narrar_cruce(e)
        if m is not None and self.c("avisar_fuera_placa"):
            self._narrar_fuera(m, t)
        self.pinza_antes = True

    def _narrar_cruce(self, e):
        """Una vez por cruce (hasta soltar la pinza): 'Cruce con <la otra serie>'."""
        if e["tipo"] not in TIPOS_LINEA or self.posicion is None or not self.cruces:
            return
        x, y = self.posicion
        lim = self._mm("cruce_mm")
        for cx, cy, a, b in self.cruces:
            if e["id"] not in (a, b) or distancia(x, y, cx, cy) > lim:
                continue
            clave = frozenset((a, b))
            if clave in self._cruces_dichos:
                continue
            self._cruces_dichos.add(clave)
            otra = self.por_id.get(b if e["id"] == a else a, {}).get("serie", "")
            self._decir(f"Cruce con {otra}." if otra else "Cruce con otra curva.", interrumpir=False)
            return

    def _narrar_fuera(self, m, t):
        if m.dentro_placa:
            self._t_fuera = None
            return
        if self._t_fuera is None:
            self._t_fuera = t
        if ((t - self._t_fuera) * 1000 >= self.c("fuera_placa_ms")
                and t - self._t_aviso_fuera >= float(self.c("fuera_placa_cada_s"))):
            self._t_aviso_fuera = t
            self._decir("Fuera de la placa.", interrumpir=False)
