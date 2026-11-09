#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRUEBAS DE LA INTERACCIÓN (Persona B)  ·  sin cámara ni voz real
================================================================

Comprueba interaccion.py con secuencias sintéticas de vision.MuestraMano:
contrato con vision.py, pinza (palma, histéresis, gracia), estabilidad, cruces,
prioridad de los ejes, narración y compatibilidad con el formato clásico.

    python probar_interaccion.py          (sale con código 1 si alguna falla)

No prueba la cámara, MediaPipe ni el sonido: eso solo se puede medir con el
montaje (medir_metricas.py camara ... --pruebas objetivos,falsos).
"""

import math
import os
import sys
import traceback

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import explorador_graficas_gui_qr as eg   # noqa: E402
import interaccion as ia                  # noqa: E402
from vision import MuestraMano, Registro  # noqa: E402

FPS = 15.0
PALMA = 90.0                               # mm: mano adulta vista sobre la placa


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
class VozFalsa:
    """Registra lo que se diría, sin sonido."""

    class _Evento:
        def __init__(self):
            self.valor = False

        def is_set(self):
            return self.valor

    def __init__(self):
        self.dichos = []          # (texto, interrumpir)
        self.detenciones = 0
        self.hablando = self._Evento()

    def decir(self, texto, interrumpir=False, esencial=False):
        self.dichos.append((texto, interrumpir))

    def detener(self):
        self.detenciones += 1


def cargar(nombre):
    return eg.cargar_desde_enlace(os.path.join(AQUI, "datos_graficas", nombre))


def nueva(datos, voz=None, **cfg):
    base = eg.cargar_config()
    base.update(cfg)
    ex = ia.Interaccion(voz or VozFalsa(), base)
    ex.configurar(datos, Registro.desde_rect((0, 0, 330, 330), datos.ancho_ref, datos.alto_ref, datos.estirar))
    return ex


def muestra(t, x, y, pinza=True, palma=PALMA, dentro=True):
    """Yema del índice en (x, y) mm; el medio a 3 mm (pinza) o a 0,6 palmas (abierta)."""
    sep = 3.0 if pinza else 0.6 * palma
    return MuestraMano(t, (x, y), (x + sep, y), palma, dentro)


def sin_mano(t):
    return MuestraMano(t, None, None, None, False)


def por_id(datos, ident):
    return next(e for e in datos.elementos if e["id"] == ident)


def centro(e):
    return (e["x1"] + e["x2"]) / 2, (e["y1"] + e["y2"]) / 2


def quieto(ex, x, y, segundos, t0=0.0, **kw):
    """Mantiene la yema quieta; devuelve la última Seleccion y el tiempo final."""
    s, n = None, int(segundos * FPS)
    for k in range(n):
        s = ex.actualizar(muestra(t0 + k / FPS, x, y, **kw))
    return s, t0 + n / FPS


PRUEBAS = []


def prueba(f):
    PRUEBAS.append(f)
    return f


# ------------------------------------------------------------
# Contrato con vision.py (Persona A)
# ------------------------------------------------------------
@prueba
def contrato_seleccion():
    d = cargar("ejemplo_temperatura.csv")
    ex = nueva(d)
    p = por_id(d, "s1_p3")
    s, _ = quieto(ex, p["x1"], p["y1"], 0.5)
    assert isinstance(s, eg.Seleccion) and s is not None
    assert s.pinza is True and s.elemento_id == "s1_p3", s
    assert "s1_p3" in s.candidatos
    # Sin mano: la Seleccion existe y la pinza acaba soltándose.
    s, _ = None, None
    for k in range(10):
        s = ex.actualizar(sin_mano(1.0 + k / FPS))
    assert s.pinza is False
    # Sin registro ni datos no falla.
    vacio = ia.Interaccion(None, {})
    assert vacio.actualizar(muestra(0, 1, 1)).elemento_id is None
    assert vacio.actualizar(None).pinza is False


@prueba
def configurar_con_rect_o_registro():
    d = cargar("ejemplo_temperatura.csv")
    a = ia.Interaccion(None, eg.cargar_config())
    a.configurar(d, (155, 2, 319, 337))                     # el antiguo rectángulo 'hoja'
    b = ia.Interaccion(None, eg.cargar_config())
    b.configurar(d, Registro.desde_rect((155, 2, 319, 337), d.ancho_ref, d.alto_ref, d.estirar))
    p = por_id(d, "s1_p5")
    assert a._buscar(p["x1"], p["y1"])["id"] == b._buscar(p["x1"], p["y1"])["id"] == "s1_p5"
    # Trabaja en mm: el resultado no depende de la calibración en px.
    c = ia.Interaccion(None, eg.cargar_config())
    c.configurar(d, (0, 0, 900, 900))
    assert c._buscar(p["x1"] + 2, p["y1"] + 1)["id"] == a._buscar(p["x1"] + 2, p["y1"] + 1)["id"]


@prueba
def la_aplicacion_usa_interaccion():
    falso = type("App", (), {"voz": VozFalsa(), "cfg": eg.cargar_config()})()
    obj = eg.App._crear_interaccion(falso, None, None)
    assert isinstance(obj, ia.Interaccion)
    assert hasattr(obj, "configurar") and hasattr(obj, "actualizar")
    assert issubclass(eg.Explorador, ia.Interaccion)         # medir_metricas (tareas) sigue funcionando


@prueba
def motor_vision_recibe_la_seleccion():
    """Integración con vision.MotorVision (sin cámara): mano falsa → MuestraMano → Interaccion
    → Seleccion → superposición y evento 'elemento' para la interfaz."""
    import queue
    import numpy as np
    from vision import MotorVision

    d = cargar("ejemplo_temperatura.csv")
    cfg = eg.cargar_config()
    cfg.update({"autodetectar_al_empezar": False, "seguir_placa": False, "depurar_vision": False,
                "esquinas_placa_px": None, "hoja": [155, 2, 319, 337]})
    voz = VozFalsa()
    cola = queue.Queue()
    motor = MotorVision(cola, voz, cfg, crear_interaccion=lambda det, avisar: ia.Interaccion(voz, cfg, avisar))
    reg = eg.registro_para(d, None, cfg["hoja"], "rect")
    p = por_id(d, "s1_p5")
    w, h = 640, 480

    class Punto:
        def __init__(self, x_mm, y_mm):
            u, v = reg.placa_a_imagen_f(x_mm, y_mm)
            self.x, self.y, self.z = u / w, v / h, 0.0

    class DetectorFalso:
        ms_ultimo = 5.0

        def detectar(self, frame, roi=None):
            pts = [Punto(p["x1"], p["y1"] + 60) for _ in range(21)]
            pts[0] = Punto(p["x1"], p["y1"] + 90)            # muñeca
            pts[9] = Punto(p["x1"], p["y1"])                 # base del medio: palma de 90 mm
            pts[8] = Punto(p["x1"], p["y1"])                 # yema del índice, sobre el punto
            pts[12] = Punto(p["x1"] + 4, p["y1"])            # medio junto al índice: pinza
            return [pts]

        def cerrar(self):
            pass

    motor.detector = DetectorFalso()
    motor.set_modo("explorar")
    motor.set_hoja(cfg["hoja"])
    motor.set_datos(d)
    import time
    for _ in range(8):                                     # al ritmo de la cámara (15 fps)
        frame = motor._paso_explorar(np.zeros((h, w, 3), np.uint8))
        time.sleep(1 / FPS)
    assert frame.shape == (h, w, 3)
    assert isinstance(motor.interaccion, ia.Interaccion)
    assert motor.interaccion.estable is not None and motor.interaccion.estable["id"] == "s1_p5"
    assert voz.dichos and voz.dichos[0][0] == p["texto"], voz.dichos
    eventos = []
    while not cola.empty():
        eventos.append(cola.get())
    assert ("elemento", p["corto"]) in eventos, eventos


# ------------------------------------------------------------
# Pinza
# ------------------------------------------------------------
@prueba
def pinza_normalizada_por_palma():
    d = cargar("ejemplo_temperatura.csv")
    for palma in (60.0, 90.0, 140.0):                      # mano cerca / normal / lejos de la cámara
        ex = nueva(d)
        juntos = MuestraMano(0.0, (100, 50), (100 + 0.2 * palma, 50), palma, True)
        assert ex.actualizar(juntos).pinza, palma
        ex = nueva(d)
        separados = MuestraMano(0.0, (100, 50), (100 + 0.5 * palma, 50), palma, True)
        assert not ex.actualizar(separados).pinza, palma


@prueba
def pinza_con_histeresis():
    d = cargar("ejemplo_temperatura.csv")
    ex = nueva(d)
    entre = 0.32 * PALMA                                   # entre cerrar (0,28) y abrir (0,36)
    assert not ex.actualizar(MuestraMano(0.0, (100, 50), (100 + entre, 50), PALMA, True)).pinza
    assert ex.actualizar(MuestraMano(0.1, (100, 50), (100 + 0.2 * PALMA, 50), PALMA, True)).pinza
    for k in range(30):                                    # 2 s en la zona intermedia: sigue cerrada
        s = ex.actualizar(MuestraMano(0.2 + k / FPS, (100, 50), (100 + entre, 50), PALMA, True))
        assert s.pinza


@prueba
def gracia_de_la_pinza():
    d = cargar("ejemplo_temperatura.csv")
    voz = VozFalsa()
    ex = nueva(d, voz)
    p = por_id(d, "s1_p4")
    _s, t = quieto(ex, p["x1"], p["y1"], 0.5)
    # Dos frames (133 ms) con los dedos separados: no se corta.
    for k in range(2):
        assert ex.actualizar(muestra(t + k / FPS, p["x1"], p["y1"], pinza=False)).pinza
    t += 2 / FPS
    _s, t = quieto(ex, p["x1"], p["y1"], 0.3, t)
    assert voz.detenciones == 0
    anuncios = len(voz.dichos)
    assert anuncios == 1, voz.dichos                       # y no se repite el anuncio
    # Separados de verdad (0,5 s): se suelta y se calla una sola vez.
    for k in range(8):
        s = ex.actualizar(muestra(t + k / FPS, p["x1"], p["y1"], pinza=False))
    assert not s.pinza and voz.detenciones == 1


@prueba
def gracia_sin_mano():
    d = cargar("ejemplo_temperatura.csv")
    voz = VozFalsa()
    ex = nueva(d, voz)
    p = por_id(d, "s1_p6")
    _s, t = quieto(ex, p["x1"], p["y1"], 0.5)
    for k in range(3):                                     # MediaPipe pierde la mano 200 ms
        s = ex.actualizar(sin_mano(t + k / FPS))
        assert s.pinza and s.elemento_id == "s1_p6"
    t += 3 / FPS
    _s, t = quieto(ex, p["x1"], p["y1"], 0.3, t)
    assert voz.detenciones == 0 and len(voz.dichos) == 1
    for k in range(10):                                    # la pierde 0,7 s: se suelta
        s = ex.actualizar(sin_mano(t + k / FPS))
    assert not s.pinza and s.elemento_id is None and voz.detenciones == 1


@prueba
def gracia_comparada_con_la_pinza_antigua():
    """Pinza con un frame de cada 5 abierto (MediaPipe que tiembla): antes, 1 corte cada vez."""
    d = cargar("ejemplo_temperatura.csv")
    ex = nueva(d)
    p = por_id(d, "s1_p6")
    cortes = cortes_antiguos = 0
    antes = antes_antiguo = False
    for k in range(150):                                   # 10 s
        abierta = k % 5 == 4
        s = ex.actualizar(muestra(k / FPS, p["x1"], p["y1"], pinza=not abierta))
        cortes += antes and not s.pinza
        antiguo = not abierta
        cortes_antiguos += antes_antiguo and not antiguo
        antes, antes_antiguo = s.pinza, antiguo
    assert cortes == 0 and cortes_antiguos >= 25, (cortes, cortes_antiguos)


# ------------------------------------------------------------
# Estabilidad
# ------------------------------------------------------------
@prueba
def estabilidad_en_la_frontera():
    """Yema temblando justo en la frontera entre dos elementos: la selección no parpadea."""
    import random
    d = cargar("ejemplo_temperatura.csv")
    ex = nueva(d)
    a, b = por_id(d, "s1_p6"), por_id(d, "s1_p7")
    # Sobre la curva, a 'ventaja_punto_mm' (3 mm) del punto: ahí cambia de punto a curva.
    largo = math.hypot(b["x1"] - a["x1"], b["y1"] - a["y1"])
    paso = ex.c("ventaja_punto_mm") / largo
    mx, my = a["x1"] + paso * (b["x1"] - a["x1"]), a["y1"] + paso * (b["y1"] - a["y1"])
    rnd = random.Random(1)
    sel, crudo = [], []
    for k in range(150):
        x, y = mx + rnd.gauss(0, 1.5), my + rnd.gauss(0, 1.5)
        sel.append(ex.actualizar(muestra(k / FPS, x, y)).elemento_id)
        e = ex._buscar(x, y)
        crudo.append(e["id"] if e else None)
    cambios = sum(p != q for p, q in zip(sel, sel[1:]))
    cambios_crudo = sum(p != q for p, q in zip(crudo, crudo[1:]))
    assert cambios_crudo >= 20 and cambios * 4 <= cambios_crudo, (cambios, cambios_crudo)


@prueba
def cambio_real_de_elemento_llega_rapido():
    d = cargar("ejemplo_temperatura.csv")
    ex = nueva(d)
    a, b = por_id(d, "s1_p3"), por_id(d, "s1_p8")
    _s, t = quieto(ex, a["x1"], a["y1"], 0.5)
    for k in range(int(0.4 * FPS)):                        # salta al otro punto y se queda
        s = ex.actualizar(muestra(t + k / FPS, b["x1"], b["y1"]))
    assert s.elemento_id == "s1_p8", s.elemento_id


# ------------------------------------------------------------
# Cruces
# ------------------------------------------------------------
def _cruce(d):
    ex = nueva(d)
    assert ex.cruces, "la gráfica 2 tiene cruces entre series"
    return ex, ex.cruces[0]


@prueba
def cruce_conserva_la_serie():
    d = cargar("73405ebe8e0e.csv")
    ex, (cx, cy, ida, idb) = _cruce(d)
    a, b = por_id(d, ida), por_id(d, idb)
    voz = VozFalsa()
    ex = nueva(d, voz)
    # Recorre la curva 'a' de punta a punta, pasando por el cruce con 'b'.
    n = 40
    series = []
    for k in range(n + 1):
        f = k / n
        x, y = a["x1"] + f * (a["x2"] - a["x1"]), a["y1"] + f * (a["y2"] - a["y1"])
        s = ex.actualizar(muestra(k / FPS, x, y))
        e = ex.por_id.get(s.elemento_id)
        if e is not None and e["tipo"] in ia.TIPOS_DATOS:
            series.append(e["serie"])
    assert series and all(s == a["serie"] for s in series), (a["serie"], set(series))
    avisos = [t for t, _i in voz.dichos if t.startswith("Cruce con")]
    assert avisos == [f"Cruce con {b['serie']}."], avisos


@prueba
def cruce_sin_estado_confunde_series():
    """Control: sin la serie seguida, el mismo recorrido salta a la otra serie cerca del cruce."""
    d = cargar("73405ebe8e0e.csv")
    ex, (cx, cy, ida, idb) = _cruce(d)
    a = por_id(d, ida)
    otras = 0
    import random
    rnd = random.Random(2)
    for k in range(200):
        f = 0.5 + rnd.uniform(-0.15, 0.15)
        x = a["x1"] + f * (a["x2"] - a["x1"]) + rnd.gauss(0, 1.5)
        y = a["y1"] + f * (a["y2"] - a["y1"]) + rnd.gauss(0, 1.5)
        if math.hypot(x - cx, y - cy) > 4:
            continue
        e = ex._buscar(x, y)
        otras += e is not None and e["tipo"] in ia.TIPOS_DATOS and e["serie"] != a["serie"]
    assert otras > 0


# ------------------------------------------------------------
# Prioridad de los ejes
# ------------------------------------------------------------
@prueba
def eje_no_queda_tapado_por_los_datos():
    d = cargar("ejemplo_temperatura.csv")
    ex = nueva(d)
    eje_y = por_id(d, "eje_y")
    p1 = por_id(d, "s1_p1")                                # enero: 8 mm a la derecha del eje Y
    y = p1["y1"]
    e = ex._buscar(eje_y["x1"], y)
    assert e["id"] == "eje_y", e["id"]
    estricta = nueva(d, ventaja_punto_mm=None, margen_resto_mm=999)  # como antes: el dato gana siempre
    assert estricta._buscar(eje_y["x1"] + 3.5, y)["tipo"] != "eje"
    # Y el dato sigue ganando encima de él.
    assert ex._buscar(p1["x1"], p1["y1"])["id"] == "s1_p1"
    eje_x = por_id(d, "eje_x")
    xm, ym = centro(eje_x)
    assert ex._buscar(xm, ym)["id"] == "eje_x"


@prueba
def punto_gana_a_su_curva():
    d = cargar("ejemplo_temperatura.csv")
    ex = nueva(d)
    for ident in ("s1_p2", "s1_p7", "s1_p11"):
        p = por_id(d, ident)
        assert ex._buscar(p["x1"] + 0.5, p["y1"] + 0.5)["id"] == ident


# ------------------------------------------------------------
# Narración
# ------------------------------------------------------------
@prueba
def narracion_interrumpe_y_no_repite():
    d = cargar("ejemplo_temperatura.csv")
    voz = VozFalsa()
    ex = nueva(d, voz)
    a, b = por_id(d, "s1_p3"), por_id(d, "s1_p9")
    _s, t = quieto(ex, a["x1"], a["y1"], 0.6)
    _s, t = quieto(ex, b["x1"], b["y1"], 0.6, t)
    _s, t = quieto(ex, b["x1"], b["y1"], 0.6, t)           # se queda: no repite
    textos = [x for x, _i in voz.dichos]
    assert textos == [a["texto"], b["texto"]], textos
    assert all(i for _x, i in voz.dichos)                  # un elemento nuevo corta el anterior
    # Soltar y volver a pinzar sobre el mismo elemento: se repite (como antes).
    for k in range(8):
        ex.actualizar(muestra(t + k / FPS, b["x1"], b["y1"], pinza=False))
    t += 8 / FPS
    _s, t = quieto(ex, b["x1"], b["y1"], 0.4, t)
    assert [x for x, _i in voz.dichos][-1] == b["texto"] and len(voz.dichos) == 3


@prueba
def narracion_sin_pinza_calla():
    d = cargar("ejemplo_temperatura.csv")
    voz = VozFalsa()
    ex = nueva(d, voz)
    p = por_id(d, "s1_p3")
    s, _t = quieto(ex, p["x1"], p["y1"], 1.0, pinza=False)
    assert voz.dichos == [] and s.elemento_id == "s1_p3"    # se resalta, pero no se lee


@prueba
def narracion_sin_interrumpir_espera():
    d = cargar("ejemplo_temperatura.csv")
    voz = VozFalsa()
    voz.hablando.valor = True                              # está leyendo la descripción
    ex = nueva(d, voz, interrumpir_narracion=False)
    p = por_id(d, "s1_p3")
    _s, t = quieto(ex, p["x1"], p["y1"], 0.5)
    assert voz.dichos == []
    voz.hablando.valor = False
    _s, t = quieto(ex, p["x1"], p["y1"], 0.2, t)
    assert voz.dichos == [(p["texto"], False)]


@prueba
def aviso_fuera_de_la_placa():
    d = cargar("ejemplo_temperatura.csv")
    voz = VozFalsa()
    ex = nueva(d, voz)
    _s, _t = quieto(ex, -10, 50, 2.0, dentro=False)
    avisos = [x for x, _i in voz.dichos if x == "Fuera de la placa."]
    assert len(avisos) == 1, voz.dichos


# ------------------------------------------------------------
# Formato clásico (lienzo sin mm)
# ------------------------------------------------------------
@prueba
def formato_clasico_sigue_funcionando():
    texto = ("# titulo: Prueba\netiqueta,x,y,valor\nA,50,150,1\nB,150,100,2\nC,250,50,3\n")
    d = eg.interpretar_csv(texto)
    assert d.formato == "clasico"
    ex = ia.Interaccion(VozFalsa(), eg.cargar_config())
    ex.configurar(d, (100, 50, 400, 300))
    b = next(e for e in d.elementos if e["tipo"] == "punto" and e["corto"].startswith("B"))
    s, _t = quieto(ex, b["x1"] + 1, b["y1"] + 1, 0.5)
    assert s.elemento_id == b["id"] and s.pinza


# ------------------------------------------------------------
def main():
    fallos = 0
    for f in PRUEBAS:
        try:
            f()
            print(f"  ok     {f.__name__}")
        except Exception:
            fallos += 1
            print(f"  FALLA  {f.__name__}")
            traceback.print_exc()
    print(f"\n{len(PRUEBAS) - fallos} de {len(PRUEBAS)} pruebas correctas.")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
