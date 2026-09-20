"""Generador de placas táctiles STL para la Fase 1.

Este módulo reemplaza a la implementación anterior del generador STL y
convierte el Bloque 3 en la fuente de verdad del proyecto, manteniendo la
compatibilidad con la API y con el flujo de segmentación existente.
"""

import math
from math import atan2, degrees, hypot

import cadquery as cq

# =============================================================================
# CONSTANTES
# =============================================================================
BASE_THICKNESS = 2.0
ESPACIADO_BRAILLE = 2.4
DIAM_PUNTO_BRAILLE = 1.4
CELDA_PITCH = 6.2
CLEARANCE_BRAILLE = 9.5
RELIEVE_EJE = 1.0
RELIEVE_TICK = 1.0
RELIEVE_LINEA = 1.6
DIAM_LINEA = 2.0
MARGEN_BORDE = 25.0

LETRAS = {
    "a": [1], "b": [1, 2], "c": [1, 4], "d": [1, 4, 5], "e": [1, 5],
    "f": [1, 2, 4], "g": [1, 2, 4, 5], "h": [1, 2, 5], "i": [2, 4],
    "j": [2, 4, 5], "k": [1, 3], "l": [1, 2, 3], "m": [1, 3, 4],
    "n": [1, 3, 4, 5], "o": [1, 3, 5], "p": [1, 2, 3, 4],
    "q": [1, 2, 3, 4, 5], "r": [1, 2, 3, 5], "s": [2, 3, 4],
    "t": [2, 3, 4, 5], "u": [1, 3, 6], "v": [1, 2, 3, 6],
    "w": [2, 4, 5, 6], "x": [1, 3, 4, 6], "y": [1, 3, 4, 5, 6],
    "z": [1, 3, 5, 6],
}
_DESPLAZAMIENTO_NEMETH = {1: 2, 2: 3, 4: 5, 5: 6}
_ORDEN_DIGITOS = list("abcdefghij")

BRAILLE = dict(LETRAS)
BRAILLE["numeral"] = [3, 4, 5, 6]
BRAILLE["menos"] = [3, 6]
for _n, _letra in enumerate(_ORDEN_DIGITOS, start=1):
    _digito = str(_n % 10)
    BRAILLE[_digito] = sorted(_DESPLAZAMIENTO_NEMETH[d] for d in LETRAS[_letra])


def agregar_punto_braille(modelo, cx, cy):
    """Genera un punto Braille en relieve sobre la placa."""
    punto = (
        cq.Workplane("XY")
        .workplane(offset=BASE_THICKNESS)
        .center(cx, cy)
        .sphere(DIAM_PUNTO_BRAILLE / 2)
    )
    return modelo.union(punto)


def agregar_caracter_braille(modelo, caracter, cx, cy):
    posiciones = {
        1: (-ESPACIADO_BRAILLE / 2, ESPACIADO_BRAILLE),
        2: (-ESPACIADO_BRAILLE / 2, 0),
        3: (-ESPACIADO_BRAILLE / 2, -ESPACIADO_BRAILLE),
        4: (ESPACIADO_BRAILLE / 2, ESPACIADO_BRAILLE),
        5: (ESPACIADO_BRAILLE / 2, 0),
        6: (ESPACIADO_BRAILLE / 2, -ESPACIADO_BRAILLE),
    }
    for p in BRAILLE.get(str(caracter).lower(), []):
        dx, dy = posiciones[p]
        modelo = agregar_punto_braille(modelo, cx + dx, cy + dy)
    return modelo


def agregar_texto_braille(modelo, texto, cx, cy):
    """Coloca una cadena de caracteres Braille en línea."""
    x = cx
    for ch in texto:
        modelo = agregar_caracter_braille(modelo, ch, x, cy)
        x += CELDA_PITCH
    return modelo


def agregar_numero_braille(modelo, valor, cx, cy):
    """Coloca un número entero en formato Nemeth."""
    x = cx
    if valor < 0:
        modelo = agregar_caracter_braille(modelo, "menos", x, cy)
        x += CELDA_PITCH
    modelo = agregar_caracter_braille(modelo, "numeral", x, cy)
    x += CELDA_PITCH
    for ch in str(abs(valor)):
        modelo = agregar_caracter_braille(modelo, ch, x, cy)
        x += CELDA_PITCH
    return modelo


def agregar_segmento_relieve(modelo, p0, p1, diametro, altura):
    """Crea un tramo en relieve con dos cilindros en los extremos."""
    x0, y0 = p0
    x1, y1 = p1
    largo = hypot(x1 - x0, y1 - y0)
    if largo < 1e-6:
        return modelo
    angulo = degrees(atan2(y1 - y0, x1 - x0))
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2

    cuerpo = (
        cq.Workplane("XY")
        .workplane(offset=BASE_THICKNESS)
        .center(mx, my)
        .transformed(rotate=(0, 0, angulo))
        .box(largo, diametro, altura, centered=(True, True, False))
    )
    tapa0 = (
        cq.Workplane("XY").workplane(offset=BASE_THICKNESS)
        .center(x0, y0).circle(diametro / 2).extrude(altura)
    )
    tapa1 = (
        cq.Workplane("XY").workplane(offset=BASE_THICKNESS)
        .center(x1, y1).circle(diametro / 2).extrude(altura)
    )
    return modelo.union(cuerpo).union(tapa0).union(tapa1)


def agregar_funcion(modelo, puntos, diametro=DIAM_LINEA, altura=RELIEVE_LINEA):
    """Dibuja una polilínea a partir de una secuencia de puntos."""
    for p0, p1 in zip(puntos[:-1], puntos[1:]):
        modelo = agregar_segmento_relieve(modelo, p0, p1, diametro, altura)
    return modelo


def _valores_tick(v_min, v_max, intervalo):
    """Devuelve los múltiplos de un intervalo dentro del rango."""
    inicio = math.ceil(v_min / intervalo) * intervalo
    valores = []
    v = inicio
    while v <= v_max:
        valores.append(int(v))
        v += intervalo
    return valores


def generar_modelo_bana(
    dim_x=210.0,
    dim_y=148.0,
    p1=(0, 0),
    p2=(150, 90),
    intervalo_ticks=25,
    archivo_salida="grafica_bana.stl",
):
    """Genera una placa táctil BANA con ejes, marcas y una línea en relieve."""
    xs = [0, p1[0], p2[0]]
    ys = [0, p1[1], p2[1]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    ancho_datos = x_max - x_min
    alto_datos = y_max - y_min
    if ancho_datos > dim_x - 2 * MARGEN_BORDE or alto_datos > dim_y - 2 * MARGEN_BORDE:
        raise ValueError(
            f"Los datos (ancho={ancho_datos}mm, alto={alto_datos}mm) no entran en la "
            f"placa de {dim_x}x{dim_y}mm con un margen de {MARGEN_BORDE}mm por lado."
        )

    origen_x_fis = MARGEN_BORDE - x_min
    origen_y_fis = MARGEN_BORDE - y_min

    def a_fisico(punto):
        return (punto[0] + origen_x_fis, punto[1] + origen_y_fis)

    modelo = (
        cq.Workplane("XY")
        .center(dim_x / 2, dim_y / 2)
        .box(dim_x, dim_y, BASE_THICKNESS, centered=(True, True, False))
    )

    grosor_eje = 2.0
    z_ejes = BASE_THICKNESS + RELIEVE_EJE
    z_ticks = BASE_THICKNESS + RELIEVE_TICK

    x0_fis, x1_fis = a_fisico((x_min, 0))[0], a_fisico((x_max, 0))[0]
    y0_fis, y1_fis = a_fisico((0, y_min))[1], a_fisico((0, y_max))[1]

    eje_x = (
        cq.Workplane("XY").center((x0_fis + x1_fis) / 2, origen_y_fis)
        .box(x1_fis - x0_fis, grosor_eje, z_ejes, centered=(True, True, False))
    )
    eje_y = (
        cq.Workplane("XY").center(origen_x_fis, (y0_fis + y1_fis) / 2)
        .box(grosor_eje, y1_fis - y0_fis, z_ejes, centered=(True, True, False))
    )
    modelo = modelo.union(eje_x).union(eje_y)

    longitud_tick = 6.0
    for valor in _valores_tick(x_min, x_max, intervalo_ticks):
        x_fis = origen_x_fis + valor
        if valor != 0:
            tick = (
                cq.Workplane("XY").center(x_fis, origen_y_fis)
                .box(grosor_eje, longitud_tick, z_ticks, centered=(True, True, False))
            )
            modelo = modelo.union(tick)
        modelo = agregar_numero_braille(
            modelo, valor, cx=x_fis, cy=origen_y_fis - CLEARANCE_BRAILLE
        )

    for valor in _valores_tick(y_min, y_max, intervalo_ticks):
        if valor == 0:
            continue
        y_fis = origen_y_fis + valor
        tick = (
            cq.Workplane("XY").center(origen_x_fis, y_fis)
            .box(longitud_tick, grosor_eje, z_ticks, centered=(True, True, False))
        )
        modelo = modelo.union(tick)
        ancho_estimado = CELDA_PITCH * (len(str(abs(valor))) + 2)
        modelo = agregar_numero_braille(
            modelo, valor, cx=origen_x_fis - CLEARANCE_BRAILLE - ancho_estimado, cy=y_fis
        )

    modelo = agregar_texto_braille(
        modelo, "x", cx=x1_fis - CELDA_PITCH,
        cy=origen_y_fis - CLEARANCE_BRAILLE - CELDA_PITCH * 2,
    )
    modelo = agregar_texto_braille(
        modelo, "y", cx=origen_x_fis - CLEARANCE_BRAILLE - CELDA_PITCH * 3,
        cy=y1_fis - CELDA_PITCH,
    )

    modelo = agregar_funcion(modelo, [a_fisico(p1), a_fisico(p2)])

    cq.exporters.export(modelo, archivo_salida)
    print(f"Modelo táctil BANA ({dim_x}x{dim_y}mm) exportado a: {archivo_salida}")
    return modelo


# Compatibilidad con el código previo del proyecto.
def _punto_braille(modelo, cx, cy):
    return agregar_punto_braille(modelo, cx, cy)


def _caracter_braille(modelo, caracter, cx, cy):
    return agregar_caracter_braille(modelo, caracter, cx, cy)


def _numero_braille(modelo, valor, cx, cy):
    return agregar_numero_braille(modelo, valor, cx, cy)


def _segmento(modelo, inicio, fin, diametro, altura):
    return agregar_segmento_relieve(modelo, inicio, fin, diametro, altura)


def _ticks(minimo, maximo, intervalo):
    return _valores_tick(minimo, maximo, intervalo)


def _ancho_numero(valor):
    return CELDA_PITCH * (len(str(abs(int(round(valor))))) + 1)


def generar_modelo_desde_recta(puntos, dim_x=210.0, dim_y=148.0, archivo_salida="grafica_tactil.stl"):
    """Genera una placa STL a partir de puntos de una recta detectada."""
    if dim_x < 130 or dim_y < 90:
        raise ValueError("La placa debe medir al menos 130 x 90 mm.")
    if not isinstance(puntos, (list, tuple)) or len(puntos) < 2:
        raise ValueError("Se necesitan al menos dos puntos para generar un STL.")

    calibrados = all(p.get("valor_x") is not None and p.get("valor_y") is not None for p in puntos)
    if calibrados:
        datos = [
            (float(puntos[0]["valor_x"]), float(puntos[0]["valor_y"])),
            (float(puntos[-1]["valor_x"]), float(puntos[-1]["valor_y"])),
        ]
    else:
        datos = [
            (float(puntos[0]["px"]), -float(puntos[0]["py"])),
            (float(puntos[-1]["px"]), -float(puntos[-1]["py"])),
        ]

    xs, ys = zip(*datos)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min or y_max == y_min:
        raise ValueError("La curva debe tener variación tanto en X como en Y.")

    izquierda, derecha, abajo, arriba = 55.0, 15.0, 25.0, 15.0
    ancho_plot, alto_plot = dim_x - izquierda - derecha, dim_y - abajo - arriba

    def escalar(x, y):
        return (
            izquierda + (x - x_min) / (x_max - x_min) * ancho_plot,
            abajo + (y - y_min) / (y_max - y_min) * alto_plot,
        )

    modelo = cq.Workplane("XY").center(dim_x / 2, dim_y / 2).box(
        dim_x, dim_y, BASE_THICKNESS, centered=(True, True, False)
    )
    modelo = agregar_segmento_relieve(modelo, (izquierda, abajo), (dim_x - derecha, abajo), 2.0, RELIEVE_EJE)
    modelo = agregar_segmento_relieve(modelo, (izquierda, abajo), (izquierda, dim_y - arriba), 2.0, RELIEVE_EJE)

    for i in range(5):
        fraccion = i / 4
        x_fis = izquierda + fraccion * ancho_plot
        y_fis = abajo + fraccion * alto_plot
        modelo = agregar_segmento_relieve(modelo, (x_fis, abajo - 3), (x_fis, abajo + 3), 1.5, RELIEVE_TICK)
        modelo = agregar_segmento_relieve(modelo, (izquierda - 3, y_fis), (izquierda + 3, y_fis), 1.5, RELIEVE_TICK)
        if calibrados:
            x_valor = x_min + fraccion * (x_max - x_min)
            y_valor = y_min + fraccion * (y_max - y_min)
            ancho_x = _ancho_numero(x_valor)
            inicio_x = min(max(3.0, x_fis - ancho_x / 2), dim_x - ancho_x - 3.0)
            modelo = agregar_numero_braille(modelo, round(x_valor), inicio_x, abajo - CLEARANCE_BRAILLE)
            ancho_y = _ancho_numero(y_valor)
            inicio_y = max(3.0, izquierda - CLEARANCE_BRAILLE - ancho_y)
            modelo = agregar_numero_braille(modelo, round(y_valor), inicio_y, y_fis)

    inicio, fin = (escalar(*datos[0]), escalar(*datos[1]))
    modelo = agregar_segmento_relieve(modelo, inicio, fin, DIAM_LINEA, RELIEVE_LINEA)

    cq.exporters.export(modelo, archivo_salida)
    return modelo


if __name__ == "__main__":
    generar_modelo_bana()
