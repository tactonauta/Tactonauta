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


def simplificar_polilinea(puntos, tolerancia=1.0, max_puntos=80):
    """
    Simplifica una polilínea conservando su forma aproximada.

    Primero utiliza una simplificación tipo Ramer-Douglas-Peucker.
    Después, si todavía quedan demasiados puntos, aplica un muestreo
    uniforme para mantener el coste geométrico bajo.

    tolerancia:
        Distancia máxima aproximada que puede separarse la curva
        simplificada de la original, en mm físicos.

    max_puntos:
        Número máximo de puntos que se utilizarán para fabricar el STL.
    """

    if len(puntos) <= 2:
        return puntos[:]

    def distancia_punto_segmento(p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b

        dx = bx - ax
        dy = by - ay

        if dx == 0 and dy == 0:
            return hypot(px - ax, py - ay)

        t = (
            (px - ax) * dx +
            (py - ay) * dy
        ) / (dx * dx + dy * dy)

        t = max(0.0, min(1.0, t))

        cx = ax + t * dx
        cy = ay + t * dy

        return hypot(px - cx, py - cy)

    def rdp(lista, eps):
        if len(lista) <= 2:
            return lista[:]

        inicio = lista[0]
        fin = lista[-1]

        max_distancia = -1.0
        indice = -1

        for i in range(1, len(lista) - 1):
            d = distancia_punto_segmento(
                lista[i],
                inicio,
                fin
            )

            if d > max_distancia:
                max_distancia = d
                indice = i

        if max_distancia > eps:
            izquierda = rdp(
                lista[:indice + 1],
                eps
            )

            derecha = rdp(
                lista[indice:],
                eps
            )

            return izquierda[:-1] + derecha

        return [inicio, fin]

    simplificados = rdp(puntos, tolerancia)

    if len(simplificados) <= max_puntos:
        return simplificados

    # Si todavía hay demasiados puntos, conservarlos
    # distribuidos uniformemente.
    resultado = []

    paso = (len(simplificados) - 1) / (max_puntos - 1)

    for i in range(max_puntos):
        indice = round(i * paso)
        resultado.append(simplificados[indice])

    return resultado


def agregar_funcion(
    modelo,
    puntos,
    diametro=DIAM_LINEA,
    altura=RELIEVE_LINEA
):
    """
    Dibuja una polilínea en relieve.

    Los puntos deben estar ya convertidos a coordenadas físicas.
    """

    if not puntos or len(puntos) < 2:
        return modelo

    for p0, p1 in zip(puntos[:-1], puntos[1:]):
        modelo = agregar_segmento_relieve(
            modelo,
            p0,
            p1,
            diametro,
            altura
        )

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


def generar_modelo_desde_recta(
    puntos,
    dim_x=210.0,
    dim_y=148.0,
    archivo_salida="grafica_tactil.stl"
):
    """
    Genera una placa táctil a partir de la curva detectada por
    segmentador.py.

    El segmentador puede entregar cientos de puntos. Estos se conservan
    durante el procesamiento, pero antes de construir la geometría STL
    se simplifican para evitar un número excesivo de operaciones booleanas
    de CadQuery.

    Los puntos pueden estar calibrados:

        {
            "px": ...,
            "py": ...,
            "valor_x": ...,
            "valor_y": ...
        }

    o pueden contener únicamente coordenadas de imagen:

        {
            "px": ...,
            "py": ...
        }
    """

    if dim_x < 130 or dim_y < 90:
        raise ValueError(
            "La placa debe medir al menos 130 x 90 mm."
        )

    if not isinstance(puntos, (list, tuple)) or len(puntos) < 2:
        raise ValueError(
            "Se necesitan al menos dos puntos para generar un STL."
        )

    # ================================================================
    # 1. DETERMINAR SI LOS PUNTOS ESTÁN CALIBRADOS
    # ================================================================

    calibrados = all(
        isinstance(p, dict)
        and p.get("valor_x") is not None
        and p.get("valor_y") is not None
        for p in puntos
    )

    datos = []

    if calibrados:

        for p in puntos:
            try:
                x = float(p["valor_x"])
                y = float(p["valor_y"])
            except (TypeError, ValueError, KeyError):
                continue

            if math.isfinite(x) and math.isfinite(y):
                datos.append((x, y))

    else:

        for p in puntos:
            try:
                x = float(p["px"])

                # Coordenadas de imagen:
                # Y crece hacia abajo.
                #
                # Las invertimos para convertirlas a coordenadas
                # cartesianas normales.
                y = -float(p["py"])

            except (TypeError, ValueError, KeyError):
                continue

            if math.isfinite(x) and math.isfinite(y):
                datos.append((x, y))

    if len(datos) < 2:
        raise ValueError(
            "No hay suficientes puntos válidos para generar la curva."
        )

    # ================================================================
    # 2. ELIMINAR DUPLICADOS CONSECUTIVOS
    # ================================================================

    datos_limpios = [datos[0]]

    for punto in datos[1:]:

        anterior = datos_limpios[-1]

        distancia = hypot(
            punto[0] - anterior[0],
            punto[1] - anterior[1]
        )

        if distancia > 1e-6:
            datos_limpios.append(punto)

    datos = datos_limpios

    if len(datos) < 2:
        raise ValueError(
            "La curva contiene menos de dos puntos distintos."
        )

    # ================================================================
    # 3. LÍMITES DE LA CURVA
    # ================================================================

    xs = [p[0] for p in datos]
    ys = [p[1] for p in datos]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    if x_max == x_min and y_max == y_min:
        raise ValueError(
            "Todos los puntos de la curva son iguales."
        )

    rango_x = x_max - x_min
    rango_y = y_max - y_min

    if rango_x == 0:
        rango_x = 1.0

    if rango_y == 0:
        rango_y = 1.0

    # ================================================================
    # 4. ÁREA DEL GRÁFICO
    # ================================================================

    izquierda = 55.0
    derecha = 15.0
    abajo = 25.0
    arriba = 15.0

    ancho_plot = dim_x - izquierda - derecha
    alto_plot = dim_y - abajo - arriba

    if ancho_plot <= 0 or alto_plot <= 0:
        raise ValueError(
            "Las dimensiones de la placa no dejan espacio suficiente "
            "para el gráfico."
        )

    # ================================================================
    # 5. CONVERSIÓN A COORDENADAS FÍSICAS
    # ================================================================

    def escalar(x, y):

        x_fis = (
            izquierda
            + (x - x_min)
            / rango_x
            * ancho_plot
        )

        y_fis = (
            abajo
            + (y - y_min)
            / rango_y
            * alto_plot
        )

        return (x_fis, y_fis)

    puntos_fisicos = [
        escalar(x, y)
        for x, y in datos
    ]

    # ================================================================
    # 6. SIMPLIFICAR LA CURVA PARA EL STL
    # ================================================================

    puntos_stl = simplificar_polilinea(
        puntos_fisicos,
        tolerancia=0.8,
        max_puntos=60
    )

    print(
        f"[STL] Puntos originales: {len(puntos_fisicos)}",
        flush=True
    )

    print(
        f"[STL] Puntos utilizados para geometría: {len(puntos_stl)}",
        flush=True
    )

    # ================================================================
    # 7. PLACA BASE
    # ================================================================

    modelo = (
        cq.Workplane("XY")
        .center(dim_x / 2, dim_y / 2)
        .box(
            dim_x,
            dim_y,
            BASE_THICKNESS,
            centered=(True, True, False)
        )
    )

    # ================================================================
    # 8. EJES
    # ================================================================

    eje_x_inicio = (izquierda, abajo)
    eje_x_fin = (dim_x - derecha, abajo)

    eje_y_inicio = (izquierda, abajo)
    eje_y_fin = (izquierda, dim_y - arriba)

    modelo = agregar_segmento_relieve(
        modelo,
        eje_x_inicio,
        eje_x_fin,
        2.0,
        RELIEVE_EJE
    )

    modelo = agregar_segmento_relieve(
        modelo,
        eje_y_inicio,
        eje_y_fin,
        2.0,
        RELIEVE_EJE
    )

    # ================================================================
    # 9. TICKS Y BRAILLE
    # ================================================================

    NUM_TICKS = 5

    for i in range(NUM_TICKS):

        fraccion = i / (NUM_TICKS - 1)

        x_fis = (
            izquierda
            + fraccion * ancho_plot
        )

        y_fis = (
            abajo
            + fraccion * alto_plot
        )

        # ------------------------------
        # Tick X
        # ------------------------------

        modelo = agregar_segmento_relieve(
            modelo,
            (x_fis, abajo - 3),
            (x_fis, abajo + 3),
            1.5,
            RELIEVE_TICK
        )

        # ------------------------------
        # Tick Y
        # ------------------------------

        modelo = agregar_segmento_relieve(
            modelo,
            (izquierda - 3, y_fis),
            (izquierda + 3, y_fis),
            1.5,
            RELIEVE_TICK
        )

        # ------------------------------
        # Valores Braille
        # ------------------------------

        if calibrados:

            x_valor = (
                x_min
                + fraccion * (x_max - x_min)
            )

            y_valor = (
                y_min
                + fraccion * (y_max - y_min)
            )

            # X

            ancho_x = _ancho_numero(x_valor)

            inicio_x = min(
                max(
                    3.0,
                    x_fis - ancho_x / 2
                ),
                dim_x - ancho_x - 3.0
            )

            modelo = agregar_numero_braille(
                modelo,
                round(x_valor),
                inicio_x,
                abajo - CLEARANCE_BRAILLE
            )

            # Y

            ancho_y = _ancho_numero(y_valor)

            inicio_y = max(
                3.0,
                izquierda
                - CLEARANCE_BRAILLE
                - ancho_y
            )

            modelo = agregar_numero_braille(
                modelo,
                round(y_valor),
                inicio_y,
                y_fis
            )

    # ================================================================
    # 10. CURVA
    # ================================================================

    modelo = agregar_funcion(
        modelo,
        puntos_stl,
        diametro=DIAM_LINEA,
        altura=RELIEVE_LINEA
    )

    # ================================================================
    # 11. EXPORTAR
    # ================================================================

    cq.exporters.export(
        modelo,
        archivo_salida
    )

    print(
        f"[STL] Modelo táctil generado: {archivo_salida}",
        flush=True
    )

    return modelo

if __name__ == "__main__":
    generar_modelo_bana()
