"""Generador de placas táctiles STL para la Fase 1.

Este módulo reemplaza a la implementación anterior del generador STL y
convierte el Bloque 3 en la fuente de verdad del proyecto, manteniendo la
compatibilidad con la API y con el flujo de segmentación existente.

Geometría: numpy + numpy-stl, sin kernel CAD.
-----------------------------------------------
Hasta esta versión, la geometría se construía con `cadquery` (un kernel CAD
paramétrico completo sobre OpenCascade/OCCT), pensado para modelado con
booleanos topológicos reales, fillets, cortes, etc. Acá nunca se usó nada de
eso: todo lo que se dibuja son cajas, cilindros y esferas simples que se
tocan o se superponen levemente entre sí y con la placa base — que es todo
lo que necesita un slicer de impresión 3D para fusionarlas al cortar por
capas; no hace falta que el STL sea un único sólido topológicamente cerrado.

Usar un kernel CAD completo para esto resultó ser, medido en producción, la
causa de que Render matara el contenedor por falta de memoria (evento "Out
of Memory" confirmado) al generar una placa con títulos y varias series:
picos de ~1.3-1.7GB de RAM y hasta 100s. El mismo modelo construido a mano
como arreglos de triángulos (numpy) y escrito con `numpy-stl` midió <50MB de
pico y ~2s — la sobrecarga era enteramente del kernel CAD, no de la
geometría en sí.
"""

import copy
import math
import re
import unicodedata
from math import atan2, degrees, hypot

import numpy as np
from stl import mesh

# Tabla y medidas Braille: una sola fuente para todo el proyecto (braille.py,
# el módulo "braille2" adaptado). Incluye el signo de mayúscula, las letras
# propias del español (á, é, í, ó, ú, ü, ñ) y el reemplazo de etiquetas
# largas por identificadores (A, B, C...) explicados en la leyenda.
from braille import (BRAILLE, CELDA_PITCH, DIAM_PUNTO_BRAILLE, ESPACIADO_BRAILLE,
                     Abreviador, ancho_braille, partir_en_renglones, texto_a_celdas)

# =============================================================================
# CONSTANTES
# =============================================================================
BASE_THICKNESS = 1.5
CLEARANCE_BRAILLE = 9.5
RELIEVE_EJE = 1.0
RELIEVE_TICK = 1.0
RELIEVE_LINEA = 1.6
DIAM_LINEA = 2.0
MARGEN_BORDE = 25.0

# Una fila de texto Braille ocupa esto de alto (dos filas de puntos + su
# diámetro), y la separación de BANA entre elementos Braille no relacionados
# es CLEARANCE_BRAILLE. Se usan para reservar espacio para títulos.
ALTURA_FILA_BRAILLE = ESPACIADO_BRAILLE * 2 + DIAM_PUNTO_BRAILLE
# Distancia entre renglones seguidos de un mismo bloque de texto (títulos
# apilados, renglones de la leyenda): el interlineado Braille habitual.
INTERLINEA_BRAILLE = 10.0

# Patrón "rayado" (serie 2): largo del tramo dibujado y del hueco, en mm.
RAYA_LARGO = 6.0
RAYA_HUECO = 3.5

# Patrón "punteado" (serie 3): separación mínima entre bultos, en mm, para
# que no se junten hasta parecer una línea continua otra vez.
PUNTEADO_ESPACIADO = 5.0

# Patrón "celdas" (curva principal): una cadena CONTINUA de celdas largas,
# anchas y altas, unidas por un cuello corto, angosto y más bajo. Al tacto
# es una sola línea que no se corta, pero con un ritmo (celda-muesca-celda)
# que la distingue de los ejes, que son lisos y más bajos.
CELDA_LARGO = 5.0            # mm de cada celda
CUELLO_LARGO = 2.0           # mm del cuello que queda a la vista entre dos celdas
CUELLO_DIAMETRO = 1.0        # mm (la celda usa el diámetro del estilo)
CUELLO_ALTURA = 1.0          # mm sobre la placa (la celda usa la altura del estilo)

# Resolución de las mallas generadas a mano (nº de caras). Antes esto lo
# decidía automáticamente el teselado de OCCT; acá se elige directamente, sin
# depender de una "tolerancia" indirecta. 8x12 y 16 lados ya son más finos de
# lo que una impresora FDM de escritorio puede resolver físicamente.
ESFERA_LAT = 8
ESFERA_LON = 12
CILINDRO_LADOS = 16

# Cuánto se hunde cada pieza en relieve dentro de la placa/pieza vecina (mm).
# Sin un kernel CAD que suelde topológicamente las piezas, dos superficies
# que solo se TOCAN (sin superponerse) dependen de que el slicer las una
# bien al cortar por capas — la mayoría lo hace, pero garantizar una pequeña
# superposición real elimina cualquier duda, sin cambiar el relieve visible
# (se resta del punto de partida y se suma a la altura, así que lo que
# sobresale por encima de la superficie mide exactamente lo mismo).
SOLAPE = 0.15

# ============================================================
# GENERADORES DE MALLA (numpy puro, sin kernel CAD)
# ============================================================
# Cada función devuelve un arreglo numpy (n_triángulos, 3, 3): n triángulos,
# 3 vértices por triángulo, 3 coordenadas (x, y, z) por vértice. Los
# devanados (orden de los vértices) están verificados a mano para que la
# normal de cada cara apunte hacia afuera de la pieza.

def _malla_caja(ancho, profundidad, altura, cx=0.0, cy=0.0, z0=0.0):
    """Una caja rectangular como 12 triángulos.

    Equivale a lo que antes hacía
    `cq.Workplane("XY").workplane(offset=z0).center(cx, cy)
       .box(ancho, profundidad, altura, centered=(True, True, False))`.
    """
    x0, x1 = cx - ancho / 2.0, cx + ancho / 2.0
    y0, y1 = cy - profundidad / 2.0, cy + profundidad / 2.0
    z1 = z0 + altura
    v = np.array([
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # 0-3: abajo
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # 4-7: arriba
    ])
    caras = (
        (0, 2, 1), (0, 3, 2),  # abajo    (normal -Z)
        (4, 5, 6), (4, 6, 7),  # arriba   (normal +Z)
        (0, 1, 5), (0, 5, 4),  # frente   (normal -Y)
        (1, 2, 6), (1, 6, 5),  # derecha  (normal +X)
        (2, 3, 7), (2, 7, 6),  # atrás    (normal +Y)
        (3, 0, 4), (3, 4, 7),  # izquierda(normal -X)
    )
    return v[np.array(caras)]


def _malla_prisma(poligono, altura, z0=0.0):
    """Un polígono CONVEXO (vértices en sentido antihorario, en mm) extruido
    `altura` mm hacia arriba desde z0: la placa base con una esquina
    recortada. Tapas en abanico y una pared por lado, con el mismo sentido
    de vértices que `_malla_caja` (normales hacia afuera)."""
    z1 = z0 + altura
    abajo = [(x, y, z0) for x, y in poligono]
    arriba = [(x, y, z1) for x, y in poligono]
    tris = []
    for i in range(1, len(poligono) - 1):
        tris.append((arriba[0], arriba[i], arriba[i + 1]))      # tapa de arriba (+Z)
        tris.append((abajo[0], abajo[i + 1], abajo[i]))         # tapa de abajo (-Z)
    for i in range(len(poligono)):
        j = (i + 1) % len(poligono)
        tris.append((abajo[i], abajo[j], arriba[j]))            # pared
        tris.append((abajo[i], arriba[j], arriba[i]))
    return np.array(tris, dtype=float)


def contorno_placa(ancho, alto, chaflan):
    """Contorno de la placa (antihorario): un rectángulo al que le falta un
    triángulo de `chaflan` mm de lado en la esquina SUPERIOR DERECHA. Esa
    esquina cortada permite orientar la lámina al tacto (saber cuál es el
    "arriba") antes de empezar a leerla."""
    if chaflan <= 0:
        return [(0.0, 0.0), (ancho, 0.0), (ancho, alto), (0.0, alto)]
    return [(0.0, 0.0), (ancho, 0.0), (ancho, alto - chaflan), (ancho - chaflan, alto), (0.0, alto)]


def _malla_cilindro(radio, altura, cx=0.0, cy=0.0, z0=0.0, lados=CILINDRO_LADOS):
    """Un cilindro (círculo extruido) aproximado por un prisma de `lados`
    caras, con sus dos tapas.

    Equivale a lo que antes hacía
    `cq.Workplane("XY").workplane(offset=z0).center(cx, cy)
       .circle(radio).extrude(altura)`.
    """
    angulos = np.linspace(0, 2 * np.pi, lados, endpoint=False)
    anillo_x = cx + radio * np.cos(angulos)
    anillo_y = cy + radio * np.sin(angulos)
    z1 = z0 + altura

    tris = []
    for i in range(lados):
        j = (i + 1) % lados
        p0b, p1b = (anillo_x[i], anillo_y[i], z0), (anillo_x[j], anillo_y[j], z0)
        p0t, p1t = (anillo_x[i], anillo_y[i], z1), (anillo_x[j], anillo_y[j], z1)
        tris.append(((cx, cy, z0), p1b, p0b))   # tapa de abajo (-Z)
        tris.append(((cx, cy, z1), p0t, p1t))   # tapa de arriba (+Z)
        tris.append((p0b, p1b, p1t))            # pared (normal radial saliente)
        tris.append((p0b, p1t, p0t))
    return np.array(tris)


def _malla_esfera(radio, cx=0.0, cy=0.0, cz=0.0, lat=ESFERA_LAT, lon=ESFERA_LON):
    """Una esfera aproximada por una grilla latitud/longitud (malla UV
    estándar).

    Equivale a lo que antes hacía
    `cq.Workplane("XY").workplane(offset=cz).center(cx, cy).sphere(radio)`
    (acá `cz` ya es la coordenada Z absoluta del centro).
    """
    def punto(theta, phi):
        return (
            cx + radio * math.sin(theta) * math.cos(phi),
            cy + radio * math.sin(theta) * math.sin(phi),
            cz + radio * math.cos(theta),
        )

    tris = []
    for i in range(lat):
        theta0, theta1 = math.pi * i / lat, math.pi * (i + 1) / lat
        for j in range(lon):
            phi0, phi1 = 2 * math.pi * j / lon, 2 * math.pi * (j + 1) / lon
            p00, p01 = punto(theta0, phi0), punto(theta0, phi1)
            p10, p11 = punto(theta1, phi0), punto(theta1, phi1)
            if i != 0:
                tris.append((p00, p10, p11))
            if i != lat - 1:
                tris.append((p00, p11, p01))
    return np.array(tris)


def _rotar_z(triangulos, angulo_grados):
    """Rota una malla alrededor del eje Z que pasa por el origen."""
    if triangulos.size == 0:
        return triangulos
    ang = math.radians(angulo_grados)
    c, s = math.cos(ang), math.sin(ang)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    forma = triangulos.shape
    return (triangulos.reshape(-1, 3) @ rot.T).reshape(forma)


def _trasladar(triangulos, dx, dy, dz):
    """Traslada una malla."""
    if triangulos.size == 0:
        return triangulos
    return triangulos + np.array([dx, dy, dz])


def _guardar_stl(triangulos, archivo_salida):
    """Escribe una malla (arreglo N×3×3) a un archivo STL binario."""
    m = mesh.Mesh(np.zeros(triangulos.shape[0], dtype=mesh.Mesh.dtype))
    m.vectors[:] = triangulos
    m.update_normals()
    m.save(archivo_salida)
    return m


def agregar_punto_braille(piezas, cx, cy):
    """Agrega un punto Braille (una esfera en relieve) a la lista `piezas`.

    No se combina con nada todavía: las piezas se acumulan y se concatenan
    con la placa base de una sola vez al final (ver `_ensamblar`). La esfera
    nace centrada exactamente en la superficie de la placa (mitad adentro,
    mitad afuera), así que ya queda bien anclada sin necesitar ningún
    boolean.

    Lo que se agrega es la descripción de la pieza (una "primitiva"), no su
    malla: la malla se arma recién al exportar (`_malla_de`). Así la misma
    distribución sirve para la vista previa de la lámina sin el costo de
    triangular todo.
    """
    piezas.append({"t": "p", "x": float(cx), "y": float(cy)})


def agregar_caracter_braille(piezas, caracter, cx, cy):
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
        agregar_punto_braille(piezas, cx + dx, cy + dy)


def agregar_texto_braille(piezas, texto, cx, cy):
    """Coloca una cadena de caracteres Braille en línea."""
    x = cx
    for ch in texto:
        agregar_caracter_braille(piezas, ch, x, cy)
        x += CELDA_PITCH


def _decimales_necesarios(valores, max_decimales=3):
    """Cuántos decimales hacen falta para que los valores de un eje (sus
    marcas/ticks) se distingan entre sí al redondear, sin pasarse de
    `max_decimales`.

    Reemplaza el `round()` a entero que antes se aplicaba siempre: con un eje
    0.0-1.0 (accuracy, loss, probabilidad — habitual en papers) cinco marcas
    equiespaciadas redondeaban a "0, 0, 0, 1, 1", es decir, dos valores
    Braille repetidos para cinco marcas físicas distintas.
    """
    if len(valores) < 2:
        return 0
    for nd in range(max_decimales + 1):
        redondeados = {round(v, nd) for v in valores}
        if len(redondeados) == len(valores):
            return nd
    return max_decimales


def _formatear_valor_braille(valor, decimales):
    """(es_negativo, dígitos_enteros, dígitos_decimales) para dibujar `valor`
    redondeado a `decimales` cifras decimales."""
    valor_r = round(float(valor), decimales)
    es_negativo = valor_r < 0
    valor_abs = abs(valor_r)
    if decimales <= 0:
        return es_negativo, str(int(round(valor_abs))), ""
    texto = f"{valor_abs:.{decimales}f}"
    entero, _, frac = texto.partition(".")
    return es_negativo, entero, frac


def agregar_numero_braille(piezas, valor, cx, cy, decimales=0):
    """Coloca un número en formato Nemeth: signo menos si corresponde,
    indicador numeral, parte entera y, si `decimales` > 0, punto decimal
    Nemeth + parte decimal.

    Con `decimales=0` (el valor por defecto) el comportamiento es idéntico
    al de la versión anterior, que solo aceptaba enteros.
    """
    es_negativo, entero, frac = _formatear_valor_braille(valor, decimales)
    x = cx
    if es_negativo:
        agregar_caracter_braille(piezas, "menos", x, cy)
        x += CELDA_PITCH
    agregar_caracter_braille(piezas, "numeral", x, cy)
    x += CELDA_PITCH
    for ch in entero:
        agregar_caracter_braille(piezas, ch, x, cy)
        x += CELDA_PITCH
    if frac:
        agregar_caracter_braille(piezas, "punto", x, cy)
        x += CELDA_PITCH
        for ch in frac:
            agregar_caracter_braille(piezas, ch, x, cy)
            x += CELDA_PITCH


def _valores_reales_eje(etiquetas, minimo, valor_min, valor_max, tolerancia=0.15):
    """Valores de eje realmente leídos por OCR (los que el segmentador
    reporta en "textos.etiquetas_eje_x/y"), en vez de inventar marcas
    equiespaciadas. Descarta duplicados y cualquier lectura muy fuera del
    dominio calibrado [valor_min, valor_max]: esas son justo las que
    ajustar_lineal_robusto() del segmentador ya identificó como probable
    error de OCR y excluyó de la calibración, así que tampoco deberían
    terminar impresas en la placa.

    Devuelve una lista ordenada y sin duplicados, o [] si hay menos de
    `minimo` valores utilizables (el llamador debe usar un respaldo).
    """
    margen = tolerancia * ((valor_max - valor_min) or 1.0)
    vistos = set()
    valores = []
    for e in etiquetas or []:
        valor = e.get("valor")
        if valor is None or not math.isfinite(valor):
            continue
        if valor < valor_min - margen or valor > valor_max + margen:
            continue
        clave = round(float(valor), 6)
        if clave in vistos:
            continue
        vistos.add(clave)
        valores.append(float(valor))
    if len(valores) < minimo:
        return []
    valores.sort()
    return valores


def _sanear_texto_braille(texto):
    """Deja el texto en minúsculas y sin tildes para que las letras
    encuentren su signo en la tabla BRAILLE.

    Limitación conocida: no hay indicador de "vuelta a letras" dentro de un
    texto corrido, así que un dígito incrustado en un título (p. ej.
    "figura 3") se dibuja con la misma forma que una letra, sin el
    indicador numeral — ambigüedad aceptable para un título, pero por eso
    los NÚMEROS DE LOS EJES (el dato que importa) se dibujan siempre con
    agregar_numero_braille(), no con esta función.
    """
    sin_tildes = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return sin_tildes.lower()


def agregar_segmento_relieve(piezas, p0, p1, diametro, altura):
    """Agrega un tramo recto en relieve, con dos extremos redondeados, a la
    lista `piezas` (tres piezas: el cuerpo y las dos tapas).

    Arrancan `SOLAPE` mm por debajo de la superficie de la placa (en vez de
    justo en el borde) para garantizar una superposición real: sin un
    kernel CAD que suelde topológicamente las piezas, dos superficies que
    solo se tocan dependen de que el slicer las una bien al cortar por
    capas. La altura visible por encima de la superficie no cambia.
    """
    x0 = float(p0[0])
    y0 = float(p0[1])
    x1 = float(p1[0])
    y1 = float(p1[1])

    dx = x1 - x0
    dy = y1 - y0

    largo = hypot(dx, dy)

    if largo < 1e-6:
        return

    piezas.append({"t": "s", "a": (x0, y0), "b": (x1, y1),
                   "d": float(diametro), "h": float(altura)})


def _malla_segmento(p0, p1, diametro, altura):
    """Malla de un tramo en relieve: el cuerpo y las dos tapas redondas."""
    x0, y0 = p0
    x1, y1 = p1
    largo = hypot(x1 - x0, y1 - y0)
    angulo = degrees(atan2(y1 - y0, x1 - x0))

    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    z0 = BASE_THICKNESS - SOLAPE
    altura_real = float(altura) + SOLAPE

    cuerpo_local = _malla_caja(largo, float(diametro), altura_real, cx=0.0, cy=0.0, z0=0.0)
    cuerpo = _trasladar(_rotar_z(cuerpo_local, angulo), mx, my, z0)

    tapa0 = _malla_cilindro(float(diametro) / 2, altura_real, cx=x0, cy=y0, z0=z0)
    tapa1 = _malla_cilindro(float(diametro) / 2, altura_real, cx=x1, cy=y1, z0=z0)
    return np.concatenate([cuerpo, tapa0, tapa1], axis=0)


def agregar_punto_relieve(piezas, punto, diametro, altura):
    """Un bulto redondo aislado en el punto dado (usado por el patrón
    "punteado" para distinguir una tercera serie al tacto). Ver `SOLAPE`
    en `agregar_segmento_relieve`."""
    x, y = float(punto[0]), float(punto[1])
    piezas.append({"t": "b", "x": x, "y": y, "d": float(diametro), "h": float(altura)})


def _malla_de(pieza):
    """Malla de una primitiva (ver agregar_punto_braille); una malla ya
    armada pasa tal cual."""
    if not isinstance(pieza, dict):
        return pieza
    if pieza["t"] == "p":
        return _malla_esfera(DIAM_PUNTO_BRAILLE / 2, cx=pieza["x"], cy=pieza["y"], cz=BASE_THICKNESS)
    if pieza["t"] == "s":
        return _malla_segmento(pieza["a"], pieza["b"], pieza["d"], pieza["h"])
    return _malla_cilindro(pieza["d"] / 2, pieza["h"] + SOLAPE,
                           cx=pieza["x"], cy=pieza["y"], z0=BASE_THICKNESS - SOLAPE)


def _primitiva_json(pieza):
    """Primitiva para la vista previa del navegador (mm, 2 decimales)."""
    r = lambda v: round(float(v), 2)  # noqa: E731
    if pieza["t"] == "s":
        return {"t": "s", "a": [r(pieza["a"][0]), r(pieza["a"][1])],
                "b": [r(pieza["b"][0]), r(pieza["b"][1])], "d": r(pieza["d"])}
    if pieza["t"] == "b":
        return {"t": "b", "x": r(pieza["x"]), "y": r(pieza["y"]), "d": r(pieza["d"])}
    return {"t": "p", "x": r(pieza["x"]), "y": r(pieza["y"])}


def _puntos_espaciados(puntos, distancia_min):
    """Filtra una polilínea para que sus puntos queden separados al menos
    `distancia_min` mm entre sí. Sin esto, un patrón "punteado" con muchos
    puntos cercanos vuelve a parecer una línea continua."""
    if not puntos:
        return []
    filtrados = [puntos[0]]
    for p in puntos[1:]:
        if hypot(p[0] - filtrados[-1][0], p[1] - filtrados[-1][1]) >= distancia_min:
            filtrados.append(p)
    if filtrados[-1] != puntos[-1]:
        filtrados.append(puntos[-1])
    return filtrados


def _dividir_en_rayas(p0, p1, largo_raya=RAYA_LARGO, largo_hueco=RAYA_HUECO):
    """Divide un segmento recto en tramos alternos "encendido"/"apagado"
    para simular una línea discontinua (patrón "rayado", segunda serie).

    El patrón reinicia en cada segmento de la polilínea ya simplificada (no
    se arrastra el hueco pendiente del segmento anterior); es una
    aproximación suficiente para distinguir series al tacto, no una réplica
    exacta de un patrón de líneas discontinuas de un editor CAD.
    """
    x0, y0 = p0
    x1, y1 = p1
    largo = hypot(x1 - x0, y1 - y0)
    if largo < 1e-9:
        return []
    ux, uy = (x1 - x0) / largo, (y1 - y0) / largo
    ciclo = largo_raya + largo_hueco
    tramos = []
    pos = 0.0
    while pos < largo - 1e-9:
        fin = min(pos + largo_raya, largo)
        a = (x0 + ux * pos, y0 + uy * pos)
        b = (x0 + ux * fin, y0 + uy * fin)
        tramos.append((a, b))
        pos += ciclo
    return tramos


def _dividir_en_celdas(puntos, largo_celda=CELDA_LARGO, largo_cuello=CUELLO_LARGO):
    """Recorre TODA la polilínea (sin reiniciar en cada vértice, así el
    ritmo es parejo aunque la curva tenga muchos quiebres) y la corta en
    tramos alternos celda / cuello. Devuelve [(es_celda, [p0, ..., pn])];
    cada tramo empieza donde terminó el anterior, así la línea no se corta."""
    tramos = []
    es_celda = True
    restante = largo_celda
    actual = [tuple(puntos[0])]
    for a, b in zip(puntos[:-1], puntos[1:]):
        largo = hypot(b[0] - a[0], b[1] - a[1])
        if largo < 1e-9:
            continue
        pos = 0.0
        while largo - pos > 1e-9:
            paso = min(restante, largo - pos)
            pos += paso
            punto = (a[0] + (b[0] - a[0]) * pos / largo, a[1] + (b[1] - a[1]) * pos / largo)
            actual.append(punto)
            restante -= paso
            if restante <= 1e-9:
                tramos.append((es_celda, actual))
                es_celda = not es_celda
                restante = largo_celda if es_celda else largo_cuello
                actual = [punto]
    if len(actual) >= 2:
        tramos.append((es_celda, actual))
    return tramos


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


_ESTILOS_SERIE = [
    # Curva principal: cadena continua de celdas (ver CELDA_LARGO).
    {"nombre": "en celdas", "patron": "celdas", "diametro": DIAM_LINEA * 1.25, "altura": RELIEVE_LINEA},
    {"nombre": "rayada", "patron": "rayado", "diametro": DIAM_LINEA * 0.85, "altura": RELIEVE_LINEA + 0.4},
    {"nombre": "punteada", "patron": "punteado", "diametro": DIAM_LINEA * 1.3, "altura": RELIEVE_LINEA - 0.3},
]

# Los dos ejes son líneas CONTINUAS, más bajas que las curvas de datos
# (RELIEVE_EJE < RELIEVE_LINEA): se sienten como guía, no como dato. Antes
# el eje X era rayado y el Y punteado para distinguirlos entre sí, pero en
# las pruebas con estudiantes esos cortes desorientaban al seguir el eje.
EJE_X_ESTILO = {
    "patron": "solido", "diametro": 2.0, "altura": RELIEVE_EJE,
    "raya_largo": RAYA_LARGO, "raya_hueco": RAYA_HUECO, "punteado_espaciado": PUNTEADO_ESPACIADO,
}
EJE_Y_ESTILO = dict(EJE_X_ESTILO)


# Leyenda, en la parte de ABAJO de la placa. Dos clases de entradas:
#  - con 2+ series, una muestra corta de la textura de cada serie + su nombre;
#  - cada texto que no entraba en su lugar y se escribió como "A", "B"...:
#    el identificador + el texto completo.
# Un texto largo sigue en el renglón de abajo (con sangría), nunca se recorta.
LEYENDA_MUESTRA = 16.0   # largo de la muestra de textura (mm)
LEYENDA_HUECO = 4.0      # de la muestra al texto
# Entre dos entradas del mismo renglón: más que un espacio entre palabras
# (8,5 mm de borde a borde), así no se lee como una sola frase; con 11 mm
# entran 4 valores de eje ("a 2012") por renglón.
LEYENDA_ENTRE = 11.0

# El área del gráfico conserva la proporción alto/ancho del original (una
# pendiente se siente igual que se ve), pero nunca más chata que esto.
PROPORCION_MIN = 0.5
# Alto mínimo del área del gráfico: si la leyenda no deja tanto, se acorta
# la leyenda (y se avisa), no el gráfico.
ALTO_PLOT_MIN = 60.0

# Números de los ejes con letras minúsculas (a, b...: 1 celda, ver
# braille.Abreviador). En el eje Y se usan solo si algún número tiene al
# menos CELDAS_NUMERO_LARGO celdas (10000, 1500,5...): ahí el gráfico gana
# ancho (una letra ocupa el mismo alto que un número).
CELDAS_LETRA = 1
CELDAS_NUMERO_LARGO = 6


# Margen libre en los cuatro bordes de la placa. El conjunto (números del
# eje Y + gráfico + filas de títulos) se reparte dentro de ese marco con el
# mismo margen a cada lado: queda centrado en la placa.
MARGEN_PLACA = 10.0

# Del borde izquierdo de un texto Braille al centro de su primera celda.
BORDE_A_CELDA = (ESPACIADO_BRAILLE + DIAM_PUNTO_BRAILLE) / 2


def _dibujar_celdas(piezas, celdas, x_primera, y):
    """Dibuja celdas Braille ya armadas (ver _celdas_braille_mixto);
    `x_primera` es el centro de la primera celda."""
    for k, celda in enumerate(celdas):
        agregar_caracter_braille(piezas, celda, x_primera + k * CELDA_PITCH, y)


def _celdas_numero(valor, decimales=0):
    """Celdas Braille de un número, igual que agregar_numero_braille():
    [menos] numeral dígitos [punto dígitos]."""
    es_negativo, entero, frac = _formatear_valor_braille(valor, decimales)
    celdas = (["menos"] if es_negativo else []) + ["numeral"] + list(entero)
    if frac:
        celdas += ["punto"] + list(frac)
    return celdas


def _ancho_celdas(n):
    """Ancho real (mm) de n celdas Braille, de borde a borde de los puntos."""
    return max(0, n - 1) * CELDA_PITCH + ESPACIADO_BRAILLE + DIAM_PUNTO_BRAILLE if n else 0.0


def _indices_legibles(posiciones, tamanos, separacion_min):
    """Qué números de un eje escribir para que no se toquen: uno de cada k,
    con el menor k posible (quedan equiespaciados: 10, 30, 50... en vez de
    un amontonamiento ilegible al tacto). Las marcas en relieve se dibujan
    todas igual; solo se ralean los números."""
    n = len(posiciones)
    for k in range(1, n + 1):
        idx = list(range(0, n, k))
        if all(abs(posiciones[b] - posiciones[a]) - (tamanos[a] + tamanos[b]) / 2 >= separacion_min
               for a, b in zip(idx, idx[1:])):
            return idx
    return [0] if n else []


def _decimales_de(valor, maximo=2):
    """Decimales con que se escribió un valor (12 -> 0, 12.5 -> 1)."""
    texto = f"{float(valor):.{maximo}f}".rstrip("0").rstrip(".")
    return len(texto.split(".")[1]) if "." in texto else 0


def _distancia_rect_polilinea(rect, puntos, paso=1.0):
    """Distancia mínima (mm) entre un rectángulo (x0, y0, x1, y1) y una
    polilínea, muestreando la polilínea cada `paso` mm."""
    x0, y0, x1, y1 = rect
    mejor = float("inf")
    for (ax, ay), (bx, by) in zip(puntos[:-1], puntos[1:]):
        n = max(1, int(hypot(bx - ax, by - ay) / paso))
        for i in range(n + 1):
            px, py = ax + (bx - ax) * i / n, ay + (by - ay) * i / n
            dx = max(x0 - px, 0.0, px - x1)
            dy = max(y0 - py, 0.0, py - y1)
            mejor = min(mejor, hypot(dx, dy))
    return mejor


def nombre_de_serie(nombre, indice):
    """Nombre a mostrar de la serie `indice` (0, 1, ...): el leído de la
    leyenda o, si no hay, "Serie A", "Serie B"... (letras y no números: en
    Braille un dígito suelto dentro de un texto se confunde con una letra)."""
    nombre = (nombre or "").strip()
    return nombre or f"Serie {chr(ord('A') + indice % 26)}"


def _celdas_braille_mixto(texto):
    """Celdas Braille de un texto con letras y números (braille.texto_a_celdas:
    signo numeral delante de cada número, signo de mayúscula, letras con
    tilde del español). Lo que no está en la tabla se omite."""
    return texto_a_celdas(texto)


def _texto_diseno(tipo, rotulo, recuadro):
    """Entrada de diseno["textos"]: un texto escrito en la placa."""
    return {
        "tipo": tipo,
        "texto": rotulo["texto_stl"],              # lo que dice el Braille
        "texto_completo": rotulo["texto_original"],
        "identificador": rotulo["identificador"],  # None si está completo
        "recuadro": list(recuadro),
    }


def _rotulo_simple(texto):
    """Un texto escrito completo (sin letra de la leyenda), como los que
    devuelve braille.Abreviador.procesar."""
    return {"texto_stl": texto, "texto_original": texto, "identificador": None}


def _texto_numero(valor, decimales):
    """Cómo se lee en tinta un número tal como queda en Braille ("-1,5")."""
    es_negativo, entero, frac = _formatear_valor_braille(valor, decimales)
    return ("-" if es_negativo else "") + entero + ("," + frac if frac else "")


# Ajustes del supervisor sobre la lámina (vista previa, antes del STL):
#   {"textos":  {id: texto nuevo},   títulos, categorías, nombres de serie,
#                                    valores anotados
#    "ocultar": [id, ...],            lo que no se imprime
#    "mover":   {id: [dx, dy] mm}}    textos corridos de su lugar
# Los id son los de diseno["elementos"]: "titulo", "titulo_eje_x",
# "titulo_eje_y", "num_x_3", "num_y_0", "cat_2", "dato_1", "serie_0",
# "leyenda_serie_1"... (el número es la posición en el JSON del segmentador,
# así no cambia aunque se oculte otra cosa).
_RE_AJUSTE = re.compile(r"(titulo|titulo_eje_x|titulo_eje_y|(?:num_x|num_y|cat|dato|serie|leyenda_serie)_\d+)")


def aplicar_ajustes(datos, ajustes):
    """Copia de la entrada del generador con los textos cambiados y sin las
    series ocultas. Devuelve (datos, ocultas): `ocultas` describe las series
    que se sacaron, para poder mostrarlas y restaurarlas.

    Los cambios de texto van acá (y no en el generador) para que la
    narración diga lo mismo que la lámina."""
    if not ajustes or not isinstance(datos, dict):
        return datos, []
    datos = copy.deepcopy(datos)
    textos = datos["textos"] = dict(datos.get("textos") or {})
    series = datos.get("series") or []
    cambios = ajustes.get("textos") or {}
    for clave, valor in cambios.items():
        valor = str(valor).strip()
        if clave in ("titulo", "titulo_eje_x", "titulo_eje_y"):
            textos[clave] = valor
            continue
        m = re.fullmatch(r"(cat|dato|serie)_(\d+)", clave)
        if not m:
            continue
        tipo, i = m.group(1), int(m.group(2))
        if tipo == "cat" and i < len(textos.get("categorias_x") or []) and valor:
            cats = list(textos["categorias_x"])
            cats[i] = valor
            textos["categorias_x"] = cats
        elif tipo == "dato" and i < len(textos.get("etiquetas_dato") or []):
            try:
                numero = float(valor.replace(",", "."))
            except ValueError:
                continue
            if math.isfinite(numero):
                etiquetas = [dict(e) for e in textos["etiquetas_dato"]]
                etiquetas[i]["valor"] = numero
                textos["etiquetas_dato"] = etiquetas
        elif tipo == "serie" and i < len(series):
            series[i] = {**series[i], "nombre": valor}

    ocultar = set(ajustes.get("ocultar") or [])
    ocultas = []
    if series:
        visibles = []
        for i, s in enumerate(series):
            s = {**s, "_id": i}
            if f"serie_{i}" in ocultar:
                ocultas.append({"id": f"serie_{i}", "clase": "serie",
                                "texto": nombre_de_serie(s.get("nombre"), i)})
            else:
                visibles.append(s)
        if not any(len(s.get("puntos") or []) >= 2 for s in visibles):
            # no se puede quitar TODAS las curvas: queda la lámina vacía
            visibles, ocultas = [{**s, "_id": i} for i, s in enumerate(series)], []
        datos["series"] = visibles
    return datos, ocultas


def _ordenar_abreviaturas(leyenda):
    """Leyenda de letras en orden de lectura: los textos (A, B...) y después
    los valores del eje Y y del eje X (a, b...), cada eje junto."""
    def grupo(a):
        donde = a.get("donde") or ""
        return 2 if donde == "número del eje X" else 1 if donde == "número del eje Y" else 0
    return sorted(leyenda, key=grupo)


def _armar_leyenda(series, abreviaturas, ancho):
    """Reparte la leyenda en renglones de `ancho` mm.

    `series`: [(índice, nombre)] -> muestra de textura + nombre.
    `abreviaturas`: entradas de braille.Abreviador.leyenda -> "A texto".

    Las entradas que entran en un renglón lo comparten (en orden, separadas
    por LEYENDA_ENTRE); una que no entra ni en un renglón entero sigue en
    los de abajo, con sangría. Devuelve una lista de renglones; cada renglón
    es una lista de piezas {"dx" (desde el borde izquierdo de la zona),
    "serie" (índice si lleva muestra, si no None), "prefijo" (celdas del
    identificador o None), "sangria" (mm del inicio a la primera celda de
    texto), "celdas", "entrada" (posición en series + abreviaturas)}."""
    entradas = [("serie", i, LEYENDA_MUESTRA + LEYENDA_HUECO, None, _celdas_braille_mixto(nombre))
                for i, nombre in series]
    entradas += [("abreviatura", None, (len(a["celdas_identificador"]) + 1) * CELDA_PITCH,
                  a["celdas_identificador"], a["celdas_texto"]) for a in abreviaturas]
    renglones, ocupado = [], None
    for k, (tipo, serie, sangria, prefijo, celdas) in enumerate(entradas):
        max_celdas = int((ancho - sangria - ESPACIADO_BRAILLE - DIAM_PUNTO_BRAILLE) // CELDA_PITCH) + 1
        partes = partir_en_renglones(celdas, max_celdas) or [[]]
        ancho_primera = sangria + ancho_braille(partes[0])
        primera = {"dx": 0.0, "serie": serie, "prefijo": prefijo, "sangria": sangria,
                   "celdas": partes[0], "entrada": k}
        if (len(partes) == 1 and renglones and ocupado is not None
                and ocupado + LEYENDA_ENTRE + ancho_primera <= ancho):
            primera["dx"] = ocupado + LEYENDA_ENTRE
            renglones[-1].append(primera)
            ocupado += LEYENDA_ENTRE + ancho_primera
            continue
        renglones.append([primera])
        ocupado = ancho_primera
        for parte in partes[1:]:
            renglones.append([{"dx": sangria, "serie": None, "prefijo": None, "sangria": 0.0,
                               "celdas": parte, "entrada": k}])
            ocupado = None   # tras un texto partido no se comparte el renglón
    return renglones


def _linea_recta(p0, p1, paso):
    """Puntos equiespaciados a lo largo de un segmento recto, cada `paso` mm
    aprox. Un eje solo tiene 2 puntos (sus extremos); el patrón "punteado"
    necesita varios puntos intermedios para poner una fila de bultos a lo
    largo, no solo uno en cada punta."""
    x0, y0 = p0
    x1, y1 = p1
    largo = hypot(x1 - x0, y1 - y0)
    if largo < 1e-9:
        return [p0, p1]
    n = max(2, int(round(largo / paso)) + 1)
    return [(x0 + (x1 - x0) * (i / (n - 1)), y0 + (y1 - y0) * (i / (n - 1))) for i in range(n)]


def agregar_funcion(
    piezas,
    puntos,
    diametro=DIAM_LINEA,
    altura=RELIEVE_LINEA,
    patron="solido",
    raya_largo=RAYA_LARGO,
    raya_hueco=RAYA_HUECO,
    punteado_espaciado=PUNTEADO_ESPACIADO,
):
    """
    Agrega una polilínea en relieve a la lista `piezas`.

    Los puntos deben estar ya convertidos a coordenadas físicas.

    `patron` distingue táctilmente varios elementos superpuestos en la
    misma placa (series de datos entre sí, y cada eje de la curva):
      - "solido"   -> línea continua.
      - "rayado"   -> tramos discontinuos (largo/hueco configurables).
      - "punteado" -> bultos redondos aislados, sin línea (espaciado
        configurable). Con solo 2 puntos (un tramo recto, como un eje) hay
        que densificarlo primero — ver `_linea_recta` — porque si no el
        patrón solo pondría un bulto en cada punta.
      - "celdas"   -> línea continua hecha de celdas largas (diámetro y
        altura del estilo) unidas por cuellos angostos y bajos.
    """

    if not puntos or len(puntos) < 2:
        return

    if patron == "celdas":
        # Cada tramo recto termina en una tapa redonda que sobresale medio
        # diámetro: la celda se acorta eso en cada punta (y el cuello se
        # alarga lo mismo) para que las tapas no tapen el cuello y la muesca
        # entre celdas mida de verdad CUELLO_LARGO.
        largo_celda = max(CELDA_LARGO - diametro, 0.5)
        for es_celda, tramo in _dividir_en_celdas(puntos, largo_celda, CUELLO_LARGO + diametro):
            d = diametro if es_celda else CUELLO_DIAMETRO
            h = altura if es_celda else CUELLO_ALTURA
            for p0, p1 in zip(tramo[:-1], tramo[1:]):
                agregar_segmento_relieve(piezas, p0, p1, d, h)
        return

    if patron == "punteado":
        if len(puntos) == 2:
            puntos = _linea_recta(puntos[0], puntos[1], punteado_espaciado)
        for p in _puntos_espaciados(puntos, punteado_espaciado):
            agregar_punto_relieve(piezas, p, diametro, altura)
        return

    for p0, p1 in zip(puntos[:-1], puntos[1:]):
        if patron == "rayado":
            for a, b in _dividir_en_rayas(p0, p1, raya_largo, raya_hueco):
                agregar_segmento_relieve(piezas, a, b, diametro, altura)
        else:
            agregar_segmento_relieve(piezas, p0, p1, diametro, altura)


def _agregar_eje(piezas, p0, p1, estilo):
    """Dibuja un eje (tramo recto de p0 a p1) con su textura propia
    (ver EJE_X_ESTILO / EJE_Y_ESTILO)."""
    agregar_funcion(
        piezas, [p0, p1],
        diametro=estilo["diametro"], altura=estilo["altura"], patron=estilo["patron"],
        raya_largo=estilo["raya_largo"], raya_hueco=estilo["raya_hueco"],
        punteado_espaciado=estilo["punteado_espaciado"],
    )


def _ensamblar(base, piezas):
    """Combina la placa base con TODAS las piezas en relieve en un solo
    arreglo de triángulos, con una simple concatenación de numpy — nada de
    boolean/CSG (ver la nota al principio del archivo sobre por qué no hace
    falta)."""
    trozos = [base] + [m for m in (_malla_de(p) for p in piezas if p is not None)
                       if m is not None and len(m)]
    return np.concatenate(trozos, axis=0)


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

    modelo = _malla_caja(dim_x, dim_y, BASE_THICKNESS, cx=dim_x / 2, cy=dim_y / 2, z0=0.0)

    grosor_eje = 2.0
    z_ejes = BASE_THICKNESS + RELIEVE_EJE
    z_ticks = BASE_THICKNESS + RELIEVE_TICK

    x0_fis, x1_fis = a_fisico((x_min, 0))[0], a_fisico((x_max, 0))[0]
    y0_fis, y1_fis = a_fisico((0, y_min))[1], a_fisico((0, y_max))[1]

    piezas = []

    # Ejes y ticks arrancan en z=0 (atraviesan toda la placa) en vez de
    # apenas en su superficie, así que ya quedan bien anclados sin
    # necesitar ningún boolean.
    eje_x = _malla_caja(
        x1_fis - x0_fis, grosor_eje, z_ejes,
        cx=(x0_fis + x1_fis) / 2, cy=origen_y_fis, z0=0.0,
    )
    eje_y = _malla_caja(
        grosor_eje, y1_fis - y0_fis, z_ejes,
        cx=origen_x_fis, cy=(y0_fis + y1_fis) / 2, z0=0.0,
    )
    piezas.append(eje_x)
    piezas.append(eje_y)

    longitud_tick = 6.0
    for valor in _valores_tick(x_min, x_max, intervalo_ticks):
        x_fis = origen_x_fis + valor
        if valor != 0:
            tick = _malla_caja(
                grosor_eje, longitud_tick, z_ticks,
                cx=x_fis, cy=origen_y_fis, z0=0.0,
            )
            piezas.append(tick)
        agregar_numero_braille(
            piezas, valor, cx=x_fis, cy=origen_y_fis - CLEARANCE_BRAILLE
        )

    for valor in _valores_tick(y_min, y_max, intervalo_ticks):
        if valor == 0:
            continue
        y_fis = origen_y_fis + valor
        tick = _malla_caja(
            longitud_tick, grosor_eje, z_ticks,
            cx=origen_x_fis, cy=y_fis, z0=0.0,
        )
        piezas.append(tick)
        ancho_estimado = CELDA_PITCH * (len(str(abs(valor))) + 2)
        agregar_numero_braille(
            piezas, valor, cx=origen_x_fis - CLEARANCE_BRAILLE - ancho_estimado, cy=y_fis
        )

    agregar_texto_braille(
        piezas, "x", cx=x1_fis - CELDA_PITCH,
        cy=origen_y_fis - CLEARANCE_BRAILLE - CELDA_PITCH * 2,
    )
    agregar_texto_braille(
        piezas, "y", cx=origen_x_fis - CLEARANCE_BRAILLE - CELDA_PITCH * 3,
        cy=y1_fis - CELDA_PITCH,
    )

    agregar_funcion(piezas, [a_fisico(p1), a_fisico(p2)])

    triangulos = _ensamblar(modelo, piezas)
    _guardar_stl(triangulos, archivo_salida)
    print(f"Modelo táctil BANA ({dim_x}x{dim_y}mm) exportado a: {archivo_salida}")
    return triangulos


# Compatibilidad con el código previo del proyecto. Nada más en el repo las
# llama, pero se actualiza su firma (reciben `piezas`, no `modelo`) para que
# sigan siendo un espejo fiel de las funciones que envuelven.
def _punto_braille(piezas, cx, cy):
    return agregar_punto_braille(piezas, cx, cy)


def _caracter_braille(piezas, caracter, cx, cy):
    return agregar_caracter_braille(piezas, caracter, cx, cy)


def _numero_braille(piezas, valor, cx, cy):
    return agregar_numero_braille(piezas, valor, cx, cy)


def _segmento(piezas, inicio, fin, diametro, altura):
    return agregar_segmento_relieve(piezas, inicio, fin, diametro, altura)


def _ticks(minimo, maximo, intervalo):
    return _valores_tick(minimo, maximo, intervalo)


def _ancho_numero(valor, decimales=0):
    """Ancho horizontal (mm) que ocupará `agregar_numero_braille` para este
    valor: se calcula con el mismo formateo que usa el dibujo, así que
    coincide celda por celda (antes no contaba la celda del signo menos en
    valores negativos, y el número podía quedar más pegado al vecino de lo
    estimado)."""
    es_negativo, entero, frac = _formatear_valor_braille(valor, decimales)
    celdas = 1 + len(entero)                  # numeral + dígitos enteros
    celdas += 1 if es_negativo else 0         # signo menos
    celdas += (1 + len(frac)) if frac else 0  # punto decimal + dígitos
    return CELDA_PITCH * celdas


# Formato de la lámina: placa cuadrada de 22 x 22 cm con la esquina superior
# derecha recortada (orientación al tacto). El gráfico va ARRIBA (con sus
# títulos, números y rótulos) y lo que sobra abajo es para la leyenda.
PLACA_ANCHO = 220.0
PLACA_ALTO = 220.0
PLACA_MARGEN_SUPERIOR = 0.0   # franja extra en blanco arriba (además de MARGEN_PLACA)
PLACA_CHAFLAN = 10.0


def generar_modelo_desde_recta(
    datos_segmentador,
    dim_x=PLACA_ANCHO,
    dim_y=PLACA_ALTO,
    archivo_salida="grafica_tactil.stl",
    incluir_etiquetas=False,
    incluir_leyenda=True,
    diseno=None,
    margen_superior=PLACA_MARGEN_SUPERIOR,
    chaflan=PLACA_CHAFLAN,
    ajustes=None,
):
    """
    Genera una placa táctil a partir del resultado completo de
    segmentador.procesar_imagen().

    Puede recibir:

    1) El resultado completo del nuevo segmentador:
        {
            "series": [...],
            "puntos_curva": [...],
            "resumen": {...},
            ...
        }

    2) Una lista antigua de puntos:
        [
            {
                "px": ...,
                "py": ...,
                "valor_x": ...,
                "valor_y": ...
            },
            ...
        ]

    `incluir_etiquetas` (por defecto False, por ahora: se está revisando
    solo la forma de ejes y curvas) controla los números Braille de los
    ticks y los títulos del gráfico/ejes. Las marcas físicas de los ticks y
    los ejes/curvas en sí no se ven afectados —no son texto—, y los
    márgenes ya quedan reservados para cuando se reactive (`True`), sin
    necesitar volver a acomodar la placa.

    `incluir_etiquetas=True` además escribe en Braille los valores que el
    gráfico original tenía anotados junto a la curva ("etiquetas de dato"),
    en su lugar, corridos lo justo para no pisar ninguna curva. Un texto que
    no entra en su lugar (un título muy largo, una categoría más ancha que
    el espacio entre marcas) se escribe como "A", "B"... y su texto completo
    va a la leyenda de abajo (braille.Abreviador).

    Distribución: el gráfico (títulos, números, ejes y curvas) va ARRIBA,
    con la proporción alto/ancho del original; lo que sobra abajo es la
    leyenda.

    `incluir_leyenda` (por defecto True): con 2 o más series, la leyenda de
    abajo empieza con una muestra de la textura de cada serie y su nombre.

    `margen_superior` (mm): franja extra en blanco arriba (0 por defecto).
    `chaflan` (mm): lado del triángulo recortado en la esquina superior
    derecha de la placa (0 = sin recorte).

    `diseno`: si se pasa un dict, se llena con dónde quedó cada cosa en la
    placa (mm, origen abajo a la izquierda) y la escala usada. Lo usa
    narracion.py para que un programa de narración sepa qué hay bajo el dedo.
    También trae "elementos": cada cosa en relieve (textos, ejes, curvas,
    leyenda) con sus primitivas, para dibujar la vista previa.

    `archivo_salida=None`: solo vista previa, se llena `diseno` sin armar
    las mallas ni escribir el STL (mucho más rápido).

    `ajustes`: lo que el supervisor cambió en la vista previa ("ocultar" y
    "mover"; los cambios de texto y las series ocultas se aplican antes,
    con aplicar_ajustes).
    """

    if dim_x < 130 or dim_y < 90:
        raise ValueError(
            "La placa debe medir al menos 130 x 90 mm."
        )
    # Alto que pueden usar el gráfico y la leyenda: todo menos la franja de
    # arriba en blanco.
    alto_util = dim_y - margen_superior
    if alto_util < 90:
        raise ValueError(
            "Con ese margen superior no quedan al menos 90 mm de alto para el gráfico."
        )

    # ================================================================
    # 1. EXTRAER INFORMACIÓN DEL NUEVO SEGMENTADOR
    # ================================================================

    es_resultado_completo = isinstance(datos_segmentador, dict)

    if es_resultado_completo:
        resultado = datos_segmentador

        series = resultado.get("series") or []

        puntos_legacy = resultado.get("puntos_curva") or []

        resumen = resultado.get("resumen") or {}

    elif isinstance(datos_segmentador, (list, tuple)):
        # Compatibilidad con el formato anterior
        resultado = {}
        series = []
        puntos_legacy = list(datos_segmentador)
        resumen = {}

    else:
        raise ValueError(
            "La entrada debe ser el resultado del segmentador "
            "o una lista de puntos."
        )

    # Título, nombres de ejes, leyenda y etiquetas numéricas realmente
    # leídas por OCR ("textos" del JSON del segmentador). Con el formato
    # antiguo (lista de puntos) no hay nada de esto disponible.
    textos = resultado.get("textos") or {}

    ajustes = ajustes or {}
    ocultos = set(ajustes.get("ocultar") or [])
    movidos = {}
    for clave, d in (ajustes.get("mover") or {}).items():
        try:
            movidos[clave] = (float(d[0]), float(d[1]))
        except (TypeError, ValueError, IndexError):
            pass
    ocultos_info = []   # lo que no se imprime por decisión del supervisor

    def _titulo_visible(clave, texto, clase):
        # un título oculto ni siquiera reserva su renglón: el gráfico crece
        texto = (texto or "").strip()
        if texto and clave in ocultos:
            ocultos_info.append({"id": clave, "clase": clase, "texto": texto})
            return ""
        return texto

    titulo_grafico = _titulo_visible("titulo", textos.get("titulo"), "titulo")
    titulo_eje_x_txt = _titulo_visible("titulo_eje_x", textos.get("titulo_eje_x"), "titulo_eje")
    titulo_eje_y_txt = _titulo_visible("titulo_eje_y", textos.get("titulo_eje_y"), "titulo_eje")

    # ================================================================
    # 2. OBTENER LAS SERIES
    # ================================================================

    if series:

        series_puntos = []
        nombres_leidos = []
        ids_series = []    # posición en el JSON original (ver aplicar_ajustes)

        for k, serie in enumerate(series):

            puntos = serie.get("puntos", [])

            if isinstance(puntos, list) and len(puntos) >= 2:
                series_puntos.append(puntos)
                nombres_leidos.append(serie.get("nombre"))
                ids_series.append(serie.get("_id", k))

    elif puntos_legacy:

        series_puntos = [puntos_legacy]
        nombres_leidos = [None]
        ids_series = [0]

    else:

        raise ValueError(
            "El segmentador no detectó ninguna serie de puntos."
        )

    if not series_puntos:
        raise ValueError(
            "No hay suficientes puntos para generar el STL."
        )

    # ================================================================
    # 3. RECTÁNGULO REAL DEL GRÁFICO
    # ================================================================

    rect = resumen.get("rect_grafico")

    if rect and len(rect) == 4:

        rect_x1 = float(rect[0])
        rect_y1 = float(rect[1])
        rect_x2 = float(rect[2])
        rect_y2 = float(rect[3])

    else:

        # Compatibilidad con versiones antiguas.
        # Calculamos el rectángulo a partir de los puntos.

        todos = [
            p
            for serie in series_puntos
            for p in serie
        ]

        columnas = [
            float(p["px"])
            for p in todos
            if p.get("px") is not None
        ]

        filas = [
            float(p["py"])
            for p in todos
            if p.get("py") is not None
        ]

        if len(columnas) < 2 or len(filas) < 2:
            raise ValueError(
                "No se pudo determinar el área del gráfico."
            )

        rect_x1 = min(columnas)
        rect_x2 = max(columnas)
        rect_y1 = min(filas)
        rect_y2 = max(filas)

    if rect_x2 <= rect_x1 or rect_y2 <= rect_y1:
        raise ValueError(
            "El rectángulo del gráfico no es válido."
        )

    # ================================================================
    # 4. OBTENER CALIBRACIÓN DE LOS EJES
    # ================================================================

    calibracion_x = resultado.get("ejes", {}).get("calibracion_x", {})
    calibracion_y = resultado.get("ejes", {}).get("calibracion_y", {})

    m_x = calibracion_x.get("m")
    b_x = calibracion_x.get("b")

    m_y = calibracion_y.get("m")
    b_y = calibracion_y.get("b")

    # Eje logarítmico (ver segmentador.calibrar_eje): m y b convierten el
    # píxel en log10(valor). Todo el armado de la placa trabaja en ese
    # "espacio del eje" (así una curva exponencial queda recta, como en el
    # original) y solo los números que se escriben en Braille son el valor
    # real.
    log_x = calibracion_x.get("escala") == "log"
    log_y = calibracion_y.get("escala") == "log"

    # Si no viene dentro de "ejes", intentar buscarlo en resumen
    # o reconstruirlo desde los puntos calibrados.

    if m_x is None or b_x is None:

        puntos_calibrados = [
            p
            for serie in series_puntos
            for p in serie
            if p.get("valor_x") is not None
        ]

        if len(puntos_calibrados) >= 2:

            px1 = float(puntos_calibrados[0]["px"])
            px2 = float(puntos_calibrados[-1]["px"])

            vx1 = float(puntos_calibrados[0]["valor_x"])
            vx2 = float(puntos_calibrados[-1]["valor_x"])

            if abs(px2 - px1) > 1e-9:

                m_x = (vx2 - vx1) / (px2 - px1)
                b_x = vx1 - m_x * px1

    if m_y is None or b_y is None:

        puntos_calibrados = [
            p
            for serie in series_puntos
            for p in serie
            if p.get("valor_y") is not None
        ]

        if len(puntos_calibrados) >= 2:

            py1 = float(puntos_calibrados[0]["py"])
            py2 = float(puntos_calibrados[-1]["py"])

            vy1 = float(puntos_calibrados[0]["valor_y"])
            vy2 = float(puntos_calibrados[-1]["valor_y"])

            if abs(py2 - py1) > 1e-9:

                m_y = (vy2 - vy1) / (py2 - py1)
                b_y = vy1 - m_y * py1

    calibrados = (
        m_x is not None
        and b_x is not None
        and m_y is not None
        and b_y is not None
    )

    # ================================================================
    # 5. DOMINIO REAL DEL GRÁFICO
    # ================================================================

    if calibrados:

        # X:
        # izquierda -> derecha
        x_val_izquierda = m_x * rect_x1 + b_x
        x_val_derecha = m_x * rect_x2 + b_x

        x_min = min(x_val_izquierda, x_val_derecha)
        x_max = max(x_val_izquierda, x_val_derecha)

        # Y:
        # En la imagen y crece hacia abajo.
        #
        # Por eso el valor superior y el inferior se calculan
        # directamente mediante la calibración.

        y_val_superior = m_y * rect_y1 + b_y
        y_val_inferior = m_y * rect_y2 + b_y

        y_min = min(y_val_superior, y_val_inferior)
        y_max = max(y_val_superior, y_val_inferior)

    else:

        # Sin calibración, utilizar coordenadas de píxel.
        x_min = rect_x1
        x_max = rect_x2

        # Invertimos Y para obtener coordenadas cartesianas.
        y_min = -rect_y2
        y_max = -rect_y1

    rango_x = x_max - x_min
    rango_y = y_max - y_min

    if abs(rango_x) < 1e-9:
        rango_x = 1.0

    if abs(rango_y) < 1e-9:
        rango_y = 1.0

    # ================================================================
    # 6. VALORES DE LAS MARCAS
    # ================================================================
    # Antes las marcas eran siempre 5 valores equiespaciados entre x_min y
    # x_max (derivados del borde del rectángulo del gráfico), así que casi
    # nunca coincidían con los números que realmente estaban impresos en la
    # gráfica original. Ahora, si el segmentador leyó al menos 2 etiquetas
    # numéricas reales por eje, se usan ESAS —mismo valor que vio el OCR—;
    # solo se cae a marcas sintéticas equiespaciadas si no hay etiquetas
    # reales suficientes (p. ej. calibración hecha con las etiquetas de dato
    # pegadas a la curva, sin números de eje legibles).
    # Se calculan ANTES de repartir la placa: el ancho de los números del
    # eje Y decide cuánto margen dejarles a la izquierda.
    NUM_TICKS_SINTETICOS = 5
    MAX_TICKS_POR_EJE = 12  # límite defensivo: no cubrir la placa de números

    def _al_eje(etiquetas, log):
        """Etiquetas con su valor pasado al espacio del eje (log10 si es log)."""
        salida = []
        for e in etiquetas or []:
            v = e.get("valor")
            if v is None or not math.isfinite(v) or (log and v <= 0):
                continue
            salida.append({**e, "valor": math.log10(v) if log else v})
        return salida

    valores_x, valores_y = [], []   # en el espacio del eje (posición de cada marca)
    if calibrados:
        etiquetas_x_json = _al_eje(textos.get("etiquetas_eje_x"), log_x)
        etiquetas_y_json = _al_eje(textos.get("etiquetas_eje_y"), log_y)

        valores_x = _valores_reales_eje(etiquetas_x_json, 2, x_min, x_max)
        if not valores_x:
            valores_x = [x_min + (i / (NUM_TICKS_SINTETICOS - 1)) * rango_x
                         for i in range(NUM_TICKS_SINTETICOS)]

        valores_y = _valores_reales_eje(etiquetas_y_json, 2, y_min, y_max)
        if not valores_y:
            valores_y = [y_min + (i / (NUM_TICKS_SINTETICOS - 1)) * rango_y
                         for i in range(NUM_TICKS_SINTETICOS)]

        valores_x = valores_x[:MAX_TICKS_POR_EJE]
        valores_y = valores_y[:MAX_TICKS_POR_EJE]

    # Lo que se ESCRIBE en cada marca: el valor real (10 ** v en un eje log).
    reales_x = [10.0 ** v if log_x else v for v in valores_x]
    reales_y = [10.0 ** v if log_y else v for v in valores_y]
    decimales_x = _decimales_necesarios(reales_x) if reales_x else 0
    decimales_y = _decimales_necesarios(reales_y) if reales_y else 0

    # ================================================================
    # 6b. DISTRIBUCIÓN EN LA PLACA: gráfico ARRIBA, leyenda ABAJO
    # ================================================================
    # Horizontal: el gráfico llena el ancho con MARGEN_PLACA en cada borde:
    #   izquierda: números del eje Y + separación BANA hasta el eje;
    #   derecha:   hasta el margen, salvo lo justo para que el último rótulo
    #              del eje X (centrado en su valor) no se salga de la placa.
    # Vertical, de arriba hacia abajo:
    #   título del gráfico y título del eje Y (un renglón cada uno);
    #   área del gráfico, con la proporción alto/ancho del original;
    #   rótulos del eje X y título del eje X;
    #   leyenda (texturas de las series y textos abreviados), en lo que sobra.
    #
    # Un texto que no entra en su lugar se escribe como "A", "B"... y su
    # texto completo va a la leyenda (braille.Abreviador). Las letras se
    # asignan en orden de lectura: título, eje Y, categorías, eje X.
    #
    # Eje X de categorías ("Ene", "Feb"...): el valor i del eje es la
    # categoría i, y en la placa se escribe su nombre en vez del número.

    categorias_x = [str(c) for c in (textos.get("categorias_x") or [])]
    ancho_texto = dim_x - 2 * MARGEN_PLACA     # un renglón de lado a lado

    def _es_categoria(valor):
        i = int(round(valor))
        return bool(categorias_x) and abs(valor - i) <= 0.25 and 0 <= i < len(categorias_x)

    nombres_series = [nombre_de_serie(n, i) for i, n in enumerate(nombres_leidos)]
    series_leyenda = ([(i, nombres_series[i]) for i in range(len(series_puntos))
                       if f"leyenda_serie_{ids_series[i]}" not in ocultos]
                      if incluir_leyenda and len(series_puntos) > 1 else [])
    if incluir_leyenda and len(series_puntos) > 1:
        ocultos_info += [{"id": f"leyenda_serie_{ids_series[i]}", "clase": "leyenda",
                          "texto": nombres_series[i]} for i in range(len(series_puntos))
                         if f"leyenda_serie_{ids_series[i]}" in ocultos]
    proporcion = max(PROPORCION_MIN, (rect_y2 - rect_y1) / (rect_x2 - rect_x1))
    y_tope = alto_util - MARGEN_PLACA
    celdas_num_x = [None if _es_categoria(v) else _celdas_numero(v, decimales_x) for v in reales_x]
    celdas_num_y = [_celdas_numero(v, decimales_y) for v in reales_y]
    # pocos números en el eje: con letras se gana poco y se pierde lectura directa
    con_letras_y_posible = bool(celdas_num_y) and max(map(len, celdas_num_y)) >= CELDAS_NUMERO_LARGO

    def _alto_leyenda(n):
        return (CLEARANCE_BRAILLE + ALTURA_FILA_BRAILLE + (n - 1) * INTERLINEA_BRAILLE) if n else 0.0

    def _legibles(posiciones, celdas, tamanos, separacion):
        """Índices (de todos) que se escriben: solo los que tienen texto, y
        de esos, los que entran sin tocarse (ver _indices_legibles)."""
        con_texto = [i for i, c in enumerate(celdas) if c]
        elegidos = _indices_legibles([posiciones[i] for i in con_texto],
                                     [tamanos[i] for i in con_texto], separacion)
        return [con_texto[k] for k in elegidos]

    # Se prueba en este orden y se queda con el primero cuya leyenda entra:
    #   1. números de los ejes con letras (si así se escriben más) y
    #      categorías largas con letras;
    #   2. solo las categorías con letras;
    #   3. todo escrito completo (se escriben los que entran sin tocarse).
    # Una letra en la placa sin su explicación no le sirve al lector: antes
    # que cortar la leyenda, se vuelve a escribir los textos completos.
    # Cada modo se arma en dos pasadas: la primera reparte letras a las
    # marcas que entrarían; si con la distribución final se escriben otras,
    # la segunda reparte las letras justo a esas (y así la leyenda explica
    # exactamente las letras que están en la placa).
    for modo in (("ejes", "categorias"), ("categorias",), ()):
        fijos_x = fijos_y = None
        for pasada in range(2):
            abreviador = Abreviador()

            def _abreviar(texto, ancho, donde, clave):
                if not incluir_etiquetas or clave in ocultos:
                    # no se escribe: no hace falta letra (ni ocupa lugar)
                    celdas = [] if clave in ocultos else _celdas_braille_mixto(texto)
                    return {"celdas": celdas, "usa_leyenda": False,
                            "identificador": None, "texto_original": texto, "texto_stl": texto}
                return abreviador.procesar(texto, ancho, donde, clave)

            def _letra(texto, celdas, donde, clave):
                if clave in ocultos:
                    return None
                return abreviador.procesar(texto, -1.0, donde, clave, celdas=celdas, minuscula=True)

            # Títulos de arriba (las letras van en orden de lectura: título,
            # eje Y, eje X). El del eje Y empieza en la columna de sus números.
            titulos_arriba = []   # (clave, nombre, rótulo)
            if titulo_grafico:
                titulos_arriba.append(("titulo", "título del gráfico",
                                       _abreviar(titulo_grafico, ancho_texto, "título del gráfico",
                                                 "titulo")))
            if titulo_eje_y_txt:
                titulos_arriba.append(("titulo_eje_y", "título del eje Y",
                                       _abreviar(titulo_eje_y_txt, ancho_texto - CLEARANCE_BRAILLE,
                                                 "título del eje Y", "titulo_eje_y")))

            # --- Números del eje Y: con letras solo si son muy anchos (una
            # letra ocupa el mismo alto que un número: no hace entrar más
            # marcas, pero le devuelve ancho al gráfico) ---
            rotulos_y = [None] * len(reales_y)
            letras_y = "ejes" in modo and incluir_etiquetas and con_letras_y_posible
            if letras_y:
                for i in (fijos_y if fijos_y is not None else range(len(reales_y))):
                    rotulos_y[i] = _letra(_texto_numero(reales_y[i], decimales_y), celdas_num_y[i],
                                          "número del eje Y", f"num_y_{i}")
            celdas_y = [r["celdas"] if r else ([] if letras_y else c)
                        for r, c in zip(rotulos_y, celdas_num_y)]
            anchos_y = [_ancho_celdas(len(c)) for c in celdas_y]
            ancho_numeros_y = max(anchos_y, default=0.0)
            bloque_izquierdo = (ancho_numeros_y + CLEARANCE_BRAILLE) if valores_y else 6.0
            izquierda = MARGEN_PLACA + bloque_izquierdo
            disponible = dim_x - MARGEN_PLACA - izquierda
            x_titulo_y = max(MARGEN_PLACA, izquierda - ancho_numeros_y)
            posiciones = [(v - x_min) / rango_x * disponible for v in valores_x]

            # --- Categorías del eje X: cada una tiene de ancho el espacio
            # hasta la marca vecina menos una celda; si no entra, va su letra ---
            orden = sorted(posiciones)
            paso_min = min((b - a for a, b in zip(orden, orden[1:])), default=disponible)
            ancho_categoria = max(0.0, paso_min - CELDA_PITCH) if "categorias" in modo else float("inf")
            rotulos_x = [_abreviar(categorias_x[int(round(v))], ancho_categoria, "categoría del eje X",
                                   f"cat_{int(round(v))}")
                         if _es_categoria(v) else None for v in reales_x]

            # --- Números del eje X: con letras si completos no entran todos
            # y con letras entran más ---
            letras_x = [None] * len(reales_x)
            numericos = [i for i, c in enumerate(celdas_num_x) if c is not None]
            if "ejes" in modo and incluir_etiquetas and len(numericos) >= 2:
                pos = [posiciones[i] for i in numericos]
                completos = _indices_legibles(
                    pos, [_ancho_celdas(len(celdas_num_x[i])) for i in numericos], CELDA_PITCH)
                con_letra = _indices_legibles(
                    pos, [_ancho_celdas(CELDAS_LETRA)] * len(numericos), CELDA_PITCH)
                if len(completos) < len(numericos) and len(con_letra) > len(completos):
                    elegidos = fijos_x if fijos_x is not None else [numericos[k] for k in con_letra]
                    for i in elegidos:
                        letras_x[i] = _letra(_texto_numero(reales_x[i], decimales_x), celdas_num_x[i],
                                             "número del eje X", f"num_x_{i}")
            hay_letras_x = any(letras_x)
            celdas_x = [rotulos_x[i]["celdas"] if rotulos_x[i]
                        else letras_x[i]["celdas"] if letras_x[i]
                        else ([] if hay_letras_x else celdas_num_x[i])
                        for i in range(len(reales_x))]
            anchos_x = [_ancho_celdas(len(c)) for c in celdas_x]

            rotulo_titulo_x = (_abreviar(titulo_eje_x_txt, ancho_texto, "título del eje X", "titulo_eje_x")
                               if titulo_eje_x_txt else None)

            # Ancho del gráfico: hasta el margen derecho, achicado solo si algún
            # rótulo del eje X (centrado en su valor) quedaría fuera de la placa.
            ancho_plot = disponible
            for v, ancho_rotulo in zip(valores_x, anchos_x):
                fraccion = (v - x_min) / rango_x
                if fraccion > 1e-6:
                    ancho_plot = min(ancho_plot, (disponible - ancho_rotulo / 2) / min(fraccion, 1.0))
            derecha = dim_x - izquierda - ancho_plot

            # --- Vertical ---
            filas_titulo = [y_tope - ALTURA_FILA_BRAILLE / 2 - k * INTERLINEA_BRAILLE
                            for k in range(len(titulos_arriba))]
            if filas_titulo:
                tope_plot = filas_titulo[-1] - ALTURA_FILA_BRAILLE / 2 - CLEARANCE_BRAILLE
            else:
                # el número más alto del eje Y va centrado en el borde superior
                # del gráfico: sobresale medio renglón
                tope_plot = y_tope - ALTURA_FILA_BRAILLE / 2
            # debajo del eje X: renglón de rótulos (+ renglón del título del eje X)
            bajo_plot = CLEARANCE_BRAILLE + ALTURA_FILA_BRAILLE / 2 \
                + (INTERLINEA_BRAILLE if titulo_eje_x_txt else 0.0)

            leyenda_letras = _ordenar_abreviaturas(abreviador.leyenda)
            renglones_leyenda = _armar_leyenda(series_leyenda, leyenda_letras, ancho_texto)
            renglones_necesarios = len(renglones_leyenda)
            alto_ideal = ancho_plot * proporcion

            def _alto_libre(n):
                return tope_plot - bajo_plot - _alto_leyenda(n) - MARGEN_PLACA

            entra = _alto_libre(renglones_necesarios) >= min(ALTO_PLOT_MIN, alto_ideal)
            n_renglones = renglones_necesarios
            while n_renglones and _alto_libre(n_renglones) < min(ALTO_PLOT_MIN, alto_ideal):
                n_renglones -= 1
            renglones_leyenda = renglones_leyenda[:n_renglones]
            alto_plot = min(alto_ideal, _alto_libre(n_renglones))
            abajo = tope_plot - alto_plot          # altura del eje X

            # Qué marcas se escriben con la distribución final
            x_finales = [min(max(izquierda + (v - x_min) / rango_x * ancho_plot, izquierda),
                             izquierda + ancho_plot) for v in valores_x]
            y_finales = [min(max(abajo + (v - y_min) / rango_y * alto_plot, abajo), abajo + alto_plot)
                         for v in valores_y]
            dibujar_x = _legibles(x_finales, celdas_x, anchos_x, CELDA_PITCH)
            dibujar_y = _legibles(y_finales, celdas_y, [ALTURA_FILA_BRAILLE] * len(celdas_y),
                                  ESPACIADO_BRAILLE)
            con_letra_x = [i for i, r in enumerate(letras_x) if r]
            con_letra_y = [i for i, r in enumerate(rotulos_y) if r]
            if pasada == 0 and ((hay_letras_x and dibujar_x != con_letra_x)
                                or (letras_y and dibujar_y != con_letra_y)):
                fijos_x = dibujar_x if hay_letras_x else None
                fijos_y = dibujar_y if letras_y else None
                continue
            break

        hay_abreviadas = ((hay_letras_x or letras_y) if "ejes" in modo
                          else any(r and r["usa_leyenda"] for r in rotulos_x))
        if entra or not (modo and hay_abreviadas):
            break

    avisos = []
    if n_renglones < renglones_necesarios:
        avisos.append(
            f"La leyenda no entra completa en la placa: se escribieron {n_renglones} de "
            f"{renglones_necesarios} renglones. El texto completo está en la descripción narrada."
        )
    for eje, letras in (("X", [letras_x[i] for i in dibujar_x if letras_x[i]]),
                        ("Y", [rotulos_y[i] for i in dibujar_y if rotulos_y[i]])):
        if letras:
            avisos.append(
                f"Eje {eje}: los números no entraban completos; van con letras (de la "
                f"«{letras[0]['identificador']}» a la «{letras[-1]['identificador']}») y su valor "
                "está en la leyenda de abajo."
            )
    # primer renglón de la leyenda: debajo de lo que va bajo el eje X
    y_leyenda = abajo - bajo_plot - CLEARANCE_BRAILLE - ALTURA_FILA_BRAILLE / 2

    if ancho_plot <= 0 or alto_plot <= 0:
        raise ValueError(
            "Las dimensiones de la placa no dejan espacio suficiente "
            "para el gráfico."
        )

    # ================================================================
    # 7. CONVERSIÓN PIXEL -> VALOR -> MILÍMETROS
    # ================================================================

    def pixel_a_valor(px, py):

        if calibrados:

            x_val = m_x * px + b_x
            y_val = m_y * py + b_y

        else:

            x_val = px
            y_val = -py

        return x_val, y_val

    def valor_a_fisico(x_val, y_val):

        x_fis = (
            izquierda
            + (x_val - x_min)
            / rango_x
            * ancho_plot
        )

        y_fis = (
            abajo
            + (y_val - y_min)
            / rango_y
            * alto_plot
        )

        return x_fis, y_fis

    # ================================================================
    # 8. CONVERTIR CADA SERIE
    # ================================================================

    series_fisicas = []
    indices_fisicas = []   # posición de cada serie dibujada en series_puntos

    for indice, puntos in enumerate(series_puntos, start=1):

        puntos_fisicos = []

        for p in puntos:

            try:

                px = float(p["px"])
                py = float(p["py"])

            except (KeyError, TypeError, ValueError):

                continue

            if not math.isfinite(px) or not math.isfinite(py):
                continue

            x_val, y_val = pixel_a_valor(px, py)

            if not (
                math.isfinite(x_val)
                and math.isfinite(y_val)
            ):
                continue

            puntos_fisicos.append(
                valor_a_fisico(x_val, y_val)
            )

        if len(puntos_fisicos) >= 2:

            # Eliminar duplicados consecutivos
            limpios = [puntos_fisicos[0]]

            for p in puntos_fisicos[1:]:

                if hypot(
                    p[0] - limpios[-1][0],
                    p[1] - limpios[-1][1]
                ) > 1e-6:

                    limpios.append(p)

            if len(limpios) >= 2:

                puntos_simplificados = simplificar_polilinea(
                    limpios,
                    tolerancia=0.8,
                    max_puntos=60
                )

                series_fisicas.append(
                    puntos_simplificados
                )
                indices_fisicas.append(indice - 1)

                print(
                    f"[STL] Serie {indice}: "
                    f"{len(limpios)} puntos -> "
                    f"{len(puntos_simplificados)} puntos",
                    flush=True
                )

    if not series_fisicas:
        raise ValueError(
            "No quedaron puntos válidos después de convertir "
            "la curva a coordenadas físicas."
        )

    # ================================================================
    # 9. PLACA BASE
    # ================================================================

    # Todas las piezas en relieve (ejes, ticks, números y texto Braille,
    # curvas) se acumulan acá y se sueldan a la placa base de una sola vez
    # al final (ver _ensamblar) en vez de unirse una por una.
    piezas = []

    # Cada cosa de la lámina (un texto, un eje, una curva, una entrada de la
    # leyenda) con el tramo de `piezas` que le corresponde: es lo que la
    # vista previa dibuja y lo que el supervisor puede quitar o mover.
    elementos = []

    def _registrar(id_, clase, desde, **info):
        elementos.append({"id": id_, "clase": clase, "desde": desde, "hasta": len(piezas),
                          "ocultable": False, "movible": False, "editable": None, **info})

    def _texto(id_, clase, celdas, x0, y, tinta, **info):
        """Escribe un texto Braille (x0 = borde izquierdo) salvo que esté
        oculto, corrido si el supervisor lo movió. Devuelve su recuadro."""
        if id_ in ocultos:
            ocultos_info.append({"id": id_, "clase": clase,
                                 "texto": info.get("valor_edicion") or tinta})
            return None
        dx, dy = movidos.get(id_, (0.0, 0.0))
        x0, y = x0 + dx, y + dy
        desde = len(piezas)
        _dibujar_celdas(piezas, celdas, x0 + BORDE_A_CELDA, y)
        recuadro = [x0, y - ALTURA_FILA_BRAILLE / 2,
                    x0 + _ancho_celdas(len(celdas)), y + ALTURA_FILA_BRAILLE / 2]
        _registrar(id_, clase, desde, texto=tinta, recuadro=recuadro, ocultable=True,
                   movible=True, movido=bool(dx or dy), **info)
        return recuadro

    # ================================================================
    # 10. EJES
    # ================================================================

    eje_x_inicio = (izquierda, abajo)
    eje_x_fin = (
        dim_x - derecha,
        abajo
    )

    eje_y_inicio = (izquierda, abajo)
    eje_y_fin = (
        izquierda,
        abajo + alto_plot
    )

    # Ejes continuos, más bajos que las curvas: ver EJE_X_ESTILO / EJE_Y_ESTILO.
    desde = len(piezas)
    _agregar_eje(piezas, eje_x_inicio, eje_x_fin, EJE_X_ESTILO)
    _registrar("eje_x", "eje", desde, texto="Eje X")
    desde = len(piezas)
    _agregar_eje(piezas, eje_y_inicio, eje_y_fin, EJE_Y_ESTILO)
    _registrar("eje_y", "eje", desde, texto="Eje Y")

    # ================================================================
    # 11. NÚMEROS DE LOS EJES
    # ================================================================
    # Cada número va a la altura (eje Y) o debajo (eje X) de su valor, sin
    # marca que cruce el eje: los segmentos transversales cortaban la línea
    # del eje y desorientaban al seguirla con el dedo. Se escriben solo los
    # números que entran sin tocarse (ver _indices_legibles).
    textos_diseno = []   # dónde quedó cada texto escrito (para la narración)
    x_marcas = [min(max(valor_a_fisico(v, y_min)[0], izquierda), izquierda + ancho_plot)
                for v in valores_x]
    y_marcas = [min(max(valor_a_fisico(x_min, v)[1], abajo), abajo + alto_plot)
                for v in valores_y]

    if incluir_etiquetas:
        con_numero_x = dibujar_x
        for i in con_numero_x:
            inicio_x = min(max(MARGEN_PLACA, x_marcas[i] - anchos_x[i] / 2),
                           dim_x - MARGEN_PLACA - anchos_x[i])
            y_rotulo = abajo - CLEARANCE_BRAILLE
            if rotulos_x[i]:
                k = int(round(reales_x[i]))
                recuadro = _texto(f"cat_{k}", "categoria", celdas_x[i], inicio_x, y_rotulo,
                                  rotulos_x[i]["texto_stl"], editable="texto",
                                  clave_edicion=f"cat_{k}", valor_edicion=categorias_x[k])
                if recuadro:
                    textos_diseno.append(_texto_diseno("categoria", rotulos_x[i], recuadro))
            else:
                # número completo, o su letra (explicada en la leyenda)
                rotulo = letras_x[i] or _rotulo_simple(_texto_numero(reales_x[i], decimales_x))
                recuadro = _texto(f"num_x_{i}", "numero_eje", celdas_x[i], inicio_x, y_rotulo,
                                  rotulo["texto_stl"], valor_edicion=rotulo["texto_original"])
                if recuadro:
                    textos_diseno.append(_texto_diseno("numero_x", rotulo, recuadro))

        con_numero_y = dibujar_y
        for i in con_numero_y:
            inicio_y = max(MARGEN_PLACA, izquierda - CLEARANCE_BRAILLE - anchos_y[i])
            rotulo = rotulos_y[i] or _rotulo_simple(_texto_numero(reales_y[i], decimales_y))
            recuadro = _texto(f"num_y_{i}", "numero_eje", celdas_y[i], inicio_y, y_marcas[i],
                              rotulo["texto_stl"], valor_edicion=rotulo["texto_original"])
            if recuadro:
                textos_diseno.append(_texto_diseno("numero_y", rotulo, recuadro))

        for eje, total, escritos in (("X", len(valores_x), len(con_numero_x)),
                                     ("Y", len(valores_y), len(con_numero_y))):
            if escritos < total:
                avisos.append(
                    f"Eje {eje}: se escribieron {escritos} de {total} números en Braille "
                    "para que no se toquen."
                )

    # ================================================================
    # 11b. TÍTULOS EN BRAILLE (título del gráfico, nombre de cada eje)
    # ================================================================
    # Título del gráfico centrado arriba de todo; título del eje Y en el
    # renglón de abajo, alineado con la columna de números del eje Y; título
    # del eje X centrado bajo el gráfico. Un título que no entra en la placa
    # no se recorta: se escribe su letra (A, B...) y el texto completo va a
    # la leyenda.
    # (con incluir_etiquetas=False se saltea todo el bloque: el espacio ya
    # quedó reservado, así que la placa no se reacomoda.)
    if incluir_etiquetas:
        def _x_centrada(celdas, centro):
            ancho = _ancho_celdas(len(celdas))
            return min(max(MARGEN_PLACA, centro - ancho / 2), dim_x - MARGEN_PLACA - ancho)

        def _escribir(clave, nombre, rotulo, x0, y):
            recuadro = _texto(clave, "titulo" if clave == "titulo" else "titulo_eje",
                              rotulo["celdas"], x0, y, rotulo["texto_stl"], editable="texto",
                              clave_edicion=clave, valor_edicion=rotulo["texto_original"])
            textos_diseno.append(_texto_diseno(clave, rotulo, recuadro))
            if rotulo["usa_leyenda"]:
                avisos.append(f"El {nombre} no entraba en la placa: dice «{rotulo['identificador']}» "
                              "y el texto completo está en la leyenda de abajo.")

        if rotulo_titulo_x:
            _escribir("titulo_eje_x", "título del eje X", rotulo_titulo_x,
                      _x_centrada(rotulo_titulo_x["celdas"], izquierda + ancho_plot / 2),
                      abajo - CLEARANCE_BRAILLE - INTERLINEA_BRAILLE)

        for (clave, nombre, rotulo), y_fila in zip(titulos_arriba, filas_titulo):
            if clave == "titulo_eje_y":
                x0 = min(x_titulo_y, dim_x - MARGEN_PLACA - _ancho_celdas(len(rotulo["celdas"])))
            else:
                x0 = _x_centrada(rotulo["celdas"], dim_x / 2)
            _escribir(clave, nombre, rotulo, x0, y_fila)

        abreviadas = [a for a in abreviador.leyenda if a["donde"] == "categoría del eje X"]
        if abreviadas:
            avisos.append(f"{len(abreviadas)} categoría(s) del eje X no entraban entre las marcas: "
                          "van con su letra y el nombre completo está en la leyenda de abajo.")

    # ================================================================
    # 11c. VALORES ANOTADOS JUNTO A LA CURVA EN EL GRÁFICO ORIGINAL
    # ================================================================
    # Los números que el gráfico ya tenía escritos al lado de sus puntos
    # (etiquetas de dato), en Braille y en su lugar. Un número Braille es
    # mucho más grande que el impreso: se prueba en su posición y, si pisa
    # una curva, un eje u otro número, se corre hacia arriba o abajo de a
    # 1,5 mm (hasta 18 mm), o se pone a la izquierda/derecha del punto; si
    # no hay lugar libre cerca, se omite (y se avisa). Puede sobresalir un
    # poco por arriba del gráfico o hacia el margen derecho (un valor
    # anotado en la esquina suele estar ahí), sin tocar la fila de títulos.
    etiquetas_dato = textos.get("etiquetas_dato") or []
    datos_diseno = []
    if incluir_etiquetas and calibrados and etiquetas_dato:
        SEPARACION_CURVA = 2.0
        ocupados = []
        omitidas = 0
        for k_dato, e in enumerate(etiquetas_dato):
            if f"dato_{k_dato}" in ocultos:
                ocultos_info.append({"id": f"dato_{k_dato}", "clase": "dato",
                                     "texto": str(e.get("valor"))})
                continue
            try:
                valor = float(e["valor"])
                x_val, y_val = pixel_a_valor(float(e["px"]), float(e["py"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(valor) and math.isfinite(x_val) and math.isfinite(y_val)):
                continue
            xf, yf = valor_a_fisico(x_val, y_val)
            decimales = _decimales_de(valor)
            ancho = _ancho_numero(valor, decimales) - CELDA_PITCH + ESPACIADO_BRAILLE + DIAM_PUNTO_BRAILLE
            alto = ALTURA_FILA_BRAILLE
            # centrado sobre el punto, o a su izquierda, o a su derecha
            x_max_texto = dim_x - MARGEN_PLACA - ancho
            opciones_x = [min(max(izquierda + 1.0, x), x_max_texto)
                          for x in (xf - ancho / 2, xf - ancho - 2.0, xf + 2.0)]
            colocado = None
            candidatos = [(x0, yf + signo * paso * 1.5)
                          for paso in range(0, 13)
                          for signo in ((1,) if paso == 0 else (1, -1))
                          for x0 in opciones_x]
            for x0, yc in candidatos:
                rect = (x0, yc - alto / 2, x0 + ancho, yc + alto / 2)
                if rect[1] < abajo + 1.0 or rect[3] > abajo + alto_plot + 3.0:
                    continue
                if any(not (rect[2] + 1 < o[0] or o[2] + 1 < rect[0] or
                            rect[3] + 1 < o[1] or o[3] + 1 < rect[1]) for o in ocupados):
                    continue
                if any(_distancia_rect_polilinea(rect, s) < SEPARACION_CURVA for s in series_fisicas):
                    continue
                colocado = rect
                break
            if not colocado:
                omitidas += 1
                continue
            ocupados.append(colocado)
            yc = (colocado[1] + colocado[3]) / 2
            tinta = _texto_numero(valor, decimales)
            recuadro = _texto(f"dato_{k_dato}", "dato", _celdas_numero(valor, decimales),
                              colocado[0], yc, tinta, editable="numero",
                              clave_edicion=f"dato_{k_dato}", valor_edicion=tinta)
            datos_diseno.append({"valor": valor, "recuadro": list(recuadro)})
            textos_diseno.append(_texto_diseno("dato", _rotulo_simple(tinta), recuadro))
        if omitidas:
            avisos.append(
                f"{omitidas} valor(es) anotado(s) junto a la curva no entraron en la lámina "
                "sin pisar una curva u otro número y se omitieron."
            )
        print(f"[STL] Valores junto a la curva: {len(datos_diseno)} escritos, "
              f"{omitidas} omitidos.", flush=True)

    # ================================================================
    # 12. DIBUJAR TODAS LAS SERIES
    # ================================================================
    # Cada serie usa una textura distinta (sólida / rayada / punteada) para
    # que, con más de una curva en la misma placa, se puedan distinguir al
    # tacto — antes todas se dibujaban idénticas pese a que el segmentador
    # ya avisa que "cada serie necesita su propia textura".

    # La textura va por la posición ORIGINAL de la serie (no por el orden
    # entre las que se pudieron dibujar), así coincide con la leyenda.
    for indice, puntos_stl in zip(indices_fisicas, series_fisicas):
        estilo = _ESTILOS_SERIE[indice % len(_ESTILOS_SERIE)]
        desde = len(piezas)
        agregar_funcion(
            piezas,
            puntos_stl,
            diametro=estilo["diametro"],
            altura=estilo["altura"],
            patron=estilo["patron"],
        )
        sid = ids_series[indice]
        _registrar(f"serie_{sid}", "serie", desde, texto=nombres_series[indice],
                   textura=estilo["nombre"], ocultable=len(series_fisicas) > 1, editable="texto",
                   clave_edicion=f"serie_{sid}", valor_edicion=nombres_series[indice])
        print(
            f"[STL] Serie {indice + 1} ('{nombres_series[indice]}') dibujada con "
            f"textura '{estilo['nombre']}' (patrón {estilo['patron']}).",
            flush=True,
        )

    # ================================================================
    # 12b. LEYENDA (abajo, en el espacio que sobra)
    # ================================================================
    # Primero las texturas de las series (con 2 o más), después cada texto
    # abreviado: "A  texto completo". Renglones de INTERLINEA_BRAILLE.
    leyenda_diseno, abreviaturas_diseno = [], []
    recuadros_entrada = {}
    inicio_entrada = {}
    for r, renglon in enumerate(renglones_leyenda):
        y_fila = y_leyenda - r * INTERLINEA_BRAILLE
        for pieza in renglon:
            inicio_entrada.setdefault(pieza["entrada"], len(piezas))
            x = MARGEN_PLACA + pieza["dx"]
            if pieza["serie"] is not None:
                estilo = _ESTILOS_SERIE[pieza["serie"] % len(_ESTILOS_SERIE)]
                r_tapa = estilo["diametro"] / 2   # la tapa redondeada no sale del margen
                agregar_funcion(
                    piezas, [(x + r_tapa, y_fila), (x + LEYENDA_MUESTRA - r_tapa, y_fila)],
                    diametro=estilo["diametro"], altura=estilo["altura"], patron=estilo["patron"],
                )
            if pieza["prefijo"]:
                _dibujar_celdas(piezas, pieza["prefijo"], x + BORDE_A_CELDA, y_fila)
            _dibujar_celdas(piezas, pieza["celdas"], x + pieza["sangria"] + BORDE_A_CELDA, y_fila)
            fin = x + pieza["sangria"] + _ancho_celdas(len(pieza["celdas"]))
            caja = [x, y_fila - ALTURA_FILA_BRAILLE / 2, fin, y_fila + ALTURA_FILA_BRAILLE / 2]
            previa = recuadros_entrada.get(pieza["entrada"])
            recuadros_entrada[pieza["entrada"]] = caja if previa is None else [
                min(previa[0], caja[0]), min(previa[1], caja[1]),
                max(previa[2], caja[2]), max(previa[3], caja[3])]

    # las piezas de una entrada son consecutivas (renglones seguidos)
    orden = sorted(inicio_entrada)
    for pos, k in enumerate(orden):
        desde = inicio_entrada[k]
        hasta = inicio_entrada[orden[pos + 1]] if pos + 1 < len(orden) else len(piezas)
        if k < len(series_leyenda):
            indice, nombre = series_leyenda[k]
            sid = ids_series[indice]
            info = {"id": f"leyenda_serie_{sid}", "texto": f"{nombre} (textura "
                    f"{_ESTILOS_SERIE[indice % len(_ESTILOS_SERIE)]['nombre']})",
                    "ocultable": True, "editable": "texto",
                    "clave_edicion": f"serie_{sid}", "valor_edicion": nombre}
        else:
            a = leyenda_letras[k - len(series_leyenda)]
            editable = bool(a.get("clave")) and not a["clave"].startswith("num_")
            info = {"id": f"leyenda_{a['identificador']}",
                    "texto": f"{a['identificador']}: {a['texto']}",
                    "editable": "texto" if editable else None,
                    "clave_edicion": a.get("clave") if editable else None, "valor_edicion": a["texto"]}
        elementos.append({"clase": "leyenda", "desde": desde, "hasta": hasta,
                          "ocultable": False, "movible": False, "editable": None,
                          "recuadro": recuadros_entrada[k], **info})

    for k, (indice, nombre) in enumerate(series_leyenda):
        if k in recuadros_entrada:
            x0, y0, _, y1 = recuadros_entrada[k]
            y_muestra = (y0 + y1) / 2 if y1 - y0 <= ALTURA_FILA_BRAILLE + 1e-6 else y1 - ALTURA_FILA_BRAILLE / 2
            leyenda_diseno.append({
                "indice": indice,
                "nombre": nombre,
                "textura": _ESTILOS_SERIE[indice % len(_ESTILOS_SERIE)]["nombre"],
                "muestra": [(x0, y_muestra), (x0 + LEYENDA_MUESTRA, y_muestra)],
                "recuadro": recuadros_entrada[k],
                "recortado": False,
            })
    for k, a in enumerate(leyenda_letras, start=len(series_leyenda)):
        abreviaturas_diseno.append({
            "identificador": a["identificador"],
            "texto": a["texto"],
            "donde": a["donde"],
            "recuadro": recuadros_entrada.get(k),   # None si no entró en la placa
        })
    if renglones_leyenda:
        print(f"[STL] Leyenda abajo: {len(leyenda_diseno)} textura(s), "
              f"{len(abreviaturas_diseno)} texto(s) abreviado(s), {len(renglones_leyenda)} renglón(es).",
              flush=True)

    if diseno is not None:
        diseno.update({
            "placa": {"ancho_mm": dim_x, "alto_mm": dim_y,
                      "margen_superior_mm": margen_superior, "chaflan_mm": chaflan,
                      "contorno": [list(p) for p in contorno_placa(dim_x, dim_y, chaflan)]},
            "area": {"izquierda": izquierda, "abajo": abajo, "ancho": ancho_plot, "alto": alto_plot},
            # zona del gráfico (arriba, con títulos y rótulos) y de la leyenda
            # (abajo): [x0, y0, x1, y1] en mm
            "zonas": {
                "grafico": [MARGEN_PLACA, abajo - bajo_plot, dim_x - MARGEN_PLACA, y_tope],
                "leyenda": [MARGEN_PLACA, MARGEN_PLACA, dim_x - MARGEN_PLACA,
                            abajo - bajo_plot - CLEARANCE_BRAILLE],
            },
            # en el espacio del eje: log10(valor) si la escala es "log"
            "dominio": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
                        "calibrado": calibrados,
                        "escala_x": "log" if log_x else "lineal",
                        "escala_y": "log" if log_y else "lineal"},
            "ejes": {"x": [eje_x_inicio, eje_x_fin], "y": [eje_y_inicio, eje_y_fin]},
            "series": [{
                "indice": indice,
                "nombre": nombres_series[indice],
                "nombre_leido": bool((nombres_leidos[indice] or "").strip()),
                "textura": _ESTILOS_SERIE[indice % len(_ESTILOS_SERIE)]["nombre"],
                "puntos_mm": [tuple(p) for p in puntos_stl],
            } for indice, puntos_stl in zip(indices_fisicas, series_fisicas)],
            "leyenda": leyenda_diseno,
            # textos que no entraban en su lugar: en la placa dicen
            # "identificador" y en la leyenda de abajo, el texto completo
            "abreviaturas": abreviaturas_diseno,
            # cada texto escrito (títulos, categorías): qué dice y dónde
            "textos": textos_diseno,
            "etiquetas_dato": datos_diseno,
            "avisos": avisos,
            # vista previa: cada cosa en relieve con sus primitivas (mm)
            "elementos": [{
                **{k: v for k, v in e.items() if k not in ("desde", "hasta")},
                "primitivas": [_primitiva_json(p) for p in piezas[e["desde"]:e["hasta"]]],
            } for e in elementos],
            "ocultos": ocultos_info,
        })

    if archivo_salida is None:      # solo vista previa
        return None

    # ================================================================
    # 13. ENSAMBLAR Y EXPORTAR
    # ================================================================
    # Una sola concatenación de la placa base con todas las piezas juntas
    # (no unión booleana una por una): ver _ensamblar más arriba.

    print(f"[STL] Ensamblando {len(piezas)} piezas en relieve...", flush=True)
    modelo = _malla_prisma(contorno_placa(dim_x, dim_y, chaflan), BASE_THICKNESS)
    triangulos = _ensamblar(modelo, piezas)
    _guardar_stl(triangulos, archivo_salida)

    print(
        f"[STL] Modelo táctil generado: {archivo_salida}",
        flush=True
    )

    return triangulos
if __name__ == "__main__":
    generar_modelo_bana()
