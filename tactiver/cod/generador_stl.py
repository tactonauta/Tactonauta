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

import math
import unicodedata
from math import atan2, degrees, hypot

import numpy as np
from stl import mesh

# =============================================================================
# CONSTANTES
# =============================================================================
BASE_THICKNESS = 0.2
ESPACIADO_BRAILLE = 2.4
DIAM_PUNTO_BRAILLE = 1.4
CELDA_PITCH = 6.2
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

# Patrón "rayado" (serie 2): largo del tramo dibujado y del hueco, en mm.
RAYA_LARGO = 6.0
RAYA_HUECO = 3.5

# Patrón "punteado" (serie 3): separación mínima entre bultos, en mm, para
# que no se junten hasta parecer una línea continua otra vez.
PUNTEADO_ESPACIADO = 5.0

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
# Punto decimal Nemeth. A diferencia de las letras/dígitos de arriba (tabla
# estándar), este signo se agregó para poder mostrar valores no enteros
# (ejes 0-1 de accuracy/loss, por ejemplo) y NO se verificó contra una
# fuente Nemeth impresa. Antes de usar láminas con decimales en un
# contexto educativo real, pedir a un transcriptor Braille certificado que
# confirme este patrón de puntos (ver también la nota equivalente sobre
# alturas de relieve en AplicarFormato/Bloque3).
BRAILLE["punto"] = [4, 6]
for _n, _letra in enumerate(_ORDEN_DIGITOS, start=1):
    _digito = str(_n % 10)
    BRAILLE[_digito] = sorted(_DESPLAZAMIENTO_NEMETH[d] for d in LETRAS[_letra])


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
    """
    piezas.append(_malla_esfera(DIAM_PUNTO_BRAILLE / 2, cx=cx, cy=cy, cz=BASE_THICKNESS))


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

    angulo = degrees(atan2(dy, dx))

    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    z0 = BASE_THICKNESS - SOLAPE
    altura_real = float(altura) + SOLAPE

    cuerpo_local = _malla_caja(largo, float(diametro), altura_real, cx=0.0, cy=0.0, z0=0.0)
    cuerpo = _trasladar(_rotar_z(cuerpo_local, angulo), mx, my, z0)

    tapa0 = _malla_cilindro(float(diametro) / 2, altura_real, cx=x0, cy=y0, z0=z0)
    tapa1 = _malla_cilindro(float(diametro) / 2, altura_real, cx=x1, cy=y1, z0=z0)

    piezas.append(cuerpo)
    piezas.append(tapa0)
    piezas.append(tapa1)


def agregar_punto_relieve(piezas, punto, diametro, altura):
    """Un bulto redondo aislado en el punto dado (usado por el patrón
    "punteado" para distinguir una tercera serie al tacto). Ver `SOLAPE`
    en `agregar_segmento_relieve`."""
    x, y = float(punto[0]), float(punto[1])
    piezas.append(_malla_cilindro(
        float(diametro) / 2, float(altura) + SOLAPE,
        cx=x, cy=y, z0=BASE_THICKNESS - SOLAPE,
    ))


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
    {"nombre": "sólida", "patron": "solido", "diametro": DIAM_LINEA, "altura": RELIEVE_LINEA},
    {"nombre": "rayada", "patron": "rayado", "diametro": DIAM_LINEA * 0.85, "altura": RELIEVE_LINEA + 0.4},
    {"nombre": "punteada", "patron": "punteado", "diametro": DIAM_LINEA * 1.3, "altura": RELIEVE_LINEA - 0.3},
]

# Textura de cada eje: distinta entre sí y de las series de datos (que se
# quedan con la línea sólida, la más alta — "dato > eje > marca de escala").
# Antes ambos ejes eran una barra lisa idéntica entre sí y de la misma
# familia de forma que una curva sólida; con esto un eje ya no se confunde
# al tacto ni con el otro eje ni con un dato. El espaciado es más fino que
# el de las series para que se sientan como una guía, no como un dato.
EJE_X_ESTILO = {
    "patron": "rayado", "diametro": 2.0, "altura": RELIEVE_EJE,
    "raya_largo": 3.0, "raya_hueco": 2.0, "punteado_espaciado": PUNTEADO_ESPACIADO,
}
EJE_Y_ESTILO = {
    "patron": "punteado", "diametro": 2.0, "altura": RELIEVE_EJE,
    "raya_largo": RAYA_LARGO, "raya_hueco": RAYA_HUECO, "punteado_espaciado": 4.0,
}


# Leyenda (solo con 2+ series): cada entrada es una muestra corta de la
# textura de la serie + su nombre en Braille, en filas arriba del gráfico.
LEYENDA_MUESTRA = 16.0   # largo de la muestra de textura (mm)
LEYENDA_HUECO = 4.0      # de la muestra al texto
LEYENDA_ENTRE = 8.0      # entre dos entradas de la misma fila
LEYENDA_MARGEN = 8.0     # borde izquierdo/derecho de la placa


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
    """Celdas Braille de un texto con letras y números: cada tramo de
    dígitos lleva delante el signo numeral (a diferencia de
    `_sanear_texto_braille`, pensada para títulos). Lo que no está en la
    tabla (puntuación, símbolos) se omite en vez de dejar celdas vacías."""
    texto = _sanear_texto_braille(texto)
    celdas = []
    anterior_digito = False
    for ch in texto:
        if ch.isdigit():
            if not anterior_digito:
                celdas.append("numeral")
            celdas.append(ch)
            anterior_digito = True
        elif ch in LETRAS or ch == " ":
            if ch == " " and (not celdas or celdas[-1] == " "):
                continue
            celdas.append(ch)
            anterior_digito = False
    while celdas and celdas[-1] == " ":
        celdas.pop()
    return celdas


def _recortar_celdas(celdas, maximo):
    """Recorta a `maximo` celdas cortando en un espacio si se puede: partir
    una palabra, y sobre todo un número ("2023" -> "202"), cambia lo que dice."""
    if len(celdas) <= maximo:
        return celdas
    celdas = celdas[:max(0, maximo)]
    if " " in celdas:
        celdas = celdas[:len(celdas) - celdas[::-1].index(" ")]
    while celdas and celdas[-1] in (" ", "numeral"):
        celdas.pop()
    return celdas


def _ancho_entrada(celdas):
    return LEYENDA_MUESTRA + LEYENDA_HUECO + len(celdas) * CELDA_PITCH


def _distribuir_leyenda(nombres, ancho_disponible):
    """Reparte las entradas de la leyenda en filas, en orden, pasando a una
    fila nueva cuando la siguiente no entra completa: se prefiere gastar una
    fila más que recortar un nombre. Solo se recorta un nombre que no entra
    ni ocupando una fila entera. Devuelve una lista de filas; cada fila es
    una lista de (índice de serie, celdas Braille)."""
    max_celdas = int((ancho_disponible - LEYENDA_MUESTRA - LEYENDA_HUECO) // CELDA_PITCH)
    filas, ocupado = [], None
    for i, nombre in enumerate(nombres):
        celdas = _recortar_celdas(_celdas_braille_mixto(nombre), max_celdas)
        ancho = _ancho_entrada(celdas)
        if filas and ocupado + LEYENDA_ENTRE + ancho <= ancho_disponible:
            filas[-1].append((i, celdas))
            ocupado += LEYENDA_ENTRE + ancho
        else:
            filas.append([(i, celdas)])
            ocupado = ancho
    return filas


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
    """

    if not puntos or len(puntos) < 2:
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
    trozos = [base] + [p for p in piezas if p is not None and len(p)]
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


def generar_modelo_desde_recta(
    datos_segmentador,
    dim_x=210.0,
    dim_y=148.0,
    archivo_salida="grafica_tactil.stl",
    incluir_etiquetas=False,
    incluir_leyenda=False,
    diseno=None,
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
    en su lugar, corridos lo justo para no pisar ninguna curva.

    `incluir_leyenda` (por defecto False, desactivada por ahora: en A5 le
    quita demasiado alto al gráfico): con 2 o más series, agrega arriba
    una leyenda con una muestra de la textura de cada serie y su nombre en
    Braille. Sin ella, qué textura es cada serie lo dice la narración.

    `diseno`: si se pasa un dict, se llena con dónde quedó cada cosa en la
    placa (mm, origen abajo a la izquierda) y la escala usada. Lo usa
    narracion.py para que un programa de narración sepa qué hay bajo el dedo.
    """

    if dim_x < 130 or dim_y < 90:
        raise ValueError(
            "La placa debe medir al menos 130 x 90 mm."
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
    titulo_grafico = (textos.get("titulo") or "").strip()
    titulo_eje_x_txt = (textos.get("titulo_eje_x") or "").strip()
    titulo_eje_y_txt = (textos.get("titulo_eje_y") or "").strip()

    # ================================================================
    # 2. OBTENER LAS SERIES
    # ================================================================

    if series:

        series_puntos = []
        nombres_leidos = []

        for serie in series:

            puntos = serie.get("puntos", [])

            if isinstance(puntos, list) and len(puntos) >= 2:
                series_puntos.append(puntos)
                nombres_leidos.append(serie.get("nombre"))

    elif puntos_legacy:

        series_puntos = [puntos_legacy]
        nombres_leidos = [None]

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
    # 6b. DISTRIBUCIÓN EN LA PLACA: el gráfico centrado
    # ================================================================
    # El recuadro del gráfico (lo que encierran los ejes) queda en el CENTRO
    # de la placa: mismo espacio a izquierda y derecha, y arriba y abajo.
    # Ese espacio es el mayor de lo que hay que escribir de cada lado:
    #   izquierda: números del eje Y + separación BANA hasta el eje;
    #   derecha:   la mitad del último rótulo del eje X (va centrado en su
    #              marca y sobresale del gráfico);
    #   abajo:     fila de rótulos del eje X + fila del título del eje X;
    #   arriba:    una fila por título (eje Y y gráfico) y por fila de leyenda;
    # más MARGEN_PLACA hasta el borde. (Antes se centraba el conjunto con
    # los números incluidos, y como los del eje Y están solo a la izquierda,
    # el gráfico quedaba corrido hacia la derecha.)
    #
    # Eje X de categorías ("Ene", "Feb"...): el valor i del eje es la
    # categoría i, y en la placa se escribe su nombre en vez del número.

    fila_reservada = ALTURA_FILA_BRAILLE + CLEARANCE_BRAILLE
    categorias_x = [str(c) for c in (textos.get("categorias_x") or [])]

    def _celdas_rotulo_x(valor):
        i = int(round(valor))
        if categorias_x and abs(valor - i) <= 0.25 and 0 <= i < len(categorias_x):
            return _celdas_braille_mixto(categorias_x[i])
        return _celdas_numero(valor, decimales_x)

    celdas_x = [_celdas_rotulo_x(v) for v in reales_x]
    celdas_y = [_celdas_numero(v, decimales_y) for v in reales_y]
    anchos_x = [_ancho_celdas(len(c)) for c in celdas_x]
    anchos_y = [_ancho_celdas(len(c)) for c in celdas_y]

    ancho_numeros_y = max(anchos_y, default=0.0)
    bloque_izquierdo = (ancho_numeros_y + CLEARANCE_BRAILLE) if valores_y else 6.0
    sobresale_derecha = anchos_x[-1] / 2 if anchos_x else 3.0
    lateral = MARGEN_PLACA + max(bloque_izquierdo, sobresale_derecha)

    filas_arriba = int(bool(titulo_grafico)) + int(bool(titulo_eje_y_txt))

    # Leyenda: filas propias por encima de los títulos (ver sección 12b).
    nombres_series = [nombre_de_serie(n, i) for i, n in enumerate(nombres_leidos)]
    filas_leyenda = []
    if incluir_leyenda and len(series_puntos) > 1:
        filas_leyenda = _distribuir_leyenda(nombres_series, dim_x - 2 * LEYENDA_MARGEN)

    bloque_abajo = ALTURA_FILA_BRAILLE / 2 + CLEARANCE_BRAILLE \
        + (fila_reservada if titulo_eje_x_txt else 0.0)
    # El número más alto del eje Y va centrado en el borde superior del
    # gráfico: sobresale media fila aunque no haya títulos.
    bloque_arriba = max(ALTURA_FILA_BRAILLE / 2,
                        (filas_arriba + len(filas_leyenda)) * fila_reservada)
    vertical = MARGEN_PLACA + max(bloque_abajo, bloque_arriba)

    izquierda = derecha = lateral
    abajo = arriba = vertical

    ancho_plot = dim_x - izquierda - derecha
    alto_plot = dim_y - abajo - arriba

    if ancho_plot <= 0 or alto_plot <= 0:
        raise ValueError(
            "Las dimensiones de la placa no dejan espacio suficiente "
            "para el gráfico."
        )

    # ================================================================
    # 7. CONVERSIÓN PIXEL -> VALOR -> MILÍMETROS    # ================================================================
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

    modelo = _malla_caja(dim_x, dim_y, BASE_THICKNESS, cx=dim_x / 2, cy=dim_y / 2, z0=0.0)

    # Todas las piezas en relieve (ejes, ticks, números y texto Braille,
    # curvas) se acumulan acá y se sueldan a la placa base de una sola vez
    # al final (ver _ensamblar) en vez de unirse una por una.
    piezas = []

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
        dim_y - arriba
    )

    # Cada eje con su propia textura (rayado el X, punteado el Y) para que
    # no se confundan entre sí ni con una curva de datos (que se queda con
    # la línea sólida): ver EJE_X_ESTILO / EJE_Y_ESTILO.
    _agregar_eje(piezas, eje_x_inicio, eje_x_fin, EJE_X_ESTILO)
    _agregar_eje(piezas, eje_y_inicio, eje_y_fin, EJE_Y_ESTILO)

    # ================================================================
    # 11. MARCAS Y NÚMEROS DE LOS EJES
    # ================================================================
    # Todas las marcas van en relieve; los números solo los que entran sin
    # tocarse (ver _indices_legibles): en el eje Y, una fila de Braille mide
    # ~6 mm, y 8 números en un eje de 45 mm quedaban pegados.
    avisos = []
    x_marcas = [min(max(valor_a_fisico(v, y_min)[0], izquierda), izquierda + ancho_plot)
                for v in valores_x]
    y_marcas = [min(max(valor_a_fisico(x_min, v)[1], abajo), abajo + alto_plot)
                for v in valores_y]

    for x_fis in x_marcas:
        agregar_segmento_relieve(
            piezas, (x_fis, abajo - 3), (x_fis, abajo + 3), 1.5, RELIEVE_TICK
        )
    for y_fis in y_marcas:
        agregar_segmento_relieve(
            piezas, (izquierda - 3, y_fis), (izquierda + 3, y_fis), 1.5, RELIEVE_TICK
        )

    if incluir_etiquetas:
        con_numero_x = _indices_legibles(x_marcas, anchos_x, CELDA_PITCH)
        for i in con_numero_x:
            inicio_x = min(max(MARGEN_PLACA, x_marcas[i] - anchos_x[i] / 2),
                           dim_x - MARGEN_PLACA - anchos_x[i])
            _dibujar_celdas(piezas, celdas_x[i], inicio_x + BORDE_A_CELDA,
                            abajo - CLEARANCE_BRAILLE)

        con_numero_y = _indices_legibles(
            y_marcas, [ALTURA_FILA_BRAILLE] * len(y_marcas), ESPACIADO_BRAILLE)
        for i in con_numero_y:
            inicio_y = max(MARGEN_PLACA, izquierda - CLEARANCE_BRAILLE - anchos_y[i])
            _dibujar_celdas(piezas, celdas_y[i], inicio_y + BORDE_A_CELDA, y_marcas[i])

        for eje, total, escritos in (("X", len(valores_x), len(con_numero_x)),
                                     ("Y", len(valores_y), len(con_numero_y))):
            if escritos < total:
                avisos.append(
                    f"Eje {eje}: se escribieron {escritos} de {total} números en Braille "
                    "para que no se toquen (todas las marcas siguen en relieve)."
                )

    # ================================================================
    # 11b. TÍTULOS EN BRAILLE (título del gráfico, nombre de cada eje)
    # ================================================================
    # Título del gráfico centrado arriba de todo; título del eje Y en la
    # fila de abajo, alineado con el eje; título del eje X centrado bajo el
    # gráfico. Si un texto no entra en el ancho de la placa se corta en un
    # espacio (no a mitad de palabra ni de número).
    # (con incluir_etiquetas=False se saltea todo el bloque: ya se reservó
    # el margen en base a si había texto, así que la placa no se reacomoda,
    # solo queda ese espacio en blanco.)
    if incluir_etiquetas:
        max_celdas = int((dim_x - 2 * MARGEN_PLACA - ESPACIADO_BRAILLE) // CELDA_PITCH)

        def _titulo(texto_crudo, nombre):
            completas = _celdas_braille_mixto(texto_crudo)
            celdas = _recortar_celdas(completas, max_celdas)
            if len(celdas) < len(completas):
                avisos.append(f"El {nombre} se recortó para que quepa en la placa: "
                              "el texto completo está en la descripción narrada.")
            return celdas

        def _x_centrada(celdas, centro):
            ancho = _ancho_celdas(len(celdas))
            x0 = min(max(MARGEN_PLACA, centro - ancho / 2), dim_x - MARGEN_PLACA - ancho)
            return x0 + BORDE_A_CELDA

        if titulo_eje_x_txt:
            celdas = _titulo(titulo_eje_x_txt, "título del eje X")
            _dibujar_celdas(piezas, celdas, _x_centrada(celdas, izquierda + ancho_plot / 2),
                            abajo - CLEARANCE_BRAILLE - fila_reservada)

        fila_arriba = 0
        for nombre, texto_crudo in (("título del eje Y", titulo_eje_y_txt),
                                    ("título del gráfico", titulo_grafico)):
            if not texto_crudo:
                continue
            celdas = _titulo(texto_crudo, nombre)
            y_fila = (dim_y - arriba) + CLEARANCE_BRAILLE + ALTURA_FILA_BRAILLE / 2 \
                + fila_arriba * fila_reservada
            if nombre == "título del eje Y":
                x_primera = max(MARGEN_PLACA, izquierda - ancho_numeros_y) + BORDE_A_CELDA
                x_primera = min(x_primera, dim_x - MARGEN_PLACA - _ancho_celdas(len(celdas))
                                + BORDE_A_CELDA)
            else:
                x_primera = _x_centrada(celdas, dim_x / 2)
            _dibujar_celdas(piezas, celdas, x_primera, y_fila)
            fila_arriba += 1

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
        for e in etiquetas_dato:
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
            agregar_numero_braille(piezas, valor, colocado[0] + BORDE_A_CELDA, yc, decimales)
            datos_diseno.append({"valor": valor, "recuadro": list(colocado)})
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
        agregar_funcion(
            piezas,
            puntos_stl,
            diametro=estilo["diametro"],
            altura=estilo["altura"],
            patron=estilo["patron"],
        )
        print(
            f"[STL] Serie {indice + 1} ('{nombres_series[indice]}') dibujada con "
            f"textura '{estilo['nombre']}' (patrón {estilo['patron']}).",
            flush=True,
        )

    # ================================================================
    # 12b. LEYENDA (solo con 2 o más series)
    # ================================================================
    # Muestra corta de la textura de cada serie + su nombre en Braille, en
    # filas arriba de los títulos. Con una sola curva no hace falta: no hay
    # nada que distinguir.
    leyenda_diseno = []
    for fila, entradas in enumerate(filas_leyenda):
        y_fila = (dim_y - arriba) + CLEARANCE_BRAILLE + ALTURA_FILA_BRAILLE / 2             + (filas_arriba + fila) * fila_reservada
        x = LEYENDA_MARGEN
        for indice, celdas in entradas:
            estilo = _ESTILOS_SERIE[indice % len(_ESTILOS_SERIE)]
            muestra = [(x, y_fila), (x + LEYENDA_MUESTRA, y_fila)]
            agregar_funcion(
                piezas, muestra,
                diametro=estilo["diametro"], altura=estilo["altura"], patron=estilo["patron"],
            )
            x_texto = x + LEYENDA_MUESTRA + LEYENDA_HUECO + ESPACIADO_BRAILLE / 2
            for k, celda in enumerate(celdas):
                agregar_caracter_braille(piezas, celda, x_texto + k * CELDA_PITCH, y_fila)
            fin_texto = x_texto + max(0, len(celdas) - 1) * CELDA_PITCH + ESPACIADO_BRAILLE / 2
            leyenda_diseno.append({
                "indice": indice,
                "nombre": nombres_series[indice],
                "textura": estilo["nombre"],
                "muestra": muestra,
                "recuadro": [x, y_fila - ALTURA_FILA_BRAILLE / 2, fin_texto, y_fila + ALTURA_FILA_BRAILLE / 2],
                "recortado": len(celdas) < len(_celdas_braille_mixto(nombres_series[indice])),
            })
            x = fin_texto + LEYENDA_ENTRE
    if leyenda_diseno:
        print(f"[STL] Leyenda: {len(leyenda_diseno)} entradas en {len(filas_leyenda)} fila(s).", flush=True)

    if diseno is not None:
        diseno.update({
            "placa": {"ancho_mm": dim_x, "alto_mm": dim_y},
            "area": {"izquierda": izquierda, "abajo": abajo, "ancho": ancho_plot, "alto": alto_plot},
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
            "etiquetas_dato": datos_diseno,
            "avisos": avisos,
        })

    # ================================================================
    # 13. ENSAMBLAR Y EXPORTAR
    # ================================================================
    # Una sola concatenación de la placa base con todas las piezas juntas
    # (no unión booleana una por una): ver _ensamblar más arriba.

    print(f"[STL] Ensamblando {len(piezas)} piezas en relieve...", flush=True)
    triangulos = _ensamblar(modelo, piezas)
    _guardar_stl(triangulos, archivo_salida)

    print(
        f"[STL] Modelo táctil generado: {archivo_salida}",
        flush=True
    )

    return triangulos
if __name__ == "__main__":
    generar_modelo_bana()
