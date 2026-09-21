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


def _truncar_para_ancho(texto, ancho_disponible, maximo_absoluto=40):
    """Recorta un texto para que su versión Braille quepa en el ancho
    disponible (evita que un título largo se salga de la placa o dispare el
    número de figuras/uniones del STL). Devuelve (texto_recortado, se_recortó).
    """
    max_celdas = max(0, min(maximo_absoluto, int(ancho_disponible // CELDA_PITCH)))
    if len(texto) <= max_celdas:
        return texto, False
    return texto[:max_celdas], True


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
    archivo_salida="grafica_tactil.stl"
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

        for serie in series:

            puntos = serie.get("puntos", [])

            if isinstance(puntos, list) and len(puntos) >= 2:
                series_puntos.append(puntos)

    elif puntos_legacy:

        series_puntos = [puntos_legacy]

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
    # 6. ÁREA FÍSICA DEL GRÁFICO
    # ================================================================
    # Los márgenes base alcanzan para los ejes, las marcas y sus números.
    # Si además hay título de eje X, de eje Y o título del gráfico (leídos
    # por OCR), se reserva una fila extra de Braille por cada uno —
    # respetando la separación BANA entre elementos no relacionados— para
    # que ese texto no quede pegado a los números ni se salga de la placa.

    fila_reservada = ALTURA_FILA_BRAILLE + CLEARANCE_BRAILLE

    izquierda = 55.0
    derecha = 15.0
    abajo = 25.0 + (fila_reservada if titulo_eje_x_txt else 0.0)

    filas_arriba = int(bool(titulo_grafico)) + int(bool(titulo_eje_y_txt))
    arriba = 15.0 + filas_arriba * fila_reservada

    ancho_plot = dim_x - izquierda - derecha
    alto_plot = dim_y - abajo - arriba

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
    # 11. TICKS
    # ================================================================
    # Antes las marcas eran siempre 5 valores equiespaciados entre x_min y
    # x_max (derivados del borde del rectángulo del gráfico), así que casi
    # nunca coincidían con los números que realmente estaban impresos en la
    # gráfica original. Ahora, si el segmentador leyó al menos 2 etiquetas
    # numéricas reales por eje, se usan ESAS —mismo valor que vio el OCR—;
    # solo se cae a marcas sintéticas equiespaciadas si no hay etiquetas
    # reales suficientes (p. ej. calibración hecha con las etiquetas de dato
    # pegadas a la curva, sin números de eje legibles).
    NUM_TICKS_SINTETICOS = 5
    MAX_TICKS_POR_EJE = 12  # límite defensivo: no cubrir la placa de números

    valores_x, valores_y = [], []
    if calibrados:
        etiquetas_x_json = textos.get("etiquetas_eje_x") or []
        etiquetas_y_json = textos.get("etiquetas_eje_y") or []

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

    decimales_x = _decimales_necesarios(valores_x) if valores_x else 0
    decimales_y = _decimales_necesarios(valores_y) if valores_y else 0

    for valor_x in valores_x:
        x_fis = min(max(valor_a_fisico(valor_x, y_min)[0], izquierda), izquierda + ancho_plot)

        agregar_segmento_relieve(
            piezas, (x_fis, abajo - 3), (x_fis, abajo + 3), 1.5, RELIEVE_TICK
        )

        ancho_x = _ancho_numero(valor_x, decimales_x)
        inicio_x = min(max(3.0, x_fis - ancho_x / 2), dim_x - ancho_x - 3.0)
        agregar_numero_braille(
            piezas, valor_x, inicio_x, abajo - CLEARANCE_BRAILLE, decimales_x
        )

    for valor_y in valores_y:
        y_fis = min(max(valor_a_fisico(x_min, valor_y)[1], abajo), abajo + alto_plot)

        agregar_segmento_relieve(
            piezas, (izquierda - 3, y_fis), (izquierda + 3, y_fis), 1.5, RELIEVE_TICK
        )

        ancho_y = _ancho_numero(valor_y, decimales_y)
        inicio_y = max(3.0, izquierda - CLEARANCE_BRAILLE - ancho_y)
        agregar_numero_braille(piezas, valor_y, inicio_y, y_fis, decimales_y)

    # ================================================================
    # 11b. TÍTULOS EN BRAILLE (título del gráfico, nombre de cada eje)
    # ================================================================
    # Antes ninguno de estos textos —que el segmentador sí extrae por
    # OCR— llegaba a la placa: solo se dibujaban números y la curva, sin
    # ningún contexto. Se recortan al ancho disponible en vez de desbordar
    # la placa o multiplicar sin límite las uniones del STL.
    ancho_disponible_titulo = dim_x - izquierda - derecha

    if titulo_eje_x_txt:
        texto, recortado = _truncar_para_ancho(
            _sanear_texto_braille(titulo_eje_x_txt), ancho_disponible_titulo
        )
        y_titulo_x = (abajo - CLEARANCE_BRAILLE) - fila_reservada + ALTURA_FILA_BRAILLE / 2
        agregar_texto_braille(piezas, texto, izquierda, y_titulo_x)
        if recortado:
            print(f"[STL] Título del eje X recortado para que quepa en la placa.", flush=True)

    fila_arriba = 0
    for etiqueta, texto_crudo in (("eje Y", titulo_eje_y_txt), ("gráfico", titulo_grafico)):
        if not texto_crudo:
            continue
        texto, recortado = _truncar_para_ancho(
            _sanear_texto_braille(texto_crudo), ancho_disponible_titulo
        )
        y_fila = (dim_y - arriba) + CLEARANCE_BRAILLE + ALTURA_FILA_BRAILLE / 2 \
            + fila_arriba * fila_reservada
        agregar_texto_braille(piezas, texto, izquierda, y_fila)
        if recortado:
            print(f"[STL] Título del {etiqueta} recortado para que quepa en la placa.", flush=True)
        fila_arriba += 1

    # ================================================================
    # 12. DIBUJAR TODAS LAS SERIES
    # ================================================================
    # Cada serie usa una textura distinta (sólida / rayada / punteada) para
    # que, con más de una curva en la misma placa, se puedan distinguir al
    # tacto — antes todas se dibujaban idénticas pese a que el segmentador
    # ya avisa que "cada serie necesita su propia textura".

    for indice, puntos_stl in enumerate(series_fisicas):
        estilo = _ESTILOS_SERIE[indice % len(_ESTILOS_SERIE)]
        agregar_funcion(
            piezas,
            puntos_stl,
            diametro=estilo["diametro"],
            altura=estilo["altura"],
            patron=estilo["patron"],
        )
        print(
            f"[STL] Serie {indice + 1} dibujada con textura '{estilo['nombre']}' "
            f"(patrón {estilo['patron']}).",
            flush=True,
        )

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
