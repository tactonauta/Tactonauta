"""Generador de placas táctiles STL para la Fase 1.

Este módulo reemplaza a la implementación anterior del generador STL y
convierte el Bloque 3 en la fuente de verdad del proyecto, manteniendo la
compatibilidad con la API y con el flujo de segmentación existente.
"""

import math
import unicodedata
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


def agregar_numero_braille(modelo, valor, cx, cy, decimales=0):
    """Coloca un número en formato Nemeth: signo menos si corresponde,
    indicador numeral, parte entera y, si `decimales` > 0, punto decimal
    Nemeth + parte decimal.

    Con `decimales=0` (el valor por defecto) el comportamiento es idéntico
    al de la versión anterior, que solo aceptaba enteros.
    """
    es_negativo, entero, frac = _formatear_valor_braille(valor, decimales)
    x = cx
    if es_negativo:
        modelo = agregar_caracter_braille(modelo, "menos", x, cy)
        x += CELDA_PITCH
    modelo = agregar_caracter_braille(modelo, "numeral", x, cy)
    x += CELDA_PITCH
    for ch in entero:
        modelo = agregar_caracter_braille(modelo, ch, x, cy)
        x += CELDA_PITCH
    if frac:
        modelo = agregar_caracter_braille(modelo, "punto", x, cy)
        x += CELDA_PITCH
        for ch in frac:
            modelo = agregar_caracter_braille(modelo, ch, x, cy)
            x += CELDA_PITCH
    return modelo


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


def agregar_segmento_relieve(modelo, p0, p1, diametro, altura):
    """Crea un tramo recto en relieve con dos extremos redondeados."""

    x0 = float(p0[0])
    y0 = float(p0[1])
    x1 = float(p1[0])
    y1 = float(p1[1])

    dx = x1 - x0
    dy = y1 - y0

    largo = hypot(dx, dy)

    if largo < 1e-6:
        return modelo

    angulo = degrees(atan2(dy, dx))

    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    # Crear el rectángulo centrado en el origen.
    cuerpo = (
        cq.Workplane("XY")
        .workplane(offset=BASE_THICKNESS)
        .box(
            largo,
            float(diametro),
            float(altura),
            centered=(True, True, False)
        )
    )

    # Girarlo alrededor del eje Z.
    cuerpo = cuerpo.rotate(
        (0, 0, 0),
        (0, 0, 1),
        angulo
    )

    # Llevarlo al punto medio del segmento.
    cuerpo = cuerpo.translate(
        (mx, my, 0)
    )

    # Extremos redondeados.
    tapa0 = (
        cq.Workplane("XY")
        .workplane(offset=BASE_THICKNESS)
        .center(x0, y0)
        .circle(float(diametro) / 2)
        .extrude(float(altura))
    )

    tapa1 = (
        cq.Workplane("XY")
        .workplane(offset=BASE_THICKNESS)
        .center(x1, y1)
        .circle(float(diametro) / 2)
        .extrude(float(altura))
    )

    return (
        modelo
        .union(cuerpo)
        .union(tapa0)
        .union(tapa1)
    )


def agregar_punto_relieve(modelo, punto, diametro, altura):
    """Un bulto redondo aislado en el punto dado (usado por el patrón
    "punteado" para distinguir una tercera serie al tacto)."""
    x, y = float(punto[0]), float(punto[1])
    bulto = (
        cq.Workplane("XY")
        .workplane(offset=BASE_THICKNESS)
        .center(x, y)
        .circle(float(diametro) / 2)
        .extrude(float(altura))
    )
    return modelo.union(bulto)


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


def agregar_funcion(
    modelo,
    puntos,
    diametro=DIAM_LINEA,
    altura=RELIEVE_LINEA,
    patron="solido",
):
    """
    Dibuja una polilínea en relieve.

    Los puntos deben estar ya convertidos a coordenadas físicas.

    `patron` distingue táctilmente varias series superpuestas en la misma
    placa (el segmentador ya avisa cuando detecta más de una: "cada serie
    necesita su propia textura"):
      - "solido"   -> línea continua (serie 1).
      - "rayado"   -> tramos discontinuos (serie 2).
      - "punteado" -> bultos redondos aislados, sin línea (serie 3).
    """

    if not puntos or len(puntos) < 2:
        return modelo

    if patron == "punteado":
        for p in _puntos_espaciados(puntos, PUNTEADO_ESPACIADO):
            modelo = agregar_punto_relieve(modelo, p, diametro, altura)
        return modelo

    for p0, p1 in zip(puntos[:-1], puntos[1:]):
        if patron == "rayado":
            for a, b in _dividir_en_rayas(p0, p1):
                modelo = agregar_segmento_relieve(modelo, a, b, diametro, altura)
        else:
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

        modelo = agregar_segmento_relieve(
            modelo, (x_fis, abajo - 3), (x_fis, abajo + 3), 1.5, RELIEVE_TICK
        )

        ancho_x = _ancho_numero(valor_x, decimales_x)
        inicio_x = min(max(3.0, x_fis - ancho_x / 2), dim_x - ancho_x - 3.0)
        modelo = agregar_numero_braille(
            modelo, valor_x, inicio_x, abajo - CLEARANCE_BRAILLE, decimales_x
        )

    for valor_y in valores_y:
        y_fis = min(max(valor_a_fisico(x_min, valor_y)[1], abajo), abajo + alto_plot)

        modelo = agregar_segmento_relieve(
            modelo, (izquierda - 3, y_fis), (izquierda + 3, y_fis), 1.5, RELIEVE_TICK
        )

        ancho_y = _ancho_numero(valor_y, decimales_y)
        inicio_y = max(3.0, izquierda - CLEARANCE_BRAILLE - ancho_y)
        modelo = agregar_numero_braille(modelo, valor_y, inicio_y, y_fis, decimales_y)

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
        modelo = agregar_texto_braille(modelo, texto, izquierda, y_titulo_x)
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
        modelo = agregar_texto_braille(modelo, texto, izquierda, y_fila)
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
        modelo = agregar_funcion(
            modelo,
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
    # 13. EXPORTAR
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
