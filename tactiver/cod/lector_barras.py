"""Lectura de gráficos de BARRAS (verticales u horizontales, de una o varias
series, y apiladas): qué barras hay, de qué serie es cada una (su color), a
qué categoría pertenece y cuánto vale.

Escribe lo mismo que segmentador.procesar_imagen (JSON, CSV y overlay, con el
mismo formato), para que la revisión del supervisor, la vista previa, el
STL, la narración y el QR funcionen igual que con las líneas. Lo propio de
las barras va en "tipo_grafica": "barras" y en "barras" (una por barra).

El JSON siempre describe el gráfico como si las barras fueran VERTICALES:
el eje X son las categorías (el valor i es la categoría i, como un eje de
categorías de una gráfica de líneas) y el eje Y los valores. Un gráfico de
barras horizontales se escribe así también ("orientacion": "horizontal"):
en la lámina todas las barras se leen de abajo hacia arriba.

Cómo se lee:
  1. Barras: zonas de tinta rellenas (sin el texto ni las líneas finas de los
     ejes y la rejilla), rectangulares y apoyadas en una misma línea de base.
     Las que se apoyan sobre otra del mismo ancho son partes de una barra
     apilada.
  2. Series: las barras se agrupan por color; el nombre de cada serie, de la
     leyenda (el texto al lado de la muestra de ese color).
  3. Categorías: el texto bajo cada grupo de barras (o a su izquierda, si son
     horizontales).
  4. Valores: el número escrito junto a la barra, si lo hay (es el dato
     exacto); si no, la altura de la barra con la escala del eje.
"""
import csv
import json
import math
import os
import re
import uuid

import cv2
import numpy as np

import segmentador as sg

VERSION = "1.0"

DIST_TINTA = 18          # distancia de color (Lab) al fondo para que sea tinta (las barras
                         # de colores pastel están a ~20)
DIST_SERIE = 22          # distancia de color entre barras de una misma serie
EXTENSION_MIN = 0.70     # área / recuadro: una barra es (casi) un rectángulo
R2_MIN = 0.99            # calidad mínima de la escala del eje para usarla


# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
def _color_lab(imagen_lab, x, y, w, h):
    """Color mediano del centro de un recuadro (sin el borde)."""
    mx, my = max(1, w // 4), max(1, h // 4)
    zona = imagen_lab[y + my:y + h - my, x + mx:x + w - mx]
    if zona.size == 0:
        zona = imagen_lab[y:y + h, x:x + w]
    return np.median(zona.reshape(-1, 3), axis=0)


def _moda(valores, tol):
    """El valor con más vecinos a menos de `tol`, y quiénes son esos vecinos."""
    if not valores:
        return None, []
    mejor = max(valores, key=lambda v: sum(abs(u - v) <= tol for u in valores))
    return mejor, [abs(u - mejor) <= tol for u in valores]


def _tokens(palabras, gris, lang):
    tokens = sg.tokens_de_palabras_pdf(palabras)
    if len(tokens) < 2 and sg.ocr_disponible():
        tokens = sg._tokens_ocr(gris, lang=lang)
    return tokens


def _unir(tokens):
    """Texto de varias palabras, en orden de lectura (renglón por renglón)."""
    if not tokens:
        return ""
    alto = float(np.median([t["ph"] for t in tokens])) or 1.0
    orden = sorted(tokens, key=lambda t: (round(t["centro_y"] / (0.7 * alto)), t["centro_x"]))
    return " ".join(t["texto"] for t in orden).strip()


def _caja(tokens):
    if not tokens:
        return None
    x0 = min(t["px"] for t in tokens); y0 = min(t["py"] for t in tokens)
    x1 = max(t["px"] + t["pw"] for t in tokens); y1 = max(t["py"] + t["ph"] for t in tokens)
    return {"px": int(x0), "py": int(y0), "pw": int(x1 - x0), "ph": int(y1 - y0)}


# ------------------------------------------------------------------
# 1. Barras
# ------------------------------------------------------------------
def _componentes(imagen):
    """Zonas de tinta rellenas: (x, y, w, h, area) de cada una, en píxeles."""
    alto, ancho = imagen.shape[:2]
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB).astype(np.int16)
    borde = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    fondo = np.median(borde, axis=0)
    dist = np.sqrt(((lab - fondo) ** 2).sum(axis=2))
    tinta = (dist > DIST_TINTA).astype(np.uint8)
    # sin las líneas finas (ejes, rejilla) ni el trazo de las letras
    k = max(3, int(round(min(alto, ancho) * 0.008)))
    sin_lineas = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(sin_lineas, connectivity=4)
    area_min = max(25, 0.0003 * alto * ancho)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < area_min or area / float(w * h) < EXTENSION_MIN:
            continue
        if (w > 0.9 * ancho and h < 2 * k) or (h > 0.9 * alto and w < 2 * k):
            continue                      # un eje o una línea de rejilla gruesa
        comps.append((x, y, w, h, area))
    return comps, sin_lineas, k, lab


def _partir_contiguas(comps, mascara, vertical):
    """Barras pegadas unas a otras (histogramas) salen como una sola zona:
    se cortan donde cambia de golpe la altura del borde superior."""
    salida = []
    for x, y, w, h, area in comps:
        largo = h if vertical else w
        zona = mascara[y:y + h, x:x + w] if vertical else mascara[y:y + h, x:x + w].T
        perfil = np.array([np.argmax(col) if col.any() else largo for col in zona.T], float)
        cortes = [0] + [i for i in range(1, len(perfil))
                        if abs(perfil[i] - perfil[i - 1]) > max(3, 0.03 * largo)] + [len(perfil)]
        if len(cortes) <= 2:
            salida.append((x, y, w, h, area))
            continue
        for a, b in zip(cortes[:-1], cortes[1:]):
            if b - a < 3:
                continue
            tope = int(np.median(perfil[a:b]))
            if vertical:
                salida.append((x + a, y + tope, b - a, h - tope, (b - a) * (h - tope)))
            else:
                salida.append((x + tope, y + a, w - tope, b - a, (b - a) * (w - tope)))
    return salida


def _tramos_de_color(lab, fondo, x, y, w, h, vertical):
    """Partes de distinto color a lo largo de una barra (apiladas que se
    tocan): [(desde, hasta)] en píxeles desde el borde de arriba (vertical) o
    de la izquierda (horizontal). Sin los tramos del color del fondo (un
    número escrito adentro) ni los muy cortos (el borde entre dos colores)."""
    if vertical:
        perfil = np.median(lab[y:y + h, x + w // 4:x + w - w // 4 or x + 1], axis=1)
    else:
        perfil = np.median(lab[y + h // 4:y + h - h // 4 or y + 1, x:x + w], axis=0)
    tramos, ref = [[0, 1]], perfil[0]
    for i in range(1, len(perfil)):
        if np.linalg.norm(perfil[i] - ref) > DIST_SERIE:
            tramos.append([i, i + 1])
            ref = perfil[i]
        else:
            tramos[-1][1] = i + 1
    utiles = []
    for a, b in tramos:
        color = np.median(perfil[a:b], axis=0)
        if b - a < 3 or np.linalg.norm(color - fondo) <= DIST_TINTA:
            continue
        if utiles and np.linalg.norm(color - utiles[-1][2]) <= DIST_SERIE:
            utiles[-1][1] = b              # el mismo color de antes (cortado por un número)
        else:
            utiles.append([a, b, color])
    largo = h if vertical else w
    # el contorno de la barra (más oscuro que el relleno) no es una parte apilada
    minimo = max(5, 0.04 * largo)
    while len(utiles) > 1 and utiles[0][1] - utiles[0][0] < minimo:
        utiles.pop(0)
    while len(utiles) > 1 and utiles[-1][1] - utiles[-1][0] < minimo:
        utiles.pop()
    if not utiles:
        return [(0, largo)]
    # las puntas de la barra son las de la zona entera (no donde empieza el relleno)
    utiles[0][0], utiles[-1][1] = 0, largo
    for k in range(1, len(utiles)):
        utiles[k][0] = utiles[k - 1][1]
    return [(a, b) for a, b, _ in utiles]


def _barras(imagen):
    comps, mascara, k, lab = _componentes(imagen)
    fondo = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), axis=0)
    alto, ancho = imagen.shape[:2]
    if not comps:
        return None
    tol_v, tol_h = 0.012 * alto + 2, 0.012 * ancho + 2
    # ¿Verticales u horizontales? Las barras se apoyan en una misma línea de
    # base: abajo (verticales) o a la izquierda (horizontales). Las negativas
    # se apoyan por el otro lado.
    base_v, abajo = _moda([y + h for x, y, w, h, a in comps], tol_v)
    _, arriba = _moda([y for x, y, w, h, a in comps], tol_v)
    base_h, izq = _moda([x for x, y, w, h, a in comps], tol_h)
    apoyo_v = [a_ or (abs(c[1] - base_v) <= tol_v) for a_, c in zip(abajo, comps)]
    apoyo_h = [a_ or (abs(c[0] + c[2] - base_h) <= tol_h) for a_, c in zip(izq, comps)]
    vertical = sum(apoyo_v) >= sum(apoyo_h)
    apoyadas = [c for c, ok in zip(comps, apoyo_v if vertical else apoyo_h) if ok]
    sueltas = [c for c, ok in zip(comps, apoyo_v if vertical else apoyo_h) if not ok]
    base = base_v if vertical else base_h
    apoyadas = _partir_contiguas(apoyadas, mascara, vertical)

    barras = []
    apiladas = False
    for x, y, w, h, area in apoyadas:
        if vertical:
            negativa = abs(y - base) <= tol_v and abs(y + h - base) > tol_v
        else:
            negativa = abs(x + w - base) <= tol_h and abs(x - base) > tol_h
        tramos = _tramos_de_color(lab, fondo, x, y, w, h, vertical)
        # la parte apoyada en la base es la barra; las demás, apiladas encima
        en_base = 0 if (negativa == vertical) else len(tramos) - 1
        if len(tramos) > 1:
            apiladas = True
        for t, (a, b) in enumerate(tramos):
            if vertical:
                bx, by, bw, bh = x, y + a, w, b - a
                lejos, cerca = (by + bh, by) if negativa else (by, by + bh)
                c0, c1 = x, x + w
            else:
                bx, by, bw, bh = x + a, y, b - a, h
                lejos, cerca = (bx, bx + bw) if negativa else (bx + bw, bx)
                c0, c1 = y, y + h
            barras.append({"x": bx, "y": by, "w": bw, "h": bh, "c0": c0, "c1": c1,
                           "lejos": lejos, "cerca": cerca, "apilada": t != en_base})
    # partes apiladas separadas por un borde: zonas que se apoyan sobre una barra
    cambio = True
    while cambio:
        cambio = False
        for c in list(sueltas):
            x, y, w, h, area = c
            c0, c1 = (x, x + w) if vertical else (y, y + h)
            for b in barras:
                solape = min(c1, b["c1"]) - max(c0, b["c0"])
                if solape < 0.8 * min(c1 - c0, b["c1"] - b["c0"]):
                    continue
                cerca, lejos = (y + h, y) if vertical else (x, x + w)
                if abs(cerca - b["lejos"]) <= max(3, tol_v / 2):
                    barras.append({"x": x, "y": y, "w": w, "h": h, "c0": c0, "c1": c1,
                                   "lejos": lejos, "cerca": cerca, "apilada": True})
                    sueltas.remove(c)
                    apiladas = cambio = True
                    break
    for b in barras:
        b["color"] = _color_lab(lab, b["x"], b["y"], b["w"], b["h"])
    return {"barras": barras, "sueltas": sueltas, "vertical": vertical, "base": base,
            "apiladas": apiladas, "k": k, "lab": lab}


# ------------------------------------------------------------------
# 2. Series (por color) y leyenda
# ------------------------------------------------------------------
def _series_por_color(barras):
    colores = []
    for b in barras:
        for j, c in enumerate(colores):
            if np.linalg.norm(b["color"] - c) < DIST_SERIE:
                b["serie"] = j
                break
        else:
            b["serie"] = len(colores)
            colores.append(b["color"])
    return colores


def _nombres_leyenda(colores, sueltas, lab, tokens, usados):
    """Nombre de cada serie: el texto a la derecha de la muestra de su color."""
    nombres = [None] * len(colores)
    for x, y, w, h, area in sueltas:
        if not (0.25 <= w / float(h) <= 4):
            continue
        color = _color_lab(lab, x, y, w, h)
        dist = [np.linalg.norm(color - c) for c in colores]
        j = int(np.argmin(dist))
        if dist[j] >= DIST_SERIE or nombres[j]:
            continue
        cy = y + h / 2.0
        renglon = sorted([t for t in tokens if id(t) not in usados and t["px"] >= x + w - 2
                          and abs(t["centro_y"] - cy) <= max(h, t["ph"]) * 0.7], key=lambda t: t["px"])
        elegidos, fin = [], x + w
        for t in renglon:
            if t["px"] - fin > 2.5 * t["ph"]:
                break
            elegidos.append(t)
            fin = t["px"] + t["pw"]
        if elegidos:
            nombres[j] = " ".join(t["texto"] for t in elegidos)
            usados.update(id(t) for t in elegidos)
    return nombres


# ------------------------------------------------------------------
# 3. Categorías
# ------------------------------------------------------------------
def _grupos(barras, n_series):
    """Barras de una misma categoría: las apiladas (misma posición) y, con
    varias series, las pegadas de distinto color."""
    orden = sorted(barras, key=lambda b: (b["c0"] + b["c1"]) / 2)
    ancho_med = float(np.median([b["c1"] - b["c0"] for b in barras]))
    grupos = []
    for b in orden:
        g = grupos[-1] if grupos else None
        if g is not None:
            g0, g1 = min(x["c0"] for x in g), max(x["c1"] for x in g)
            solape = min(b["c1"], g1) - max(b["c0"], g0)
            pegada = b["c0"] - g1 <= 0.5 * ancho_med
            if solape > 0.5 * (b["c1"] - b["c0"]) or (n_series > 1 and pegada
                                                      and b["serie"] not in {x["serie"] for x in g}):
                g.append(b)
                continue
        grupos.append([b])
    return grupos


_RE_NOTA = re.compile(r"^(fuente|elaboraci[oó]n|nota|source|note)s?\b", re.I)


def _rotulos_categorias(grupos, tokens, usados, vertical, base, alto_texto, palabras=None):
    # la fuente o una nota al pie no es el nombre de una categoría
    for t in tokens:
        if _RE_NOTA.match(t["texto"]):
            renglon = [x for x in tokens if abs(x["centro_y"] - t["centro_y"]) <= 0.6 * t["ph"]]
            usados.update(id(x) for x in renglon)
    nombres = []
    extensiones = [(min(b["c0"] for b in g), max(b["c1"] for b in g)) for g in grupos]
    for k, (c0, c1) in enumerate(extensiones):
        # hasta la mitad del hueco con los grupos vecinos
        izq = (extensiones[k - 1][1] + c0) / 2 if k else c0 - (c1 - c0)
        der = (c1 + extensiones[k + 1][0]) / 2 if k + 1 < len(extensiones) else c1 + (c1 - c0)
        if vertical:
            elegidos = [t for t in tokens if id(t) not in usados and izq <= t["centro_x"] <= der
                        and base - 0.3 * t["ph"] <= t["centro_y"] <= base + 3 * alto_texto]
        else:
            elegidos = [t for t in tokens if id(t) not in usados and izq <= t["centro_y"] <= der
                        and t["centro_x"] < base]
        if vertical and elegidos:
            # solo los renglones pegados a la base (debajo puede estar el título del eje)
            elegidos.sort(key=lambda t: t["centro_y"])
            corte = [elegidos[0]]
            for t in elegidos[1:]:
                if t["py"] - max(c["py"] + c["ph"] for c in corte) > 0.8 * alto_texto:
                    break
                corte.append(t)
            elegidos = corte
        usados.update(id(t) for t in elegidos)
        nombres.append((_unir(elegidos), elegidos))
    # rótulos inclinados o verticales (texto del PDF): por la punta más cercana al eje
    if vertical and palabras and sum(1 for txt, _ in nombres if not txt) > len(nombres) / 2:
        inclinados = sg.categorias_inclinadas_de_palabras(palabras, base - 2)
        if inclinados:
            centros = [(c0 + c1) / 2 for c0, c1 in extensiones]
            paso = float(np.median(np.diff(centros))) if len(centros) > 1 else (extensiones[0][1] - extensiones[0][0])
            for r in inclinados:
                k = int(np.argmin([abs(r["centro_x"] - c) for c in centros]))
                if abs(r["centro_x"] - centros[k]) <= 0.6 * paso and not nombres[k][0]:
                    nombres[k] = (r["texto"], [])
    return nombres


# ------------------------------------------------------------------
# 4. Valores
# ------------------------------------------------------------------
def _escala(tokens, usados, barras, vertical, base):
    """Números del eje de valores: una columna a la izquierda de las barras
    (verticales) o una fila debajo (horizontales)."""
    nums = [t for t in tokens if t["valor"] is not None and id(t) not in usados]
    if vertical:
        borde = min(b["c0"] for b in barras)
        lado = [t for t in nums if t["centro_x"] < borde]
        if not lado:
            return [], None
        derecha = max(t["px"] + t["pw"] for t in lado)
        col = [t for t in lado if derecha - (t["px"] + t["pw"]) <= max(4, 0.6 * t["ph"])]
        pix = [t["centro_y"] for t in col]
    else:
        borde = max(b["c1"] for b in barras)
        lado = [t for t in nums if t["centro_y"] > borde]
        if not lado:
            return [], None
        arriba = min(t["py"] for t in lado)
        col = [t for t in lado if t["py"] - arriba <= max(4, 0.6 * t["ph"])]
        pix = [t["centro_x"] for t in col]
    if len({t["valor"] for t in col}) < 2:
        return [], None
    m, b, info = sg.calibrar_eje(pix, [t["valor"] for t in col])
    if m is None or (info.get("r2") or 0) < R2_MIN:
        return col, None
    return col, (m, b, info)


def _valor(cal, pixel):
    m, b, info = cal
    v = m * pixel + b
    return 10 ** v if info.get("escala") == "log" else v


def _etiquetas_de_valor(barras, tokens, usados, vertical):
    """El número escrito en o junto a cada barra (encima, o adentro cerca del
    extremo): es el dato exacto."""
    for b in sorted(barras, key=lambda b: -(abs(b["lejos"] - b["cerca"]))):
        largo = abs(b["lejos"] - b["cerca"])
        ancho = b["c1"] - b["c0"]
        afuera = 1 if (b["lejos"] < b["cerca"]) == vertical else -1   # hacia dónde crece
        mejor = None
        for t in tokens:
            if t["valor"] is None or id(t) in usados:
                continue
            c = t["centro_x"] if vertical else t["centro_y"]
            v = t["centro_y"] if vertical else t["centro_x"]
            alto_t = t["ph"] if vertical else t["pw"]
            if not (b["c0"] - 0.3 * ancho <= c <= b["c1"] + 0.3 * ancho):
                continue
            # distancia desde el extremo de la barra, positiva hacia afuera
            d = (b["lejos"] - v) if vertical else (v - b["lejos"])
            d = d if afuera == 1 else -d
            if -0.6 * largo - alto_t <= d <= 2.5 * alto_t and (mejor is None or abs(d) < abs(mejor[0])):
                mejor = (d, t)
        if mejor:
            b["etiqueta"] = mejor[1]
            usados.add(id(mejor[1]))


# ------------------------------------------------------------------
# Lectura completa
# ------------------------------------------------------------------
def leer_barras(ruta_imagen, dir_resultados, palabras=None, lang=None):
    os.makedirs(dir_resultados, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    advertencias = []
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError(f"No se pudo abrir la imagen: {ruta_imagen}")
    alto_img, ancho_img = imagen.shape[:2]
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    tokens = _tokens(palabras, gris, lang)
    alto_texto = float(np.median([t["ph"] for t in tokens])) if tokens else 0.03 * alto_img

    lectura = _barras(imagen)
    if not lectura or not lectura["barras"]:
        raise ValueError("No se encontraron barras en la imagen.")
    barras, vertical, base = lectura["barras"], lectura["vertical"], lectura["base"]
    if lectura["apiladas"]:
        advertencias.append("Barras apiladas: en la lámina cada parte va como una barra al lado "
                            "de la otra, con su valor. Revisar en la vista previa.")

    colores = _series_por_color(barras)
    usados = set()
    nombres_series = _nombres_leyenda(colores, lectura["sueltas"], lectura["lab"], tokens, usados) \
        if len(colores) > 1 else [None]
    if len(colores) > 3:
        advertencias.append(f"Se distinguieron {len(colores)} colores de barras: la lámina tiene "
                            "texturas para 3 series. Revisar.")

    # escala del eje (antes de las etiquetas de cada barra: sus números son otros)
    col_escala, cal = _escala(tokens, usados, barras, vertical, base)
    usados.update(id(t) for t in col_escala)
    _etiquetas_de_valor(barras, tokens, usados, vertical)

    # valores: el número escrito, o la altura con la escala
    con_etiqueta = [b for b in barras if b.get("etiqueta") and not b.get("apilada")]
    if cal is None and len({b["lejos"] for b in con_etiqueta}) >= 2:
        m, b0, info = sg.calibrar_eje([b["lejos"] for b in con_etiqueta],
                                      [b["etiqueta"]["valor"] for b in con_etiqueta])
        if m is not None and (info.get("r2") or 0) >= 0.95:
            cal = (m, b0, info)
            advertencias.append("El gráfico no tiene números en el eje: la escala se calculó con "
                                "los valores escritos sobre las barras.")
    for b in barras:
        if b.get("etiqueta"):
            b["valor"], b["leido"] = float(b["etiqueta"]["valor"]), True
        elif cal is not None:
            if b.get("apilada"):
                b["valor"] = _valor(cal, b["lejos"]) - _valor(cal, b["cerca"])
            else:
                b["valor"] = _valor(cal, b["lejos"])
            b["leido"] = False
        else:
            b["valor"], b["leido"] = None, False
    # lo leído con la escala, redondeado a una décima del paso entre sus marcas
    # (0,99 con marcas 0, 1, 2... es 1): más precisión que eso es ruido de píxeles
    marcas = sorted({t["valor"] for t in col_escala})
    if len(marcas) >= 2 and (cal is None or cal[2].get("escala") != "log"):
        resolucion = float(np.median(np.diff(marcas))) / 10
        for b in barras:
            if b["valor"] is not None and not b["leido"] and resolucion > 0:
                b["valor"] = round(round(b["valor"] / resolucion) * resolucion, 10)
    sin_valor = [b for b in barras if b["valor"] is None]
    if sin_valor:
        advertencias.append(f"No se pudo leer el valor de {len(sin_valor)} barra(s): el gráfico no "
                            "tiene escala ni números sobre esas barras. Escríbelos en la vista previa.")

    grupos = _grupos(barras, len(colores))
    if len(colores) > 1 and not lectura["apiladas"] and all(len(g) == 1 for g in grupos):
        for b in barras:
            b["serie"] = 0
        colores = [colores[0]]
        nombres_series = [None]
    rotulos = _rotulos_categorias(grupos, tokens, usados, vertical, base, alto_texto, palabras)
    # la muestra de color de una leyenda puesta a la altura de las barras no es
    # una categoría: un grupo en una punta, sin nombre y muy separado del resto
    while len(grupos) > 2:
        centros = [(min(b["c0"] for b in g) + max(b["c1"] for b in g)) / 2 for g in grupos]
        huecos = np.diff(centros)
        tipico = float(np.median(huecos))
        def _con_texto_al_lado(g):
            # una muestra de leyenda tiene su nombre a la derecha, a su altura
            b = g[0]
            cy = b["y"] + b["h"] / 2
            return len(g) == 1 and any(0 <= t["px"] - (b["x"] + b["w"]) <= 3 * t["ph"]
                                       and abs(t["centro_y"] - cy) <= max(b["h"], t["ph"])
                                       for t in tokens)
        if not rotulos[-1][0] and (huecos[-1] > 1.6 * tipico or _con_texto_al_lado(grupos[-1])):
            quitar = -1
        elif not rotulos[0][0] and (huecos[0] > 1.6 * tipico or _con_texto_al_lado(grupos[0])):
            quitar = 0
        else:
            break
        for b in grupos[quitar]:
            barras.remove(b)
        grupos.pop(quitar)
        rotulos.pop(quitar)
    categorias = [txt or f"Categoría {k + 1}" for k, (txt, _) in enumerate(rotulos)]
    if sum(1 for txt, _ in rotulos if not txt) > 0:
        advertencias.append("No se leyó el nombre de alguna categoría: va como «Categoría N». "
                            "Escríbelo en la vista previa.")
    for k, g in enumerate(grupos):
        for b in g:
            b["categoria"] = k

    # títulos
    if vertical:
        tope = min([b["y"] for b in barras] + [t["py"] for t in col_escala])
    else:
        tope = min([b["y"] for b in barras] + [t["py"] for _, ts in rotulos for t in ts])
    arriba = sorted([t for t in tokens if id(t) not in usados and t["centro_y"] < tope - 0.3 * alto_texto],
                    key=lambda t: t["centro_y"])
    titulo_toks = []
    for t in arriba:
        if titulo_toks and t["py"] - max(x["py"] + x["ph"] for x in titulo_toks) > 1.2 * alto_texto:
            break
        titulo_toks.append(t)
    usados.update(id(t) for t in titulo_toks)
    abajo_de = max([b["y"] + b["h"] for b in barras] + [t["py"] + t["ph"] for _, ts in rotulos for t in ts]
                   + ([t["py"] + t["ph"] for t in col_escala] if not vertical else []))
    bajo = sorted([t for t in tokens if id(t) not in usados and t["centro_y"] > abajo_de], key=lambda t: t["centro_y"])
    titulo_bajo = [t for t in bajo if bajo and t["centro_y"] - bajo[0]["centro_y"] <= 0.8 * alto_texto]
    usados.update(id(t) for t in titulo_bajo)
    titulo_vertical, bbox_vertical = "", None
    limite = min([b["x"] for b in barras] + [t["px"] for t in col_escala] if vertical
                 else [b["x"] for b in barras] + [t["px"] for _, ts in rotulos for t in ts])
    if palabras:
        titulo_vertical, bbox_vertical = sg.titulo_vertical_de_palabras(palabras, limite)
    elif sg.ocr_disponible():
        try:
            titulo_vertical, bbox_vertical = sg.detectar_texto_vertical(gris, limite, lang=lang)
        except Exception:
            pass
    # en la lámina: eje X = categorías, eje Y = valores
    if vertical:
        titulo_x, titulo_y = _unir(titulo_bajo), titulo_vertical
    else:
        titulo_x, titulo_y = titulo_vertical, _unir(titulo_bajo)
    titulo = _unir(titulo_toks)
    if not sg._texto_plausible(titulo):
        titulo = ""

    # calibraciones (en la orientación normalizada): categoría i <-> píxel
    centros = [((min(b["c0"] for b in g) + max(b["c1"] for b in g)) / 2) for g in grupos]
    if len(centros) >= 2:
        m_x, b_x = np.polyfit(centros, list(range(len(centros))), 1)
        cal_x = {"m": float(m_x), "b": float(b_x), "r2": 1.0, "escala": "lineal"}
    else:
        cal_x = {"m": None, "b": None, "r2": None, "escala": "lineal"}
    cal_y = ({"m": float(cal[0]), "b": float(cal[1]), **cal[2]} if cal
             else {"m": None, "b": None, "r2": None, "escala": "lineal"})

    # series de salida: un punto por barra, en el orden de las categorías
    series_salida = []
    for j, color in enumerate(colores):
        bgr = cv2.cvtColor(np.uint8([[np.clip(color, 0, 255)]]), cv2.COLOR_LAB2BGR)[0][0]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        puntos = []
        for b in sorted([b for b in barras if b["serie"] == j], key=lambda b: b["categoria"]):
            cx = (b["c0"] + b["c1"]) / 2
            px, py = (cx, b["lejos"]) if vertical else (b["lejos"], cx)
            puntos.append({"px": float(px), "py": float(py), "valor_x": float(b["categoria"]),
                           "valor_y": None if b["valor"] is None else float(b["valor"]),
                           "categoria": categorias[b["categoria"]], "valor_leido": b["leido"]})
        series_salida.append({"id": f"serie_{j + 1}", "nombre": nombres_series[j] if j < len(nombres_series) else None,
                              "hue": int(hsv[0]), "color_bgr": [int(v) for v in bgr],
                              "modo": "color", "puntos": puntos})

    rect = [int(min(b["x"] for b in barras)), int(min(b["y"] for b in barras)),
            int(max(b["x"] + b["w"] for b in barras)), int(max(b["y"] + b["h"] for b in barras))]
    if not vertical:          # en la orientación normalizada (ver arriba)
        rect = [rect[1], ancho_img - rect[2], rect[3], ancho_img - rect[0]]
    eje_x, eje_y = sg.detectar_ejes(gris)
    listo = bool(barras) and not sin_valor

    # ---------- overlay ----------
    overlay = imagen.copy()
    capa = imagen.copy()
    paleta = [(0, 160, 0), (0, 0, 220), (220, 120, 0), (160, 0, 160), (0, 160, 200)]
    for b in barras:
        color = paleta[b["serie"] % len(paleta)]
        cv2.rectangle(capa, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, -1)
    overlay = cv2.addWeighted(capa, 0.35, overlay, 0.65, 0)
    for b in barras:
        color = paleta[b["serie"] % len(paleta)]
        cv2.rectangle(overlay, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, 2)
        if b.get("etiqueta"):
            t = b["etiqueta"]
            cv2.rectangle(overlay, (t["px"], t["py"]), (t["px"] + t["pw"], t["py"] + t["ph"]), (128, 0, 128), 1)
    for t in col_escala:
        cv2.rectangle(overlay, (t["px"], t["py"]), (t["px"] + t["pw"], t["py"] + t["ph"]), (0, 200, 255), 1)
    for _, ts in rotulos:
        for t in ts:
            cv2.rectangle(overlay, (t["px"], t["py"]), (t["px"] + t["pw"], t["py"] + t["ph"]), (0, 200, 200), 1)
    nombre_overlay = f"overlay_{uid}.png"
    ruta_overlay = os.path.join(dir_resultados, nombre_overlay)
    cv2.imwrite(ruta_overlay, overlay)

    resumen = {
        "tipo_grafica": "barras", "orientacion": "vertical" if vertical else "horizontal",
        "apiladas": lectura["apiladas"],
        "eje_x_detectado": eje_x is not None, "eje_y_detectado": eje_y is not None,
        "curva_detectada": bool(barras), "rect_grafico": rect,
        "n_series": len(series_salida), "n_barras": len(barras), "n_categorias": len(grupos),
        "n_etiquetas_x": 0, "n_etiquetas_y": len(col_escala), "n_etiquetas_dato": sum(1 for b in barras if b["leido"]),
        "valores_etiquetas_x": [], "valores_etiquetas_y": [t["valor"] for t in col_escala],
        "titulo": titulo, "titulo_eje_x": titulo_x, "titulo_eje_y": titulo_y, "leyenda": "",
        "n_puntos_curva": 0, "calibrado_x": True, "calibrado_y": listo,
        "calibrado_y_por_datos": cal is not None and not col_escala,
        "r2_x": 1.0, "r2_y": cal_y.get("r2"), "ancho_imagen": ancho_img, "alto_imagen": alto_img,
        "version_segmentador": f"barras {VERSION}", "listo_para_stl": listo,
    }
    datos = {
        "tipo_grafica": "barras", "orientacion": resumen["orientacion"],
        "imagen": {"ancho": ancho_img, "alto": alto_img, "archivo": os.path.basename(ruta_imagen)},
        "ejes": {"eje_x": list(eje_x) if eje_x else None, "eje_y": list(eje_y) if eje_y else None,
                 "rect_grafico": rect, "calibracion_x": cal_x, "calibracion_y": cal_y,
                 # las calibraciones son de la orientación normalizada: con
                 # barras horizontales, x de la imagen = valor e y = categoría
                 "transpuesta": not vertical},
        "textos": {
            "titulo": titulo, "titulo_eje_x": titulo_x, "titulo_eje_y": titulo_y,
            "titulo_bbox": _caja(titulo_toks), "titulo_eje_x_bbox": None, "titulo_eje_y_bbox": bbox_vertical,
            "leyenda": " | ".join(n for n in nombres_series if n), "leyenda_entradas": [],
            "etiquetas_eje_x": [], "etiquetas_dato": [],
            "etiquetas_eje_y": [{"valor": t["valor"], "py": t["centro_y"] if vertical else t["centro_x"]}
                                for t in col_escala],
            "categorias_x": categorias,
        },
        "barras": [{"serie": b["serie"], "categoria": b["categoria"], "valor": b["valor"],
                    "valor_leido": b["leido"], "apilada": bool(b.get("apilada")),
                    "bbox": [b["x"], b["y"], b["w"], b["h"]]} for b in barras],
        "series": series_salida, "resumen": resumen, "advertencias": advertencias,
        "_ruta_imagen_original": os.path.abspath(ruta_imagen),
    }
    nombre_json = f"grafica_{uid}.json"
    ruta_json = os.path.join(dir_resultados, nombre_json)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

    nombre_csv = f"descripcion_{uid}.csv"
    ruta_csv = os.path.join(dir_resultados, nombre_csv)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "elemento", "x1", "y1", "x2", "y2", "ancho_px", "alto_px",
                    "valor_x", "valor_y", "texto", "descripcion"])
        w.writerow(["resumen", "grafica", "", "", "", "", ancho_img, alto_img, "", "", titulo,
                    f"Gráfico de barras {resumen['orientacion']}es con {len(grupos)} categoría(s), "
                    f"{len(series_salida)} serie(s) y {len(barras)} barra(s)."])
        for adv in advertencias:
            w.writerow(["advertencia", "revisar", "", "", "", "", "", "", "", "", "", adv])
        for b in sorted(barras, key=lambda b: (b["categoria"], b["serie"])):
            nombre = series_salida[b["serie"]]["nombre"] or f"serie {b['serie'] + 1}"
            valor = "" if b["valor"] is None else f"{b['valor']:.6g}"
            w.writerow(["barra", f"serie_{b['serie'] + 1}", b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"],
                        b["w"], b["h"], b["categoria"], valor, categorias[b["categoria"]],
                        f"Barra de {categorias[b['categoria']]} ({nombre}): "
                        + (f"{valor} ({'escrito en el gráfico' if b['leido'] else 'leído con la escala'})."
                           if valor else "sin valor legible.")])

    return {
        "ruta_csv": ruta_csv, "nombre_csv": nombre_csv, "ruta_json": ruta_json, "nombre_json": nombre_json,
        "ruta_overlay": ruta_overlay, "nombre_overlay": nombre_overlay,
        "series": series_salida, "puntos_curva": [], "advertencias": advertencias, "resumen": resumen,
        "tipo_grafica": "barras",
    }
