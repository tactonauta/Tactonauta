"""
segmentador.py — extrae los datos de una gráfica (imagen) para la etapa BANA/STL.

Pasos: ejes y área de dibujo -> curvas por color -> textos (OCR + plantillas)
-> calibración píxel->valor -> CSV, JSON y overlay de verificación.
"""

import os
import re
import csv
import json
import math
import uuid

import cv2
import numpy as np

import pytesseract
from pytesseract import Output

__version__ = "2026-10-05-v8-unido"

_RUTA_TESSERACT_WINDOWS = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
if os.name == "nt" and os.path.isfile(_RUTA_TESSERACT_WINDOWS):
    pytesseract.pytesseract.tesseract_cmd = _RUTA_TESSERACT_WINDOWS


# ---------- 0) UTILIDADES GENERALES ----------

def _tolerancias(forma):
    """Tolerancias en píxeles, proporcionales al tamaño de la imagen."""
    alto, ancho = forma[:2]
    diag = math.hypot(alto, ancho)
    return {
        "tol_linea": max(2, int(round(0.004 * diag))),
        "largo_min_h": max(40, int(round(0.30 * ancho))),
        "largo_min_v": max(40, int(round(0.30 * alto))),
        "gap_hough": max(5, int(round(0.012 * diag))),
        "margen_texto": max(4, int(round(0.012 * diag))),
        "radio_dato": max(10, int(round(0.022 * diag))),
        "area_min_curva": max(25, int(round(0.00035 * alto * ancho))),
    }


_OCR_DISPONIBLE = None


def ocr_disponible():
    """¿Está Tesseract instalado? (se consulta una sola vez). Sin él, el
    segmentador sigue funcionando con el texto del PDF o sin texto, en vez
    de caerse con TesseractNotFoundError."""
    global _OCR_DISPONIBLE
    if _OCR_DISPONIBLE is None:
        try:
            pytesseract.get_tesseract_version()
            _OCR_DISPONIBLE = True
        except Exception:
            _OCR_DISPONIBLE = False
    return _OCR_DISPONIBLE


def _idioma_ocr(preferido=None):
    """Idioma de Tesseract: spa+eng si están instalados."""
    if preferido:
        return preferido
    try:
        disponibles = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    if "spa" in disponibles and "eng" in disponibles:
        return "spa+eng"
    if "spa" in disponibles:
        return "spa"
    return "eng"


_RE_NUMERO = re.compile(
    r"^[−–—-]?\d{1,3}(?:[ .,]\d{3})+(?:[.,]\d+)?$"      # 1.000  /  12 500,5
    r"|^[−–—-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?$"       # 12  /  3,5  /  1.2e3
)


# Lo que puede acompañar a un número en un eje sin cambiar su valor:
# "20%", "$1,500", "S/ 30", "€ 12", "30 °C".
_RE_ADORNO_NUMERO = re.compile(r"^(?:S/\.?|US\$|\$|€|£|¥)\s*|\s*(?:%|‰|°C|°F|°)$")


def _sin_adornos(texto):
    return _RE_ADORNO_NUMERO.sub("", texto.strip()).strip()


def _es_numero(texto):
    return bool(_RE_NUMERO.match(_sin_adornos(texto)))


def _a_float(texto):
    """Texto -> float (notación española o inglesa); None si no se puede."""
    t = _sin_adornos(texto).replace("−", "-").replace("–", "-").replace("—", "-")
    t = t.replace(" ", "")
    tiene_punto, tiene_coma = "." in t, "," in t
    if tiene_punto and tiene_coma:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif tiene_coma:
        entero, _, dec = t.partition(",")
        t = t.replace(",", "") if (len(dec) == 3 and len(entero.lstrip("-")) <= 3) else t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _mediana_movil(v, k):
    if k < 3 or v.size < k:
        return v
    if k % 2 == 0:
        k += 1
    ext = np.pad(v, k // 2, mode="edge")
    ventanas = np.lib.stride_tricks.sliding_window_view(ext, k)
    return np.median(ventanas, axis=1)


# ---------- 1) DETECCIÓN DE EJES Y DEL ÁREA DE DIBUJO ----------

def _fusionar_lineas(lineas, orientacion, tol):
    """Une las líneas casi colineales que Hough devuelve duplicadas o en trozos."""
    if not lineas:
        return []

    def pos(l):
        return (l[1] + l[3]) / 2.0 if orientacion == "h" else (l[0] + l[2]) / 2.0

    def extremos(l):
        return (min(l[0], l[2]), max(l[0], l[2])) if orientacion == "h" \
            else (min(l[1], l[3]), max(l[1], l[3]))

    grupos = []
    for l in sorted(lineas, key=pos):
        p = pos(l)
        a, b = extremos(l)
        if grupos and abs(grupos[-1]["pos"] - p) <= tol:
            g = grupos[-1]
            g["pos"] = (g["pos"] * g["n"] + p) / (g["n"] + 1)
            g["n"] += 1
            g["a"] = min(g["a"], a)
            g["b"] = max(g["b"], b)
        else:
            grupos.append({"pos": p, "a": a, "b": b, "n": 1})

    for g in grupos:
        g["largo"] = g["b"] - g["a"]
    return grupos


def detectar_ejes(gris, tol=None):
    """Devuelve (eje_x, eje_y) como (x1, y1, x2, y2), o None si no se encuentran."""
    t = tol or _tolerancias(gris.shape)
    alto, ancho = gris.shape[:2]

    bordes = cv2.Canny(gris, 50, 150)
    lineas = cv2.HoughLinesP(
        bordes, rho=1, theta=np.pi / 180, threshold=60,
        minLineLength=min(t["largo_min_h"], t["largo_min_v"]),
        maxLineGap=t["gap_hough"],
    )
    if lineas is None:
        return None, None

    horizontales, verticales = [], []
    for x1, y1, x2, y2 in lineas.reshape(-1, 4):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy <= t["tol_linea"] and dx >= t["largo_min_h"] * 0.5:
            horizontales.append((x1, y1, x2, y2))
        elif dx <= t["tol_linea"] and dy >= t["largo_min_v"] * 0.5:
            verticales.append((x1, y1, x2, y2))

    gh = [g for g in _fusionar_lineas(horizontales, "h", t["tol_linea"] * 2)
          if g["largo"] >= t["largo_min_h"] and 0.02 * alto < g["pos"] < 0.99 * alto]
    gv = [g for g in _fusionar_lineas(verticales, "v", t["tol_linea"] * 2)
          if g["largo"] >= t["largo_min_v"] and 0.01 * ancho < g["pos"] < 0.98 * ancho]

    margen_marco = max(3, int(round(0.01 * min(alto, ancho))))

    def _es_marco(g, dim):
        return g["a"] <= margen_marco and g["b"] >= dim - 1 - margen_marco

    gh_sin_marco = [g for g in gh if not _es_marco(g, ancho)]
    gv_sin_marco = [g for g in gv if not _es_marco(g, alto)]

    eje_x, eje_y = _elegir_esquina(gh_sin_marco, gv_sin_marco, t, alto, ancho)
    if eje_x is None and eje_y is None:
        eje_x, eje_y = _elegir_esquina(gh, gv, t, alto, ancho)
    if eje_x is not None or eje_y is not None:
        return eje_x, eje_y

    gh_resp = gh_sin_marco or gh
    gv_resp = gv_sin_marco or gv

    eje_x = None
    if gh_resp:
        mejor = max(gh_resp, key=lambda g: (round(g["pos"]), g["largo"]))
        fila = int(round(mejor["pos"]))
        eje_x = (int(mejor["a"]), fila, int(mejor["b"]), fila)

    eje_y = None
    if gv_resp:
        if eje_x is not None:
            x_esq, y_esq = eje_x[0], eje_x[1]
            mejor = min(
                gv_resp,
                key=lambda g: abs(g["pos"] - x_esq) + 0.5 * abs(g["b"] - y_esq) - 0.2 * g["largo"],
            )
        else:
            mejor = min(gv_resp, key=lambda g: g["pos"])
        col = int(round(mejor["pos"]))
        eje_y = (col, int(mejor["a"]), col, int(mejor["b"]))

    return eje_x, eje_y


def _elegir_esquina(gh, gv, t, alto, ancho):
    """Elige el par horizontal/vertical que mejor forma la "L" inferior izquierda."""
    if not gh or not gv:
        return None, None
    diag = math.hypot(alto, ancho)
    sobrepaso = max(t["tol_linea"] * 3, int(round(0.10 * min(alto, ancho))))

    mejor, mejor_costo = None, None
    for h in gh:
        for v in gv:
            vx, hy = v["pos"], h["pos"]
            if not (h["a"] - sobrepaso <= vx <= h["a"] + 0.25 * h["largo"]):
                continue
            if not (v["b"] - sobrepaso <= hy <= v["b"] + sobrepaso):
                continue
            if v["a"] >= hy - 0.5 * t["largo_min_v"]:      # la vertical debe subir desde la fila
                continue
            desvio = abs(h["a"] - vx) + abs(v["b"] - hy)
            costo = desvio / diag - 0.6 * (h["largo"] / ancho + v["largo"] / alto)
            if mejor_costo is None or costo < mejor_costo:
                mejor, mejor_costo = (h, v), costo

    if mejor is None:
        return None, None

    h, v = mejor
    col = int(round(v["pos"]))
    fila = int(round(h["pos"]))
    return (col, fila, int(h["b"]), fila), (col, int(v["a"]), col, fila)


def area_de_dibujo(eje_x, eje_y, forma):
    """Rectángulo (x0, y0, x1, y1) delimitado por los ejes."""
    alto, ancho = forma[:2]
    x0 = eje_y[0] if eje_y else 0
    x1 = eje_x[2] if eje_x else ancho - 1
    y1 = eje_x[1] if eje_x else alto - 1
    y0 = eje_y[1] if eje_y else 0
    x0, x1 = max(0, min(x0, x1)), min(ancho - 1, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(alto - 1, max(y0, y1))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return (0, 0, ancho - 1, alto - 1)
    return (x0, y0, x1, y1)


# ---------- 2) DETECCIÓN DE CURVAS (una o varias series) ----------

def _hues_dominantes(h, mask, n_max=3, tol=12, minimo_relativo=0.06):
    """Tonos más frecuentes (histograma circular, para no partir el rojo)."""
    valores = h[mask]
    if valores.size == 0:
        return []
    hist = np.bincount(valores.astype(np.int64), minlength=180).astype(float)
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    k /= k.sum()
    circ = np.concatenate([hist[-2:], hist, hist[:2]])
    hist_s = np.convolve(circ, k, mode="same")[2:-2]

    total = hist_s.sum()
    restante = hist_s.copy()
    picos = []
    for _ in range(n_max):
        i = int(np.argmax(restante))
        if restante[i] <= 0 or hist_s[i] < minimo_relativo * total:
            break
        picos.append(i)
        restante[[(i + d) % 180 for d in range(-tol, tol + 1)]] = 0.0
    return picos


def _mascara_por_hue(h, base, hue, tol):
    d = np.abs(h.astype(np.int16) - hue)
    d = np.minimum(d, 180 - d)          # distancia circular
    return (d <= tol) & base


def nombres_de_leyenda(imagen_bgr, series, tokens, tolerancia_hue=12, sat_minima=60):
    """
    Nombre de cada serie según la leyenda de la gráfica, emparejado por COLOR:
    una entrada de leyenda es una muestra corta del color de la curva (una
    rayita o un cuadradito) con su texto justo a la derecha, en la misma línea.

    Devuelve (nombres, muestras, usados): `nombres[i]` es el texto de la serie
    i o None si no se encontró; `muestras[i]` es el recuadro (x0, y0, x1, y1)
    de su muestra de color o None; `usados` son los tokens que forman esos
    nombres. Solo series detectadas por color: una gráfica en blanco y negro
    no tiene cómo asociar texto a curva.
    """
    nombres = [None] * len(series)
    muestras = [None] * len(series)
    usados_tokens = []
    tokens = [tk for tk in (tokens or []) if any(c.isalnum() for c in tk["texto"])]
    if not tokens:
        return nombres, muestras, usados_tokens

    alto, ancho = imagen_bgr.shape[:2]
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    base = (s > sat_minima) & (v > 40)

    candidatos = []   # (distancia al texto, índice de serie, muestra, tokens del nombre)
    for i, serie in enumerate(series):
        if serie.get("modo") != "color" or serie.get("hue") is None:
            continue
        m = _mascara_por_hue(h, base, serie["hue"], tolerancia_hue).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            x, y, w, hh, area = stats[k]
            # Una muestra de leyenda es chica: la curva misma (o un trozo largo
            # de ella) no puede serlo.
            if area < 6 or w > 0.2 * ancho or hh > 0.1 * alto:
                continue
            cy = y + hh / 2.0
            derecha = [tk for tk in tokens
                       if abs(tk["centro_y"] - cy) <= max(hh, tk["ph"]) * 0.7
                       and x + w - 2 <= tk["px"] <= x + w + max(2.0 * tk["ph"], 0.6 * w)]
            if not derecha:
                continue
            primero = min(derecha, key=lambda tk: tk["px"])
            # El nombre sigue hacia la derecha mientras las palabras estén a
            # distancia de "espacio entre palabras" y en la misma línea.
            nombre = [primero]
            while True:
                ult = nombre[-1]
                sig = [tk for tk in tokens
                       if tk not in nombre
                       and abs(tk["centro_y"] - ult["centro_y"]) <= 0.6 * ult["ph"]
                       and 0 <= tk["px"] - (ult["px"] + ult["pw"]) <= 1.2 * ult["ph"]]
                if not sig:
                    break
                nombre.append(min(sig, key=lambda tk: tk["px"]))
            candidatos.append((primero["px"] - (x + w), i, (int(x), int(y), int(x + w), int(y + hh)), nombre))

    # Cada serie se queda con su muestra más pegada a un texto, y un mismo
    # texto no puede ser el nombre de dos series.
    usados = set()
    for dist, i, muestra, nombre in sorted(candidatos, key=lambda c: c[0]):
        clave = id(nombre[0])
        if nombres[i] is not None or clave in usados:
            continue
        usados.add(clave)
        nombres[i] = " ".join(tk["texto"] for tk in nombre).strip()
        muestras[i] = muestra
        usados_tokens.extend(nombre)
    return nombres, muestras, usados_tokens


def asignar_nombres_de_leyenda(imagen_bgr, series, tokens):
    """Pone en cada serie su "nombre" según la leyenda (ver
    `nombres_de_leyenda`) y borra de su máscara la muestra de color de la
    leyenda: si la leyenda está dentro del área de dibujo, esa rayita era
    parte de la máscara y en esas columnas la polilínea se iba hacia ella
    (valores falsos en el CSV, la lámina y la narración).
    Devuelve (nombres, usados): los nombres (None donde no se encontró) y
    los tokens de texto que los forman."""
    nombres, muestras, usados = nombres_de_leyenda(imagen_bgr, series, tokens)
    for serie, nombre, muestra in zip(series, nombres, muestras):
        serie["nombre"] = nombre
        serie["_muestra"] = muestra
    for muestra in muestras:
        if muestra is None:
            continue
        x0, y0, x1, y1 = muestra
        for serie in series:
            m = serie.get("mascara")
            if m is not None:
                m[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = False
    # Las series quedan en el ORDEN DE LA LEYENDA (arriba->abajo,
    # izquierda->derecha), no en el de "color más abundante": la primera
    # de la leyenda es la serie 1, la de textura sólida en la lámina.
    series.sort(key=lambda s: (s["_muestra"] is None,
                               (s["_muestra"] or (0, 0))[1], (s["_muestra"] or (0, 0))[0]))
    for s in series:
        s.pop("_muestra", None)
    return [s["nombre"] for s in series], usados


def _limpiar_componentes(mask_u8, area_minima):
    """Cierra huecos y conserva todos los trozos grandes de la máscara."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cerrada = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cerrada, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    area_mayor = int(areas.max())
    if area_mayor < area_minima:
        # Todos los trozos son chicos: puede ser una línea DISCONTINUA o
        # PUNTEADA de color (cada guion por separado no llega al área
        # mínima). Se cierran los huecos con un radio del orden del espacio
        # entre guiones y se vuelve a medir.
        lado = max(5, int(round(0.025 * max(mask_u8.shape)))) | 1
        grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
        cerrada = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, grande, iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cerrada, connectivity=8)
        if n <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        area_mayor = int(areas.max())
        if area_mayor < area_minima:
            return None
    umbral = max(area_minima, int(0.02 * area_mayor))  # descarta cuadritos de leyenda
    keep = np.isin(labels, np.where(np.concatenate([[0], areas >= umbral]))[0])
    keep &= labels > 0
    return keep


def detectar_curvas(imagen_bgr, rect=None, sat_minima=60, tolerancia_hue=12,
                    area_minima=40, max_series=3):
    """Lista de series {"hue", "mascara", "modo"}; respaldo por intensidad si no hay color."""
    alto, ancho = imagen_bgr.shape[:2]
    dentro = np.zeros((alto, ancho), dtype=bool)
    if rect is None:
        dentro[:, :] = True
    else:
        x0, y0, x1, y1 = rect
        dentro[y0 + 1:y1, x0 + 1:x1] = True   # inset: excluye los propios ejes

    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    base = (s > sat_minima) & (v > 40) & dentro
    series = []
    if base.any():
        for hue in _hues_dominantes(h, base, n_max=max_series, tol=tolerancia_hue):
            m = _mascara_por_hue(h, base, hue, tolerancia_hue).astype(np.uint8)
            limpia = _limpiar_componentes(m, area_minima)
            if limpia is not None:
                series.append({"hue": hue, "mascara": limpia, "modo": "color"})

    if series:
        return series

    if rect is not None:
        inset = max(3, int(round(0.004 * max(alto, ancho))))
        dentro = np.zeros((alto, ancho), dtype=bool)
        x0, y0, x1, y1 = rect
        dentro[min(y0 + inset, y1):max(y1 - inset, y0 + inset),
               min(x0 + inset, x1):max(x1 - inset, x0 + inset)] = True

    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    oscuro = ((gris < 128) & dentro).astype(np.uint8)
    limpia = _limpiar_componentes(oscuro, max(area_minima, 80))
    if limpia is not None:
        series.append({"hue": None, "mascara": limpia, "modo": "intensidad"})
    return series


# ---------- 3) OCR: DETECCIÓN Y CLASIFICACIÓN DE TEXTO ----------

def _mapa_distancia_curva(mascaras):
    if not mascaras:
        return None
    union = np.zeros_like(mascaras[0], dtype=bool)
    for m in mascaras:
        union |= m
    no_curva = (~union).astype(np.uint8) * 255
    return cv2.distanceTransform(no_curva, cv2.DIST_L2, 5)


def _categorias_en_linea(tokens, x0, x1, margen):
    """Rótulos de categoría de un eje X: los textos de la primera línea
    bajo el eje, agrupados en rótulos (palabras pegadas = un rótulo), si son
    3 o más, caben en el ancho del gráfico y están aproximadamente
    equiespaciados. Devuelve [{"texto", "centro_x", "px", "py", "pw", "ph",
    "tokens"}] ordenados de izquierda a derecha, o []."""
    toks = sorted(tokens, key=lambda z: z["centro_y"])
    h_med = float(np.median([z["ph"] for z in toks])) or 1.0
    linea = [z for z in toks if z["centro_y"] - toks[0]["centro_y"] <= 0.8 * h_med]
    if x0 is not None and x1 is not None:
        linea = [z for z in linea if x0 - 3 * margen <= z["centro_x"] <= x1 + 3 * margen]
    linea.sort(key=lambda z: z["px"])
    # Dos palabras son un mismo rótulo ("Nueva York") solo si son de la misma
    # línea de texto y van una a continuación de la otra. Antes se juntaba
    # todo lo cercano, y rótulos inclinados (recuadros anchos) o fechas que
    # se pisan entre sí ("2023-01" "2023-02"...) quedaban en un único rótulo.
    grupos = []
    for z in linea:
        if grupos:
            prev = grupos[-1][-1]
            hueco = z["px"] - (prev["px"] + prev["pw"])
            misma_linea = tuple(z["orden"][:-1]) == tuple(prev["orden"][:-1])
            if misma_linea and -0.1 * h_med <= hueco <= 0.6 * h_med:
                grupos[-1].append(z)
                continue
        grupos.append([z])
    if len(grupos) < 3:
        return []
    centros = [(g[0]["px"] + g[-1]["px"] + g[-1]["pw"]) / 2.0 for g in grupos]
    pasos = np.diff(centros)
    if pasos.min() <= 0 or pasos.std() > 0.25 * pasos.mean():
        return []
    categorias = []
    for g, c in zip(grupos, centros):
        px, py = min(z["px"] for z in g), min(z["py"] for z in g)
        categorias.append({
            "texto": " ".join(z["texto"] for z in g), "centro_x": c,
            "px": px, "py": py,
            "pw": max(z["px"] + z["pw"] for z in g) - px,
            "ph": max(z["py"] + z["ph"] for z in g) - py,
            "tokens": g,
        })
    return categorias


def tokens_de_palabras_pdf(palabras):
    """Palabras HORIZONTALES del PDF (ver pipeline_rapido.extraer_palabras)
    con el mismo formato que los tokens del OCR de `detectar_textos`."""
    tokens = []
    for p in palabras or []:
        if p.get("vertical"):
            continue
        texto = p["texto"].strip()
        b, l, w = (list(p.get("orden") or []) + [0, 0, 0])[:3]
        tokens.append({
            "texto": texto, "valor": _a_float(texto) if _es_numero(texto) else None,
            "conf": 100.0,
            "px": int(round(p["px"])), "py": int(round(p["py"])),
            "pw": max(1, int(round(p["pw"]))), "ph": max(1, int(round(p["ph"]))),
            "centro_x": p["px"] + p["pw"] / 2.0, "centro_y": p["py"] + p["ph"] / 2.0,
            "orden": (b, 0, l, w),
        })
    return tokens


def titulo_vertical_de_palabras(palabras, x_limite):
    """Título del eje Y desde las palabras ROTADAS del PDF a la izquierda de
    `x_limite` (borde de los números del eje). Se lee de abajo hacia arriba,
    como el texto rotado 90° habitual. Devuelve (texto, bbox) o ("", None)."""
    lineas = {}
    for p in palabras or []:
        if p.get("vertical") and p["px"] + p["pw"] <= x_limite + 3:
            lineas.setdefault(tuple((p.get("orden") or [0, 0])[:2]), []).append(p)
    if not lineas:
        return "", None
    # la línea más cercana a los números del eje
    linea = max(lineas.values(), key=lambda ps: max(q["px"] + q["pw"] for q in ps))
    linea.sort(key=lambda q: -(q["py"] + q["ph"]))
    texto = " ".join(q["texto"] for q in linea).strip()
    x0 = min(q["px"] for q in linea)
    y0 = min(q["py"] for q in linea)
    bbox = {"px": int(x0), "py": int(y0),
            "pw": int(max(q["px"] + q["pw"] for q in linea) - x0),
            "ph": int(max(q["py"] + q["ph"] for q in linea) - y0)}
    return texto, bbox


def detectar_textos(gris, eje_x_fila, eje_y_col, rect=None, mascaras_curva=None,
                    tol=None, escala_ocr=2.0, lang=None, excluir=None, tokens_pdf=None):
    """OCR global de la imagen y clasificación de cada texto (ejes, títulos, leyenda, datos).

    `tokens_pdf`: si viene (texto real del PDF, ver tokens_de_palabras_pdf),
    se clasifica eso en vez de correr el OCR: es exacto y no depende de
    Tesseract."""
    t = tol or _tolerancias(gris.shape)
    margen = t["margen_texto"]
    radio_dato = t["radio_dato"]
    mapa_dist = _mapa_distancia_curva(mascaras_curva)

    if tokens_pdf is not None:
        datos = {"text": []}
        tokens = list(tokens_pdf)
    else:
        if escala_ocr and escala_ocr != 1:
            gris_ocr = cv2.resize(gris, None, fx=escala_ocr, fy=escala_ocr,
                                  interpolation=cv2.INTER_CUBIC)
        else:
            gris_ocr = gris
            escala_ocr = 1.0

        datos = pytesseract.image_to_data(
            gris_ocr, output_type=Output.DICT,
            config="--psm 11", lang=_idioma_ocr(lang),
        )
        tokens = []

    for i in range(len(datos["text"])):
        texto = datos["text"][i].strip()
        if not texto:
            continue
        try:
            conf = float(datos["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf != -1 and conf < 30:
            continue

        x = int(round(datos["left"][i] / escala_ocr))
        y = int(round(datos["top"][i] / escala_ocr))
        w = int(round(datos["width"][i] / escala_ocr))
        h = int(round(datos["height"][i] / escala_ocr))
        valor = _a_float(texto) if _es_numero(texto) else None

        tokens.append({
            "texto": texto, "valor": valor, "conf": conf,
            "px": x, "py": y, "pw": w, "ph": h,
            "centro_x": x + w / 2.0, "centro_y": y + h / 2.0,
            "orden": (datos["block_num"][i], datos["par_num"][i],
                      datos["line_num"][i], datos["word_num"][i]),
        })

    if excluir:
        def _dentro(tk):
            for b in excluir:
                if (b["px"] - 3 <= tk["centro_x"] <= b["px"] + b["pw"] + 3
                        and b["py"] - 3 <= tk["centro_y"] <= b["py"] + b["ph"] + 3):
                    return True
            return False
        tokens = [tk for tk in tokens if not _dentro(tk)]

    x0, y0, x1, y1 = rect if rect else (None, None, None, None)
    etiquetas_x, etiquetas_y, etiquetas_dato = [], [], []
    tok_titulo, tok_titulo_x, tok_leyenda = [], [], []

    for tk in tokens:
        cx, cy = tk["centro_x"], tk["centro_y"]

        if tk["valor"] is not None:
            cerca_curva = False
            if mapa_dist is not None:
                fila = min(max(int(round(cy)), 0), mapa_dist.shape[0] - 1)
                col = min(max(int(round(cx)), 0), mapa_dist.shape[1] - 1)
                cerca_curva = mapa_dist[fila, col] < radio_dato
            if cerca_curva:
                etiquetas_dato.append(tk)
                continue

            es_x = (
                eje_x_fila is not None and cy > eje_x_fila - margen
                and (eje_y_col is None or cx > eje_y_col - margen)
                and (x1 is None or cx < x1 + margen * 3)
            )
            es_y = (
                eje_y_col is not None and cx < eje_y_col + margen
                and (eje_x_fila is None or cy < eje_x_fila + margen)
                and (y0 is None or cy > y0 - margen * 3)
            )
            if es_x and es_y:      # ambigüedad: gana el eje más cercano
                d_x = abs(cy - eje_x_fila)
                d_y = abs(cx - eje_y_col)
                es_x, es_y = (d_x <= d_y), (d_y < d_x)
            if es_x:
                etiquetas_x.append(tk)
            elif es_y:
                etiquetas_y.append(tk)
            else:
                tok_leyenda.append(tk)
            continue

        limite_superior = y0 if y0 is not None else eje_x_fila
        if limite_superior is not None and cy < limite_superior - margen:
            tok_titulo.append(tk)
        elif eje_x_fila is not None and cy > eje_x_fila + margen:
            tok_titulo_x.append(tk)
        else:
            tok_leyenda.append(tk)

    def _unir(toks):
        if not toks:
            return "", None
        toks = sorted(toks, key=lambda z: z["orden"])
        texto = " ".join(z["texto"] for z in toks).strip()
        bbox = {
            "px": min(z["px"] for z in toks),
            "py": min(z["py"] for z in toks),
        }
        bbox["pw"] = max(z["px"] + z["pw"] for z in toks) - bbox["px"]
        bbox["ph"] = max(z["py"] + z["ph"] for z in toks) - bbox["py"]
        return texto, bbox

    if tok_titulo_x and etiquetas_x:
        base_num = max(e["py"] + e["ph"] for e in etiquetas_x)
        bajo = [z for z in tok_titulo_x if z["centro_y"] > base_num]
        if bajo:
            tok_titulo_x = bajo
    if len(tok_titulo_x) > 1:
        tok_titulo_x.sort(key=lambda z: z["centro_y"])
        h_med = float(np.median([z["ph"] for z in tok_titulo_x])) or 1.0
        y_ref = tok_titulo_x[0]["centro_y"]
        primera = [z for z in tok_titulo_x if z["centro_y"] - y_ref <= 0.8 * h_med]
        tok_leyenda += [z for z in tok_titulo_x if z not in primera]
        tok_titulo_x = primera

    # Eje X de CATEGORÍAS ("Ene Feb Mar...", "Lima Cusco Piura"): si bajo el
    # eje no hay números pero sí una fila de 3+ textos equiespaciados dentro
    # del ancho del gráfico, esos son los rótulos de cada marca, no el título
    # del eje (antes quedaba "Eje horizontal: Ene Feb Mar Abr" y sin escala).
    # El título del eje, si lo hay, es la línea siguiente.
    categorias_x = []
    if len(etiquetas_x) < 2 and tok_titulo_x:
        categorias_x = _categorias_en_linea(tok_titulo_x, x0, x1, margen)
        if categorias_x:
            usados = {id(z) for c in categorias_x for z in c["tokens"]}
            siguientes = [z for z in tok_leyenda if eje_x_fila is not None
                          and z["centro_y"] > eje_x_fila + margen and z["valor"] is None]
            tok_leyenda = [z for z in tok_leyenda if z not in siguientes]
            tok_titulo_x = [z for z in tok_titulo_x if id(z) not in usados] + siguientes
            if tok_titulo_x:
                tok_titulo_x.sort(key=lambda z: z["centro_y"])
                h_med = float(np.median([z["ph"] for z in tok_titulo_x])) or 1.0
                y_ref = tok_titulo_x[0]["centro_y"]
                primera = [z for z in tok_titulo_x if z["centro_y"] - y_ref <= 0.8 * h_med]
                tok_leyenda += [z for z in tok_titulo_x if z not in primera]
                tok_titulo_x = primera

    titulo, titulo_bbox = _unir(tok_titulo)
    titulo_x, titulo_x_bbox = _unir(tok_titulo_x)
    leyenda, _ = _unir(tok_leyenda)
    if not _texto_plausible(titulo):
        titulo, titulo_bbox = "", None
    if not _texto_plausible(titulo_x):
        titulo_x, titulo_x_bbox = "", None

    return {
        "etiquetas_x": etiquetas_x,
        "etiquetas_y": etiquetas_y,
        "etiquetas_dato": etiquetas_dato,
        "titulo": titulo, "titulo_bbox": titulo_bbox,
        "titulo_x": titulo_x, "titulo_x_bbox": titulo_x_bbox,
        "leyenda": leyenda,
        "categorias_x": categorias_x,
        # Todas las palabras con su recuadro: `nombres_de_leyenda` las usa para
        # saber qué nombre de la leyenda va al lado de qué muestra de color.
        "tokens": tokens,
    }


# ---------- 3a) ETIQUETAS NUMÉRICAS DE LOS EJES (lectura por regiones) ----------

_WHITELIST_NUM = "0123456789.,-+%"
_CONFUSIONES = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "l": "1", "I": "1", "|": "1", "i": "1", "!": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9", "q": "9",
})


def _normalizar_numero(texto):
    """Texto listo para _a_float, o None si no es un número plausible."""
    t = texto.strip().strip(".,:;")
    if t.endswith("%"):
        t = t[:-1].strip()
    if not t:
        return None
    if not _es_numero(t):
        t2 = t.replace(" ", "")
        if not _es_numero(t2) and any(c.isdigit() for c in t2):
            t2 = t2.translate(_CONFUSIONES)
        t = t2
    return t if _es_numero(t) else None


def _grosor_hasta_borde(gris, eje, lado, fondo):
    """Fila/columna donde termina el trazo del eje (para no incluirlo en la franja)."""
    alto, ancho = gris.shape[:2]
    tinta = gris < (int(fondo) - 40)
    if lado == "abajo":
        y, a, b = int(eje[1]), int(eje[0]), int(eje[2])
        a, b = max(0, min(a, b)), min(ancho, max(a, b) + 1)
        cand = range(max(0, y - 2), min(alto, y + 3))
        fila = max(cand, key=lambda r: tinta[r, a:b].mean())
        while fila + 1 < alto and tinta[fila + 1, a:b].mean() > 0.5:
            fila += 1
        return fila
    x, a, b = int(eje[0]), int(eje[1]), int(eje[3])
    a, b = max(0, min(a, b)), min(alto, max(a, b) + 1)
    cand = range(max(0, x - 2), min(ancho, x + 3))
    col = max(cand, key=lambda c: tinta[a:b, c].mean())
    while col - 1 >= 0 and tinta[a:b, col - 1].mean() > 0.5:
        col -= 1
    return col


def _candidatos_etiquetas(gris, region, lado_eje):
    """Cajas de posibles etiquetas numéricas en una franja junto al eje."""
    x0, y0, x1, y1 = region
    sub = gris[y0:y1, x0:x1]
    if sub.size == 0 or int(sub.max()) - int(sub.min()) < 40:
        return [], 0
    g = sub if np.median(sub) >= 127 else 255 - sub
    _, tinta = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n, lab, st, _ = cv2.connectedComponentsWithStats(tinta, connectivity=8)
    H, W = tinta.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        toca = (y <= 1) if lado_eje == "arriba" else (x + w >= W - 1)
        comps.append({"i": i, "x": x, "y": y, "w": w, "h": h, "a": a, "toca": toca})

    glifos = [c for c in comps if not c["toca"] and c["h"] >= 3 and c["a"] >= 3]
    if not glifos:
        return [], 0
    h_ref = float(np.median([c["h"] for c in glifos]))

    utiles = []
    for c in comps:
        delgado = min(c["w"], c["h"]) <= 2
        if delgado and max(c["w"], c["h"]) >= 2.5 * h_ref:
            continue                       # resto del eje o del marco
        if c["toca"]:
            if lado_eje == "arriba":
                es_marca = c["w"] <= max(3, 0.4 * h_ref) and c["h"] <= 0.65 * h_ref
            else:
                es_marca = c["h"] <= max(3, 0.4 * h_ref) and c["w"] <= 0.65 * h_ref * 1.6
            if es_marca:
                continue
        utiles.append(c["i"])
    if not utiles:
        return [], h_ref

    limpia = np.isin(lab, utiles).astype(np.uint8) * 255
    r = max(1, int(round(0.25 * h_ref)))
    dil = cv2.dilate(limpia, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 1)))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)

    cajas = []
    for i in range(1, n2):
        x, y, w, h, _a = (int(v) for v in st2[i])
        ys, xs = np.where((lab2[y:y + h, x:x + w] == i) & (limpia[y:y + h, x:x + w] > 0))
        if xs.size == 0:
            continue
        bx, by = x + int(xs.min()), y + int(ys.min())
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if bh < 0.5 * h_ref or bh > 2.2 * h_ref or bw < 2 or bw > 12 * h_ref:
            continue
        cajas.append((x0 + bx, y0 + by, bw, bh))
    cajas.sort(key=lambda c: (c[0], c[1]))
    return cajas, h_ref


def _ocr_numero(crop, margen_px=2):
    """Lee UN número en un recorte pequeño. Devuelve (texto, confianza)."""
    if crop.size == 0:
        return None, 0.0
    if np.median(crop) < 127:
        crop = 255 - crop
    h_tinta = max(crop.shape[0] - 2 * margen_px, 1)
    esc = float(np.clip(42.0 / h_tinta, 3.0, 12.0))
    big = cv2.resize(crop, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
    fondo = int(np.percentile(big, 90))
    pad = int(round(6 * esc))
    big = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fondo)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    intentos = ((7, True), (8, True), (7, False), (10, False))
    mejor, mejor_conf = None, -1.0
    for variante in (big, otsu):
        for psm, con_lista in intentos:
            cfg = f"--oem 3 --psm {psm}"
            if con_lista:
                cfg += f" -c tessedit_char_whitelist={_WHITELIST_NUM}"
            try:
                d = pytesseract.image_to_data(variante, output_type=Output.DICT,
                                              config=cfg, lang="eng")
            except pytesseract.TesseractError:
                continue
            palabras, confs = [], []
            for txt, cf in zip(d["text"], d["conf"]):
                if str(txt).strip():
                    palabras.append(str(txt).strip())
                    try:
                        confs.append(max(float(cf), 0.0))
                    except (TypeError, ValueError):
                        confs.append(0.0)
            if not palabras:
                continue
            num = _normalizar_numero(" ".join(palabras))
            if num is None:
                continue
            conf = sum(confs) / len(confs)
            if conf > mejor_conf:
                mejor, mejor_conf = num, conf
            if mejor_conf >= 80:
                return mejor, mejor_conf
        if mejor is not None and mejor_conf >= 60:
            break

    if mejor is None:
        hueco = np.full((big.shape[0], int(round(big.shape[0] * 0.6))), fondo, dtype=big.dtype)
        doble = np.hstack([big, hueco, big])
        for psm in (7, 8):
            cfg = (f"--oem 3 --psm {psm} -c tessedit_char_whitelist={_WHITELIST_NUM}")
            try:
                d = pytesseract.image_to_data(doble, output_type=Output.DICT,
                                              config=cfg, lang="eng")
            except pytesseract.TesseractError:
                continue
            crudo = "".join(str(z).strip() for z in d["text"] if str(z).strip())
            confs = []
            for z, cf in zip(d["text"], d["conf"]):
                if str(z).strip():
                    try:
                        confs.append(max(float(cf), 0.0))
                    except (TypeError, ValueError):
                        pass
            n = len(crudo)
            conf_media = (sum(confs) / len(confs)) if confs else 0.0
            if n >= 2 and n % 2 == 0 and crudo[:n // 2] == crudo[n // 2:] and conf_media >= 50:
                num = _normalizar_numero(crudo[:n // 2])
                if num is not None:
                    return num, conf_media
    return mejor, max(mejor_conf, 0.0)


def _elegir_alineacion(tokens, clave, tol, distancia_eje):
    """Se queda con el grupo de números mejor alineado (misma fila o columna)."""
    if len(tokens) <= 1:
        return tokens
    orden = sorted(tokens, key=clave)
    grupos, actual = [], [orden[0]]
    for tk in orden[1:]:
        if clave(tk) - clave(actual[0]) <= tol:
            actual.append(tk)
        else:
            grupos.append(actual)
            actual = [tk]
    grupos.append(actual)
    return max(grupos, key=lambda g: (len(g), -min(distancia_eje(z) for z in g)))


def detectar_etiquetas_ejes(gris, eje_x, eje_y, rect=None, tol=None):
    """Lee los números de ambos ejes por regiones. Devuelve (etiquetas_x, etiquetas_y)."""
    t = tol or _tolerancias(gris.shape)
    alto, ancho = gris.shape[:2]
    margen = t["margen_texto"]
    fondo = int(np.median(gris))

    def _tokens(cajas, orden0):
        out = []
        for k, (x, y, w, h) in enumerate(cajas[:40]):
            c = 2
            crop = gris[max(0, y - c):min(alto, y + h + c), max(0, x - c):min(ancho, x + w + c)]
            num, conf = _ocr_numero(crop)
            if num is None:
                continue
            valor = _a_float(num)
            if valor is None:
                continue
            out.append({
                "texto": num, "valor": valor, "conf": conf,
                "px": x, "py": y, "pw": w, "ph": h,
                "centro_x": x + w / 2.0, "centro_y": y + h / 2.0,
                "orden": (orden0, 0, 0, k),
            })
        return out

    etiquetas_x = []
    if eje_x is not None:
        borde = _grosor_hasta_borde(gris, eje_x, "abajo", fondo)
        xa = max(0, int(eje_x[0]) - int(2.5 * margen))
        xb = min(ancho, int(eje_x[2]) + int(5 * margen))
        ya = min(alto - 1, borde + 1)
        yb = min(alto, borde + 1 + max(40, int(0.25 * alto)))
        cajas, h_ref = _candidatos_etiquetas(gris, (xa, ya, xb, yb), "arriba")
        toks = _tokens(cajas, 1)
        if toks:
            tol_fila = max(3.0, 0.6 * h_ref)
            etiquetas_x = _elegir_alineacion(
                toks, lambda z: z["centro_y"], tol_fila,
                lambda z: z["centro_y"] - borde)

    etiquetas_y = []
    if eje_y is not None:
        borde = _grosor_hasta_borde(gris, eje_y, "izq", fondo)
        xb = max(0, borde)
        xa = max(0, xb - max(40, int(0.35 * ancho)))
        ya = max(0, int(eje_y[1]) - 3 * margen)
        yb = min(alto, (int(eje_x[1]) if eje_x else int(eje_y[3])) + 3 * margen + 1)
        cajas, h_ref = _candidatos_etiquetas(gris, (xa, ya, xb, yb), "derecha")
        toks = _tokens(cajas, 2)
        if toks:
            tol_borde = max(2.0, 0.5 * h_ref)
            etiquetas_y = _elegir_alineacion(
                toks, lambda z: -(z["px"] + z["pw"]), tol_borde,
                lambda z: borde - (z["px"] + z["pw"]))

    etiquetas_x.sort(key=lambda z: z["centro_x"])
    etiquetas_y.sort(key=lambda z: z["centro_y"])
    return etiquetas_x, etiquetas_y


_RE_LETRA = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\u0370-\u03FF\u2126\u00B5]"


def _texto_plausible(texto):
    """True si el texto es mayoritariamente letras (descarta basura del OCR)."""
    if not texto:
        return False
    toks = [w for w in texto.split() if re.search(_RE_LETRA[:-1] + r"0-9]", w)]
    limpio = "".join(toks)
    if not limpio:
        return False
    letras = len(re.findall(_RE_LETRA, limpio))
    return letras >= 1 and letras / len(limpio) >= 0.5


def _limite_titulo_vertical(gris, eje_y, eje_x, fondo, margen):
    """x donde termina el título del eje Y cuando no se leyeron sus números."""
    alto, ancho = gris.shape[:2]
    borde = _grosor_hasta_borde(gris, eje_y, "izq", fondo)
    ya = max(0, int(eje_y[1]) - 3 * margen)
    yb = min(alto, (int(eje_x[1]) if eje_x else int(eje_y[3])) + 3 * margen + 1)
    franja = gris[ya:yb, :max(borde, 1)] < (fondo - 40)
    col = franja.any(axis=0)
    if not col.any():
        return None
    hueco_min = max(3, int(round(0.008 * ancho)))
    runs, ini, vacias = [], None, 0
    for x, c in enumerate(col):
        if c:
            if ini is None:
                ini = x
            vacias = 0
            fin = x
        elif ini is not None:
            vacias += 1
            if vacias >= hueco_min:
                runs.append((ini, fin))
                ini = None
    if ini is not None:
        runs.append((ini, fin))
    if len(runs) < 2:
        return None
    return runs[-1][0]              # inicio de la banda pegada al eje


# ---------- 3b) TEXTO VERTICAL (título del eje Y, rotado 90°) ----------

def detectar_texto_vertical(gris, x_limite, margen=6, ancho_minimo=15, lang=None):
    """Lee el título del eje Y (texto girado 90°) a la izquierda de x_limite."""
    x_limite = int(x_limite) - margen
    if x_limite < ancho_minimo:
        return "", None

    franja0 = gris[:, :x_limite]
    if franja0.size == 0:
        return "", None
    esc = float(np.clip(1200.0 / max(franja0.shape[0], 1), 1.5, 4.0))
    franja = cv2.resize(franja0, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
    alto_franja, ancho_franja = franja.shape[:2]

    idioma = _idioma_ocr(lang)
    mejor_texto, mejor_bbox, mejor_puntaje = "", None, -1.0

    for rotacion in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotada = cv2.rotate(franja, rotacion)
        datos = pytesseract.image_to_data(rotada, output_type=Output.DICT,
                                          config="--psm 6", lang=idioma)
        palabras, confs, cajas = [], [], []
        for i in range(len(datos["text"])):
            texto = datos["text"][i].strip()
            if not texto:
                continue
            try:
                conf = float(datos["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            if conf < 30:
                continue

            x, y = datos["left"][i], datos["top"][i]
            w, h = datos["width"][i], datos["height"][i]
            if rotacion == cv2.ROTATE_90_CLOCKWISE:
                px, py = y, alto_franja - (x + w)
            else:
                px, py = ancho_franja - (y + h), x
            palabras.append(texto)
            confs.append(conf)
            cajas.append((px, py, h, w))

        if not palabras:
            continue
        texto_final = " ".join(palabras)
        if not _texto_plausible(texto_final):
            continue

        puntaje = (sum(confs) / len(confs)) * math.sqrt(len(texto_final))
        if puntaje > mejor_puntaje:
            xs1 = [c[0] for c in cajas]
            ys1 = [c[1] for c in cajas]
            xs2 = [c[0] + c[2] for c in cajas]
            ys2 = [c[1] + c[3] for c in cajas]
            mejor_bbox = {"px": int(min(xs1) / esc), "py": int(min(ys1) / esc),
                          "pw": int(math.ceil((max(xs2) - min(xs1)) / esc)),
                          "ph": int(math.ceil((max(ys2) - min(ys1)) / esc))}
            mejor_texto, mejor_puntaje = texto_final, puntaje

    return mejor_texto, mejor_bbox


# ---------- 3c) TEXTOS CORTOS Y SÍMBOLOS: título del eje X y leyenda ----------

_SIMBOLOS_PLANTILLA = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "αβγδεζηθικλμνξπρστυφχψω"      # griego minúscula
    "ΓΔΘΛΞΠΣΦΨΩ"                   # griego mayúscula (las demás = latín)
    "°%±∞"
)

_FUENTES_CANDIDATAS = [
    r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]

_CACHE_PLANTILLAS = None


def _cargar_plantillas(max_fuentes=6, tam=64):
    """Dibuja cada símbolo con las fuentes del sistema (una vez por proceso)."""
    global _CACHE_PLANTILLAS
    if _CACHE_PLANTILLAS is not None:
        return _CACHE_PLANTILLAS
    _CACHE_PLANTILLAS = []
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return _CACHE_PLANTILLAS

    extra = [r for r in os.environ.get("SEGMENTADOR_FUENTES", "").split(";") if r.strip()]
    rutas = [r for r in extra + _FUENTES_CANDIDATAS if os.path.isfile(r)][:max_fuentes]

    def _dibujar(fuente, ch):
        img = Image.new("L", (tam * 3, tam * 3), 0)
        ImageDraw.Draw(img).text((tam // 2, tam // 2), ch, fill=255, font=fuente)
        a = np.asarray(img)
        ys, xs = np.nonzero(a > 40)
        if xs.size == 0:
            return None
        return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0

    for ruta in rutas:
        try:
            fuente = ImageFont.truetype(ruta, tam)
        except Exception:
            continue
        tofu = _dibujar(fuente, "\U0010FFFD")
        for ch in _SIMBOLOS_PLANTILLA:
            g = _dibujar(fuente, ch)
            if g is None:
                continue
            if tofu is not None and g.shape == tofu.shape and np.allclose(g, tofu):
                continue
            _CACHE_PLANTILLAS.append((ch, g))
    return _CACHE_PLANTILLAS


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / den if den > 0 else 0.0


def _clasificar_glifo(tinta):
    """Compara un glifo con las plantillas. Devuelve [(símbolo, puntaje)], mejor primero."""
    plantillas = _cargar_plantillas()
    h, w = tinta.shape
    if not plantillas or h < 4 or w < 1:
        return []
    mejores = {}
    for ch, tpl in plantillas:
        th, tw = tpl.shape
        wt = max(1, int(round(tw * h / th)))
        if wt > 4 * w + 2 or w > 4 * wt + 2:
            continue
        t_red = cv2.resize(tpl, (wt, h), interpolation=cv2.INTER_AREA)
        W = max(w, wt) + 2
        H = h + 2
        lienzo_s = np.zeros((H, W), np.float32)
        xs = (W - w) // 2
        lienzo_s[1:1 + h, xs:xs + w] = tinta
        mejor = -1.0
        for dy in (0, 1, 2):
            for dx in (-1, 0, 1):
                xt = (W - wt) // 2 + dx
                if xt < 0 or xt + wt > W:
                    continue
                lienzo_t = np.zeros((H, W), np.float32)
                lienzo_t[dy:dy + h, xt:xt + wt] = t_red
                mejor = max(mejor, _ncc(lienzo_s, lienzo_t))
        r = min(w, wt) / max(w, wt)
        puntaje = mejor - 0.5 * (1.0 - r)
        if puntaje > mejores.get(ch, -9):
            mejores[ch] = puntaje
    return sorted(mejores.items(), key=lambda kv: -kv[1])


def _idioma_ocr_simbolos(lang=None):
    """Como _idioma_ocr, añadiendo griego (ell/grc) si está instalado."""
    base = _idioma_ocr(lang)
    try:
        disponibles = set(pytesseract.get_languages(config=""))
    except Exception:
        return base
    for extra in ("ell", "grc"):
        if extra in disponibles and extra not in base.split("+"):
            return base + "+" + extra
    return base


def _tinta_de_recorte(crop):
    """Recorte en gris -> (tinta 0..1, máscara binaria)."""
    c = crop.astype(np.float32)
    fondo = float(np.percentile(c, 90)) if np.median(c) >= 127 else float(np.percentile(c, 10))
    tinta = np.abs(fondo - c)
    mx = float(tinta.max())
    if mx < 30:
        return None, None
    tinta /= mx
    return tinta, tinta > 0.35


def _glifos(binaria):
    """Cajas (x, y, w, h) de cada glifo."""
    n, _, st, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), connectivity=8)
    comps = sorted((tuple(int(v) for v in st[i][:4]) for i in range(1, n)
                    if st[i][cv2.CC_STAT_AREA] >= 2), key=lambda c: c[0])
    unidos = []
    for x, y, w, h in comps:
        if unidos:
            ux, uy, uw, uh = unidos[-1]
            solape = min(ux + uw, x + w) - max(ux, x)
            if solape >= 0.5 * min(uw, w):
                nx, ny = min(ux, x), min(uy, y)
                unidos[-1] = (nx, ny, max(ux + uw, x + w) - nx, max(uy + uh, y + h) - ny)
                continue
        unidos.append((x, y, w, h))
    return unidos


def _leer_texto_corto(gris, caja, lang=None, umbral_plantilla=0.55):
    """Lee una palabra, letra o símbolo dentro de caja. Devuelve (texto, conf, método)."""
    alto, ancho = gris.shape[:2]
    x, y, w, h = caja
    c = 2
    crop = gris[max(0, y - c):min(alto, y + h + c), max(0, x - c):min(ancho, x + w + c)]
    if crop.size == 0:
        return "", 0.0, None
    tinta, binaria = _tinta_de_recorte(crop)
    if tinta is None:
        return "", 0.0, None
    ox, oy = x - max(0, x - c), y - max(0, y - c)
    dentro = np.zeros_like(binaria)
    dentro[oy:oy + h, ox:ox + w] = True
    tinta = np.where(dentro, tinta, 0.0).astype(np.float32)
    binaria &= dentro
    glifos = _glifos(binaria)
    if not glifos:
        return "", 0.0, None

    ys, xs = np.nonzero(binaria)
    h_tinta = int(ys.max() - ys.min() + 1)
    esc = float(np.clip(48.0 / max(h_tinta, 1), 2.0, 12.0))
    img = (255 - np.clip(tinta * 255, 0, 255)).astype(np.uint8)       # tinta negra
    big = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
    pad = int(round(8 * esc))
    big = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    idioma = _idioma_ocr_simbolos(lang)
    psms = (10, 8, 7) if len(glifos) == 1 else (7, 8, 6)
    ocr_txt, ocr_conf = "", -1.0
    for psm in psms:
        try:
            d = pytesseract.image_to_data(big, output_type=Output.DICT,
                                          config=f"--oem 3 --psm {psm}", lang=idioma)
        except pytesseract.TesseractError:
            continue
        pal, cf = [], []
        for t_, c_ in zip(d["text"], d["conf"]):
            if str(t_).strip():
                pal.append(str(t_).strip())
                try:
                    cf.append(max(float(c_), 0.0))
                except (TypeError, ValueError):
                    cf.append(0.0)
        if pal:
            conf = sum(cf) / len(cf)
            if conf > ocr_conf:
                ocr_txt, ocr_conf = " ".join(pal), conf

    if len(glifos) > 1:
        if ocr_txt and ocr_conf >= 50 and _texto_plausible(ocr_txt):
            return ocr_txt, ocr_conf, "ocr"
        if len(glifos) <= 4:
            letras, puntajes = [], []
            for gx, gy, gw, gh in glifos:
                rk = _clasificar_glifo(tinta[gy:gy + gh, gx:gx + gw])
                if not rk or rk[0][1] < umbral_plantilla:
                    break
                letras.append(rk[0][0])
                puntajes.append(rk[0][1])
            else:
                return "".join(letras), 100.0 * min(puntajes), "plantilla"
        if ocr_txt and _texto_plausible(ocr_txt):
            return ocr_txt, ocr_conf, "ocr"
        return "", 0.0, None

    gx, gy, gw, gh = glifos[0]
    ranking = _clasificar_glifo(tinta[gy:gy + gh, gx:gx + gw])
    if not ranking:
        return (ocr_txt, ocr_conf, "ocr_sin_verificar") if ocr_txt else ("", 0.0, None)
    mejor_ch, mejor_p = ranking[0]
    puntajes = dict(ranking)
    if len(ocr_txt) == 1 and ocr_conf >= 50:
        if puntajes.get(ocr_txt, -9) >= mejor_p - 0.03:
            return ocr_txt, ocr_conf, "ocr"
    if mejor_p >= umbral_plantilla:
        return mejor_ch, 100.0 * mejor_p, "plantilla"
    if ocr_txt and ocr_conf >= 60 and _texto_plausible(ocr_txt):
        return ocr_txt, ocr_conf, "ocr"
    return "", 0.0, None


def _lineas_de_texto(gris, region, fondo=None):
    """Agrupa la tinta de una región en cajas de texto (x, y, w, h)."""
    x0, y0, x1, y1 = (int(v) for v in region)
    sub = gris[y0:y1, x0:x1]
    if sub.size == 0:
        return []
    if fondo is None:
        fondo = float(np.median(gris))
    tinta = (np.abs(sub.astype(np.int16) - int(fondo)) > 60).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(tinta, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < 2:
            continue
        if min(w, h) <= 2 and max(w, h) >= 15:          # marco / rejilla / eje
            continue
        comps.append([x, y, w, h])
    if not comps:
        return []
    h_ref = float(np.median([c[3] for c in comps]))
    comps = [c for c in comps if c[3] <= 3.0 * h_ref + 2 and c[2] <= 40 * h_ref]
    if not comps:
        return []

    comps.sort(key=lambda c: c[1] + c[3] / 2.0)
    lineas = []
    for c in comps:
        cy = c[1] + c[3] / 2.0
        for L in lineas:
            if L["y0"] - 0.3 * h_ref <= cy <= L["y1"] + 0.3 * h_ref:
                L["c"].append(c)
                L["y0"] = min(L["y0"], c[1])
                L["y1"] = max(L["y1"], c[1] + c[3])
                break
        else:
            lineas.append({"y0": c[1], "y1": c[1] + c[3], "c": [c]})

    cajas = []
    hueco_max = max(4.0, 1.5 * h_ref)
    for L in sorted(lineas, key=lambda L: L["y0"]):
        cs = sorted(L["c"], key=lambda c: c[0])
        grupo = [cs[0]]
        for c in cs[1:]:
            fin = max(g[0] + g[2] for g in grupo)
            if c[0] - fin <= hueco_max:
                grupo.append(c)
            else:
                cajas.append(grupo)
                grupo = [c]
        cajas.append(grupo)
    out = []
    for g in cajas:
        gx0 = min(c[0] for c in g)
        gy0 = min(c[1] for c in g)
        gx1 = max(c[0] + c[2] for c in g)
        gy1 = max(c[1] + c[3] for c in g)
        out.append((x0 + gx0, y0 + gy0, gx1 - gx0, gy1 - gy0))
    out.sort(key=lambda b: (b[1], b[0]))
    return out


def _solapa(a, b, holgura=2):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + holgura < bx or bx + bw + holgura < ax or
                ay + ah + holgura < by or by + bh + holgura < ay)


def detectar_leyenda(imagen_bgr, gris, rect, series, tol=None, lang=None,
                     sat_minima=60, tolerancia_hue=12, advertencias=None):
    """Busca la muestra de color de cada serie y lee el texto a su derecha."""
    alto, ancho = gris.shape[:2]
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    fondo = float(np.median(gris))
    entradas = []

    for serie in series:
        hue = serie.get("hue")
        if hue is None:
            continue
        m = _mascara_por_hue(h, (s > sat_minima) & (v > 40), hue, tolerancia_hue)
        if "mascara" in serie:            # fuera la propia curva (y un poco alrededor)
            curva = cv2.dilate(serie["mascara"].astype(np.uint8), np.ones((5, 5), np.uint8))
            m &= curva == 0
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        candidatas = []
        for i in range(1, n):
            x, y, w, hh, a = (int(z) for z in st[i])
            if a < 6 or hh > 0.06 * alto + 3:
                continue
            es_trazo = w >= 2.5 * hh and 6 <= w <= 0.3 * ancho
            es_cuadro = 0.6 <= w / max(hh, 1) <= 1.7 and 4 <= w <= 0.05 * ancho and a >= 0.6 * w * hh
            if es_trazo or es_cuadro:
                candidatas.append((x, y, w, hh))

        for (mx, my, mw, mh) in candidatas:
            cy = my + mh / 2.0
            banda = max(8, int(round(0.04 * alto)), 2 * mh)
            region = (min(ancho - 1, mx + mw + 1), max(0, int(cy - banda)),
                      min(ancho, mx + mw + 1 + int(0.45 * ancho)), min(alto, int(cy + banda) + 1))
            if region[2] - region[0] < 3:
                continue
            lineas = _lineas_de_texto(gris, region, fondo)
            hueco_max = max(10, 4 * banda)
            lineas = [b for b in lineas
                      if b[1] <= cy <= b[1] + b[3] + 2 and b[0] - (mx + mw) <= hueco_max]
            if not lineas:
                continue
            caja = min(lineas, key=lambda b: b[0])
            texto, conf, metodo = _leer_texto_corto(gris, caja, lang=lang)
            if not texto:
                if advertencias is not None:
                    advertencias.append(
                        f"Se encontró la muestra de la leyenda en ({mx},{my}) pero no se "
                        "pudo leer el texto de al lado; corrígelo a mano."
                    )
                continue
            entradas.append({
                "hue": hue, "texto": texto, "conf": conf, "metodo": metodo,
                "bbox": {"px": caja[0], "py": caja[1], "pw": caja[2], "ph": caja[3]},
                "muestra_bbox": {"px": mx, "py": my, "pw": mw, "ph": mh},
            })
            break                           # una entrada por serie
    return entradas


def detectar_leyenda_sin_muestra(gris, rect, tol=None, lang=None, excluir_cajas=()):
    """Respaldo: lee el texto a la derecha del área de dibujo."""
    t = tol or _tolerancias(gris.shape)
    alto, ancho = gris.shape[:2]
    x0, y0, x1, y1 = rect
    region = (min(ancho - 1, x1 + t["margen_texto"]), max(0, y0 - t["margen_texto"]),
              ancho, min(alto, y1 + t["margen_texto"]))
    if region[2] - region[0] < 6:
        return []
    entradas = []
    for caja in _lineas_de_texto(gris, region):
        if any(_solapa(caja, e) for e in excluir_cajas):
            continue
        texto, conf, metodo = _leer_texto_corto(gris, caja, lang=lang)
        if texto and _texto_plausible(texto):
            entradas.append({"hue": None, "texto": texto, "conf": conf, "metodo": metodo,
                             "bbox": {"px": caja[0], "py": caja[1], "pw": caja[2], "ph": caja[3]},
                             "muestra_bbox": None})
    return entradas


def detectar_titulo_eje_x_por_region(gris, eje_x, etiquetas_x, rect, tol=None,
                                     lang=None, excluir_cajas=()):
    """Primera línea de texto bajo los números del eje X. Devuelve (texto, bbox)."""
    if eje_x is None:
        return "", None
    t = tol or _tolerancias(gris.shape)
    alto, ancho = gris.shape[:2]
    margen = t["margen_texto"]
    if etiquetas_x:
        y_ini = max(e["py"] + e["ph"] for e in etiquetas_x) + 2
    else:
        y_ini = int(eje_x[1]) + 3 * margen
    x0 = max(0, min(int(eje_x[0]), rect[0]) - 3 * margen)
    x1 = min(ancho, max(int(eje_x[2]), rect[2]) + 3 * margen)
    region = (x0, min(alto - 1, y_ini), x1, alto)
    if region[3] - region[1] < 4:
        return "", None
    numeros = [(e["px"], e["py"], e["pw"], e["ph"]) for e in etiquetas_x]
    for caja in _lineas_de_texto(gris, region):
        if any(_solapa(caja, e) for e in list(excluir_cajas) + numeros):
            continue
        texto, conf, metodo = _leer_texto_corto(gris, caja, lang=lang)
        if texto and _texto_plausible(texto):
            return texto, {"px": caja[0], "py": caja[1], "pw": caja[2], "ph": caja[3],
                           "conf": conf, "metodo": metodo}
        break          # solo la línea más cercana a los números
    return "", None


# ---------- 4) CALIBRACIÓN LINEAL ROBUSTA ----------

def _r2(p, v):
    p, v = np.asarray(p, float), np.asarray(v, float)
    if len(p) < 3 or np.ptp(v) == 0:
        return None
    m, b = np.polyfit(p, v, 1)
    res = v - (m * p + b)
    return 1.0 - float((res ** 2).sum()) / float(((v - v.mean()) ** 2).sum())


def es_escala_log(pixeles, valores):
    """¿Los números del eje están en escala logarítmica? (1, 10, 100, 1000
    equiespaciados): todos positivos, al menos 3 y 3 órdenes de magnitud no
    necesarios, pero el logaritmo ajusta a una recta claramente mejor que
    los valores tal cual."""
    v = np.asarray(valores, float)
    if len(v) < 3 or (v <= 0).any() or v.max() / v.min() < 20:
        return False
    r2_log = _r2(pixeles, np.log10(v))
    r2_lin = _r2(pixeles, v)
    return r2_log is not None and r2_lin is not None and r2_log > 0.999 and r2_lin < 0.98


def calibrar_eje(pixeles, valores):
    """ajustar_lineal_robusto(), en escala lineal o logarítmica según los
    números del eje. En un eje logarítmico la recta se ajusta sobre
    log10(valor) y info["escala"] = "log": valor = 10 ** (m * pixel + b).
    Antes un eje 1-10-100-1000 se calibraba como lineal y los valores de la
    curva salían muy mal."""
    if es_escala_log(pixeles, valores):
        m, b, info = ajustar_lineal_robusto(pixeles, [math.log10(v) for v in valores])
        info["escala"] = "log"
        if info.get("descartados"):
            info["descartados"] = [round(10 ** d, 6) for d in info["descartados"]]
        return m, b, info
    m, b, info = ajustar_lineal_robusto(pixeles, valores)
    info["escala"] = "lineal"
    return m, b, info


def ajustar_lineal_robusto(pixeles, valores, umbral_rel=0.04):
    """Ajuste lineal tipo RANSAC que descarta números mal leídos. Devuelve (m, b, info)."""
    p = np.asarray(pixeles, dtype=float)
    v = np.asarray(valores, dtype=float)
    info = {"n_total": int(p.size), "n_usados": 0, "r2": None, "descartados": [],
            "usados": []}

    if p.size < 2 or np.unique(p).size < 2 or np.unique(v).size < 2:
        return None, None, info

    if p.size == 2:
        m, b = np.polyfit(p, v, 1)
        info.update(n_usados=2, r2=1.0, usados=[True, True])
        return float(m), float(b), info

    rango = float(np.ptp(v)) or 1.0
    umbral = umbral_rel * rango
    mejor_inliers = None

    for i in range(p.size):
        for j in range(i + 1, p.size):
            if p[j] == p[i]:
                continue
            m = (v[j] - v[i]) / (p[j] - p[i])
            b = v[i] - m * p[i]
            inliers = np.abs(m * p + b - v) <= umbral
            if mejor_inliers is None or inliers.sum() > mejor_inliers.sum():
                mejor_inliers = inliers

    if mejor_inliers is None or mejor_inliers.sum() < 2:
        return None, None, info

    m, b = np.polyfit(p[mejor_inliers], v[mejor_inliers], 1)
    pred = m * p[mejor_inliers] + b
    ss_res = float(np.sum((v[mejor_inliers] - pred) ** 2))
    ss_tot = float(np.sum((v[mejor_inliers] - v[mejor_inliers].mean()) ** 2))
    info.update(
        n_usados=int(mejor_inliers.sum()),
        r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        descartados=[float(x) for x in v[~mejor_inliers]],
        usados=[bool(x) for x in mejor_inliers],
    )
    return float(m), float(b), info


def _variantes_numero(texto):
    """Lecturas alternativas de un número (punto o cero perdidos/sobrantes)."""
    neg = texto.strip().startswith(("-", "−", "–"))
    d = re.sub(r"[^0-9]", "", texto)
    if not d:
        return []
    cands = set()
    for k in range(1, len(d)):
        cands.add(d[:k] + "." + d[k:])
    cands.add("0." + d)
    cands.add(d)
    cands.add(d + "0")
    if len(d) > 1:
        cands.add(d[:-1])
    out = []
    for c in cands:
        v = _a_float(("-" if neg else "") + c)
        if v is not None:
            out.append((c, v))
    return out


def rescatar_etiquetas(etiquetas, clave_px):
    """Corrige etiquetas incoherentes con la escala si una variante encaja."""
    if len(etiquetas) < 4:
        return []
    px = [clave_px(e) for e in etiquetas]
    vals = [e["valor"] for e in etiquetas]
    m, b, info = ajustar_lineal_robusto(px, vals)
    if m is None or info["n_usados"] < 3 or not info["usados"]:
        return []
    buenos = np.array([v for v, u in zip(vals, info["usados"]) if u], dtype=float)
    tol = 0.04 * float(np.ptp(buenos))
    if tol <= 0:
        return []
    hechos = []
    for e, p_, usado in zip(etiquetas, px, info["usados"]):
        if usado:
            continue
        pred = m * p_ + b
        mejor = min(_variantes_numero(e["texto"]), key=lambda c: abs(c[1] - pred), default=None)
        if mejor is not None and abs(mejor[1] - pred) <= tol:
            hechos.append((e["texto"], mejor[0]))
            e["texto"], e["valor"], e["corregida"] = mejor[0], mejor[1], True
    return hechos


# ---------- 5) MÁSCARA -> POLILÍNEA ----------

def mascara_a_polilinea(mascara, n_puntos=300, ventana_mediana=5):
    """Máscara -> (cols, filas_suavizadas, cols_remuestreadas, filas_remuestreadas)."""
    filas, cols = np.where(mascara)
    if cols.size == 0:
        return None

    orden = np.argsort(cols, kind="stable")
    cols_o, filas_o = cols[orden], filas[orden].astype(float)
    cols_u, inicios = np.unique(cols_o, return_index=True)
    sumas = np.add.reduceat(filas_o, inicios)
    cuentas = np.diff(np.append(inicios, filas_o.size))
    filas_med = sumas / cuentas

    filas_suav = _mediana_movil(filas_med, ventana_mediana)

    if cols_u.size >= 2 and n_puntos >= 2:
        cols_rs = np.linspace(float(cols_u[0]), float(cols_u[-1]), int(n_puntos))
        filas_rs = np.interp(cols_rs, cols_u.astype(float), filas_suav)
    else:
        cols_rs = cols_u.astype(float)
        filas_rs = filas_suav

    return cols_u.astype(float), filas_suav, cols_rs, filas_rs


def simplificar_polilinea(puntos, tolerancia):
    """Ramer–Douglas–Peucker: deja solo los vértices donde cambia la pendiente."""
    p = np.asarray(puntos, dtype=float)
    if len(p) <= 2 or tolerancia <= 0:
        return p
    conservar = np.zeros(len(p), dtype=bool)
    conservar[0] = conservar[-1] = True
    pila = [(0, len(p) - 1)]
    while pila:                                   # iterativo: sin límite de recursión
        i, j = pila.pop()
        if j - i < 2:
            continue
        a, b = p[i], p[j]
        seg = b - a
        largo = math.hypot(seg[0], seg[1])
        tramo = p[i + 1:j]
        if largo == 0:
            dist = np.hypot(tramo[:, 0] - a[0], tramo[:, 1] - a[1])
        else:
            dist = np.abs(seg[0] * (tramo[:, 1] - a[1]) - seg[1] * (tramo[:, 0] - a[0])) / largo
        k = int(np.argmax(dist))
        if dist[k] > tolerancia:
            m = i + 1 + k
            conservar[m] = True
            pila.append((i, m))
            pila.append((m, j))
    return p[conservar]


def densificar_polilinea(puntos, n_puntos=300):
    """Reparte ~n_puntos sobre la polilínea conservando sus vértices."""
    p = np.asarray(puntos, dtype=float).reshape(-1, 2)
    if len(p) < 2 or not n_puntos or len(p) >= n_puntos:
        return [tuple(q) for q in p]
    largos = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    total = float(largos.sum())
    if total == 0:
        return [tuple(q) for q in p]
    salida = [tuple(p[0])]
    disponibles = n_puntos - len(p)
    for i, largo in enumerate(largos):
        extra = int(round(disponibles * largo / total))
        for k in range(1, extra + 1):
            f = k / (extra + 1)
            salida.append(tuple(p[i] + f * (p[i + 1] - p[i])))
        salida.append(tuple(p[i + 1]))
    return salida


def tolerancia_simplificacion(forma, rect=None):
    """Desvío máximo (px) para simplificar: 0,6 % de la diagonal del área de dibujo."""
    if rect is not None:
        diag = math.hypot(rect[2] - rect[0], rect[3] - rect[1])
    else:
        diag = math.hypot(forma[0], forma[1])
    return max(1.5, 0.006 * diag)


# ---------- 6) PIPELINE COMPLETO ----------

def _bbox_json(b):
    """Caja {px, py, pw, ph} con enteros, lista para JSON."""
    if not b:
        return None
    out = {k: int(round(float(b[k]))) for k in ("px", "py", "pw", "ph") if k in b}
    if b.get("manual"):
        out["manual"] = True
    return out


def _bbox_desde_posicion(pos, forma):
    """Punto marcado en la pizarra -> caja pequeña centrada en él."""
    if not pos or len(pos) < 2:
        return None
    alto, ancho = forma[:2]
    lado = max(6, int(round(0.03 * min(alto, ancho))))
    cx = min(max(float(pos[0]), 0), ancho - 1)
    cy = min(max(float(pos[1]), 0), alto - 1)
    return {"px": int(round(cx - lado / 2)), "py": int(round(cy - lado / 2)),
            "pw": lado, "ph": lado, "manual": True}


def _dibujar_texto(img, texto, centro, color_bgr, alto_px):
    """Escribe `texto` (admite Ω, Δ, μ...) centrado en `centro`, con fondo blanco y borde."""
    alto, ancho = img.shape[:2]
    cx, cy = centro
    try:
        from PIL import Image, ImageDraw, ImageFont
        rutas = [r for r in _FUENTES_CANDIDATAS if os.path.isfile(r)]
        fuente = ImageFont.truetype(rutas[0], alto_px) if rutas else ImageFont.load_default()
        pil = Image.fromarray(img[:, :, ::-1])
        dib = ImageDraw.Draw(pil)
        x0, y0, x1, y1 = dib.textbbox((0, 0), texto, font=fuente)
        w, h = x1 - x0, y1 - y0
        x = int(min(max(cx - w / 2, 3), ancho - w - 3))
        y = int(min(max(cy - h / 2, 3), alto - h - 3))
        color_rgb = tuple(int(c) for c in color_bgr[::-1])
        dib.rectangle((x - 3, y - 3, x + w + 2, y + h + 2), fill=(255, 255, 255), outline=color_rgb)
        dib.text((x - x0, y - y0), texto, font=fuente, fill=color_rgb)
        img[:] = np.asarray(pil)[:, :, ::-1]
    except Exception:
        # sin Pillow: solo caracteres ASCII
        t = texto.encode("ascii", "replace").decode()
        esc = alto_px / 30.0
        (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, esc, 1)
        x, y = int(cx - w / 2), int(cy + h / 2)
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + 3), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + 3), color_bgr, 1)
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, color_bgr, 1, cv2.LINE_AA)


def _calibrar_y_exportar(imagen, ruta_imagen, dir_resultados, uid,
                          eje_x, eje_y, rect, series, txt,
                          etiquetas_x, etiquetas_y, etiquetas_dato,
                          titulo_eje_y, titulo_eje_y_bbox,
                          advertencias, n_puntos=300, simplificar=False):
    """Calibra píxel->valor y escribe JSON, CSV y overlay."""
    alto_imagen, ancho_imagen = imagen.shape[:2]

    m_x, b_x, info_x = calibrar_eje(
        [e["centro_x"] for e in etiquetas_x], [e["valor"] for e in etiquetas_x])
    m_y, b_y, info_y = calibrar_eje(
        [e["centro_y"] for e in etiquetas_y], [e["valor"] for e in etiquetas_y])

    calibrado_y_por_datos = False
    mascaras = [s["mascara"] for s in series if "mascara" in s]
    if m_y is None and etiquetas_dato and mascaras:
        union = np.zeros_like(mascaras[0], dtype=bool)
        for m in mascaras:
            union |= m
        _, cols_c = np.where(union)
        if cols_c.size:
            pl = mascara_a_polilinea(union, n_puntos=0)
            if pl is not None:
                cols_u, filas_med = pl[0], pl[1]
                pares_fila, pares_valor = [], []
                for e in etiquetas_dato:
                    idx = int(np.clip(np.searchsorted(cols_u, e["centro_x"]),
                                      0, cols_u.size - 1))
                    pares_fila.append(float(filas_med[idx]))
                    pares_valor.append(e["valor"])
                m_y, b_y, info_y = calibrar_eje(pares_fila, pares_valor)
                calibrado_y_por_datos = m_y is not None
                if calibrado_y_por_datos:
                    advertencias.append(
                        "Eje Y calibrado con las etiquetas impresas sobre la curva, "
                        "no con las marcas del eje: revisar antes de imprimir."
                    )

    if m_x is None:
        advertencias.append("Sin calibración del eje X (faltan etiquetas numéricas legibles).")
    if m_y is None:
        advertencias.append("Sin calibración del eje Y (faltan etiquetas numéricas legibles).")
    if m_y is not None and m_y > 0:
        advertencias.append(
            "La calibración del eje Y sale creciente hacia abajo: probable error de OCR "
            "en los números del eje."
        )
    for nombre, info in (("X", info_x), ("Y", info_y)):
        if info.get("escala") == "log":
            advertencias.append(f"Eje {nombre} en escala logarítmica (cada marca multiplica el valor).")
        if info["r2"] is not None and info["r2"] < 0.995:
            advertencias.append(
                f"El eje {nombre} no ajusta bien a una recta (R²={info['r2']:.3f}); "
                "¿escala logarítmica o números mal leídos?"
            )
        if info["descartados"]:
            advertencias.append(
                f"Eje {nombre}: se descartaron como erróneos los valores {info['descartados']}."
            )

    log_x, log_y = info_x.get("escala") == "log", info_y.get("escala") == "log"

    def pixel_a_x(x_px):
        if m_x is None:
            return None
        v = float(m_x * x_px + b_x)
        return 10.0 ** v if log_x else v

    def pixel_a_y(y_px):
        if m_y is None:
            return None
        v = float(m_y * y_px + b_y)
        return 10.0 ** v if log_y else v

    series_salida = []
    for i, s in enumerate(series, start=1):
        if "puntos_px" in s:
            puntos_px = densificar_polilinea(s["puntos_px"], n_puntos)
        else:
            pl = mascara_a_polilinea(s["mascara"], n_puntos=n_puntos)
            if pl is None:
                continue
            _, _, cols_rs, filas_rs = pl
            puntos_px = list(zip(cols_rs, filas_rs))
            if simplificar:
                tol_px = tolerancia_simplificacion(imagen.shape, rect)
                puntos_px = [tuple(q) for q in simplificar_polilinea(puntos_px, tol_px)]
        puntos = [{
            "px": float(c), "py": float(f),
            "valor_x": pixel_a_x(c), "valor_y": pixel_a_y(f),
        } for c, f in puntos_px]
        series_salida.append({
            "id": f"serie_{i}", "nombre": s.get("nombre") or "",
            "hue": s.get("hue"), "modo": s.get("modo", "manual"),
            "n_puntos": len(puntos), "puntos": puntos,
        })

    puntos_curva = series_salida[0]["puntos"] if series_salida else []

    overlay = imagen.copy()
    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (200, 200, 200), 1)
    colores = [(0, 255, 0), (0, 200, 255), (255, 200, 0)]
    for i, s in enumerate(series):
        color = colores[i % len(colores)]
        if "mascara" in s:
            overlay[s["mascara"]] = color
        elif "puntos_px" in s and len(s["puntos_px"]) >= 2:
            pts = np.array(s["puntos_px"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], False, color, 2)
    if eje_x:
        cv2.line(overlay, eje_x[:2], eje_x[2:], (0, 0, 255), 2)      # rojo
    if eje_y:
        cv2.line(overlay, eje_y[:2], eje_y[2:], (255, 0, 0), 2)      # azul
    for e in etiquetas_x + etiquetas_y:
        cv2.rectangle(overlay, (e["px"], e["py"]),
                      (e["px"] + e["pw"], e["py"] + e["ph"]), (0, 255, 255), 1)
    for e in etiquetas_dato:
        cv2.rectangle(overlay, (e["px"], e["py"]),
                      (e["px"] + e["pw"], e["py"] + e["ph"]), (128, 0, 128), 1)
    alto_letra = max(12, int(round(0.05 * alto_imagen)))
    for bbox, color, texto in ((txt["titulo_bbox"], (255, 255, 0), txt["titulo"]),
                               (txt["titulo_x_bbox"], (0, 165, 255), txt["titulo_x"]),
                               (titulo_eje_y_bbox, (255, 0, 255), titulo_eje_y)):
        if not bbox:
            continue
        if bbox.get("manual") and texto:
            # no está en la imagen original: se escribe para poder verificarlo
            centro = (bbox["px"] + bbox["pw"] / 2, bbox["py"] + bbox["ph"] / 2)
            _dibujar_texto(overlay, texto, centro, color, alto_letra)
        else:
            cv2.rectangle(overlay, (bbox["px"], bbox["py"]),
                          (bbox["px"] + bbox["pw"], bbox["py"] + bbox["ph"]), color, 1)

    for e in txt.get("leyenda_entradas", []):
        for bbox, color in ((e.get("bbox"), (0, 128, 255)), (e.get("muestra_bbox"), (0, 255, 0))):
            if bbox:
                cv2.rectangle(overlay, (bbox["px"] - 1, bbox["py"] - 1),
                              (bbox["px"] + bbox["pw"], bbox["py"] + bbox["ph"]), color, 1)

    nombre_overlay = f"overlay_{uid}.png"
    ruta_overlay = os.path.join(dir_resultados, nombre_overlay)
    cv2.imwrite(ruta_overlay, overlay)

    resumen = {
        "eje_x_detectado": eje_x is not None,
        "eje_y_detectado": eje_y is not None,
        "curva_detectada": len(series_salida) > 0,
        "rect_grafico": list(rect),
        "n_series": len(series_salida),
        "n_etiquetas_x": len(etiquetas_x),
        "n_etiquetas_y": len(etiquetas_y),
        "n_etiquetas_dato": len(etiquetas_dato),
        "valores_etiquetas_x": [e["valor"] for e in etiquetas_x],
        "valores_etiquetas_y": [e["valor"] for e in etiquetas_y],
        "titulo": txt["titulo"],
        "titulo_eje_x": txt["titulo_x"],
        "titulo_eje_y": titulo_eje_y,
        "leyenda": txt["leyenda"],
        "n_puntos_curva": len(puntos_curva),
        "calibrado_x": m_x is not None,
        "calibrado_y": m_y is not None,
        "calibrado_y_por_datos": calibrado_y_por_datos,
        "r2_x": info_x["r2"],
        "r2_y": info_y["r2"],
        "ancho_imagen": ancho_imagen,
        "alto_imagen": alto_imagen,
        "version_segmentador": __version__,
        "listo_para_stl": bool(
          series_salida and m_x is not None and m_y is not None
        ),
    }

    nombre_json = f"grafica_{uid}.json"
    ruta_json = os.path.join(dir_resultados, nombre_json)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump({
            "imagen": {"ancho": ancho_imagen, "alto": alto_imagen,
                       "archivo": os.path.basename(ruta_imagen)},
            "ejes": {
                "eje_x": list(eje_x) if eje_x else None,
                "eje_y": list(eje_y) if eje_y else None,
                "rect_grafico": list(rect),
                "calibracion_x": {"m": m_x, "b": b_x, **info_x},
                "calibracion_y": {"m": m_y, "b": b_y, **info_y},
            },
            "textos": {
                "titulo": txt["titulo"],
                "titulo_eje_x": txt["titulo_x"],
                "titulo_eje_y": titulo_eje_y,
                "titulo_bbox": _bbox_json(txt.get("titulo_bbox")),
                "titulo_eje_x_bbox": _bbox_json(txt.get("titulo_x_bbox")),
                "titulo_eje_y_bbox": _bbox_json(titulo_eje_y_bbox),
                "leyenda": txt["leyenda"],
                "leyenda_entradas": [
                    {"texto": e["texto"], "hue": e["hue"], "metodo": e.get("metodo"),
                     "conf": e.get("conf"), "bbox": e.get("bbox"),
                     "muestra_bbox": e.get("muestra_bbox")}
                    for e in txt.get("leyenda_entradas", [])
                ],
                "etiquetas_eje_x": [{"valor": e["valor"], "px": e["centro_x"]} for e in etiquetas_x],
                "etiquetas_eje_y": [{"valor": e["valor"], "py": e["centro_y"]} for e in etiquetas_y],
                "etiquetas_dato": [{"valor": e["valor"], "px": e["centro_x"],
                                    "py": e["centro_y"]} for e in etiquetas_dato],
                # nombres de las categorías del eje X (el valor i del eje es
                # la categoría i); [] si el eje X es numérico
                "categorias_x": [c["texto"] if isinstance(c, dict) else c
                                 for c in txt.get("categorias_x") or []],
            },
            "series": series_salida,
            "resumen": resumen,
            "advertencias": advertencias,
            "_ruta_imagen_original": os.path.abspath(ruta_imagen),
        }, f, ensure_ascii=False, indent=2)

    nombre_csv = f"descripcion_{uid}.csv"
    ruta_csv = os.path.join(dir_resultados, nombre_csv)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "elemento", "x1", "y1", "x2", "y2",
                    "ancho_px", "alto_px", "valor_x", "valor_y", "texto", "descripcion"])

        partes = [f"título '{txt['titulo']}'" if txt["titulo"] else "sin título detectado",
                  "eje X detectado" if eje_x else "eje X no detectado",
                  "eje Y detectado" if eje_y else "eje Y no detectado",
                  f"{len(series_salida)} serie(s) de datos"]
        w.writerow(["resumen", "grafica", "", "", "", "", ancho_imagen, alto_imagen,
                    "", "", txt["titulo"],
                    f"Gráfica de {ancho_imagen}x{alto_imagen} px con " + ", ".join(partes) + "."])

        for adv in advertencias:
            w.writerow(["advertencia", "revisar", "", "", "", "", "", "", "", "", "", adv])

        if eje_x:
            w.writerow(["eje", "eje_x", eje_x[0], eje_x[1], eje_x[2], eje_x[3], "", "", "", "", "",
                        f"Eje X (horizontal), de ({eje_x[0]},{eje_x[1]}) a ({eje_x[2]},{eje_x[3]})."])
        if eje_y:
            w.writerow(["eje", "eje_y", eje_y[0], eje_y[1], eje_y[2], eje_y[3], "", "", "", "", "",
                        f"Eje Y (vertical), de ({eje_y[0]},{eje_y[1]}) a ({eje_y[2]},{eje_y[3]})."])

        for clave, etiqueta, texto, bbox in (
            ("titulo", "Título", txt["titulo"], txt["titulo_bbox"]),
            ("titulo_eje_x", "Título del eje X", txt["titulo_x"], txt["titulo_x_bbox"]),
            ("titulo_eje_y", "Título del eje Y (texto vertical)", titulo_eje_y, titulo_eje_y_bbox),
        ):
            if bbox:
                origen = "escrito en la pizarra" if bbox.get("manual") else "leído por OCR"
                desc = (f"{etiqueta} {origen}: '{texto}', "
                        f"recuadro {bbox['pw']}x{bbox['ph']} px.")
                w.writerow(["texto", clave, bbox["px"], bbox["py"], "", "",
                            bbox["pw"], bbox["ph"], "", "", texto, desc])
            elif texto:
                w.writerow(["texto", clave, "", "", "", "", "", "", "", "", texto,
                            f"{etiqueta} leído por OCR: '{texto}'."])
            else:
                w.writerow(["texto", clave, "", "", "", "", "", "", "", "", "",
                            f"No se detectó {etiqueta.lower()}."])

        entradas_ley = txt.get("leyenda_entradas", [])
        for e in entradas_ley:
            b = e["bbox"]
            serie_id = next((s["id"] for s in series_salida
                             if e["hue"] is not None and s["hue"] == e["hue"]), "")
            desc = f"Entrada de leyenda: '{e['texto']}'"
            desc += f" (nombre de {serie_id})." if serie_id else "."
            if e.get("metodo") == "plantilla":
                desc += " Símbolo reconocido por comparación de formas: verificar."
            w.writerow(["texto", "leyenda", b["px"], b["py"], "", "", b["pw"], b["ph"],
                        "", "", e["texto"], desc])
        if txt["leyenda"] and not entradas_ley:
            w.writerow(["texto", "leyenda", "", "", "", "", "", "", "", "", txt["leyenda"],
                        f"Texto dentro del área de dibujo (leyenda o anotación): '{txt['leyenda']}'."])

        for e in etiquetas_dato:
            w.writerow(["texto", "etiqueta_dato", e["px"], e["py"], "", "", e["pw"], e["ph"],
                        "", e["valor"], "",
                        f"Etiqueta de dato (número pegado a la curva) con valor {e['valor']}."
                        + (" Usada para calibrar el eje Y." if calibrado_y_por_datos else "")])
        for e in etiquetas_y:
            w.writerow(["texto", "etiqueta_eje_y", e["px"], e["py"], "", "", e["pw"], e["ph"],
                        "", e["valor"], "", f"Etiqueta numérica del eje Y con valor {e['valor']}."])
        for e in etiquetas_x:
            w.writerow(["texto", "etiqueta_eje_x", e["px"], e["py"], "", "", e["pw"], e["ph"],
                        e["valor"], "", "", f"Etiqueta numérica del eje X con valor {e['valor']}."])

        for s in series_salida:
            for p in s["puntos"]:
                if p["valor_x"] is not None and p["valor_y"] is not None:
                    desc = (f"Punto de {s['id']}: valor real "
                            f"(x={p['valor_x']:.3f}, y={p['valor_y']:.3f}), "
                            f"píxel ({p['px']:.1f}, {p['py']:.1f}).")
                else:
                    desc = (f"Punto de {s['id']} en píxel ({p['px']:.1f}, {p['py']:.1f}); "
                            "sin calibración suficiente para valor real.")
                w.writerow(["curva", s["id"], f"{p['px']:.1f}", f"{p['py']:.1f}", "", "", "", "",
                            "" if p["valor_x"] is None else f"{p['valor_x']:.3f}",
                            "" if p["valor_y"] is None else f"{p['valor_y']:.3f}",
                            "", desc])

    return {
        "ruta_csv": ruta_csv, "nombre_csv": nombre_csv,
        "ruta_json": ruta_json, "nombre_json": nombre_json,
        "ruta_overlay": ruta_overlay, "nombre_overlay": nombre_overlay,
        "series": series_salida,
        "puntos_curva": puntos_curva,
        "advertencias": advertencias,
        "resumen": resumen,
    }


def aplicar_correcciones(ruta_imagen, dir_resultados, correcciones,
                          ruta_json_original=None, n_puntos=300):
    """Recalcula con las correcciones de la pizarra (lo que falte se toma del JSON original).

    Claves opcionales de `correcciones`: eje_x, eje_y, etiquetas_x, etiquetas_y,
    series [{"puntos_px": [[x, y], ...]}], titulo_eje_x(_pos), titulo_eje_y(_pos), leyenda.
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError(f"No se pudo abrir la imagen: {ruta_imagen}")

    base = {}
    if ruta_json_original and os.path.exists(ruta_json_original):
        with open(ruta_json_original, encoding="utf-8") as f:
            base = json.load(f)
    ejes_base = base.get("ejes", {})
    textos_base = base.get("textos", {})

    def _eje(clave):
        if clave in correcciones:
            v = correcciones[clave]
            return tuple(int(round(x)) for x in v) if v else None
        v = ejes_base.get(clave)
        return tuple(v) if v else None

    eje_x, eje_y = _eje("eje_x"), _eje("eje_y")
    rect = area_de_dibujo(eje_x, eje_y, imagen.shape)

    def _normalizar_etiqueta_x(e):
        cx = e.get("centro_x", e.get("px", 0))
        return {"centro_x": float(cx), "valor": float(e["valor"]),
                "px": int(round(cx)), "py": int(e.get("py", 0)),
                "pw": int(e.get("pw", 0)), "ph": int(e.get("ph", 0))}

    def _normalizar_etiqueta_y(e):
        cy = e.get("centro_y", e.get("py", 0))
        return {"centro_y": float(cy), "valor": float(e["valor"]),
                "px": int(e.get("px", 0)), "py": int(round(cy)),
                "pw": int(e.get("pw", 0)), "ph": int(e.get("ph", 0))}

    etiquetas_x = correcciones.get("etiquetas_x")
    if etiquetas_x is None:
        etiquetas_x = [{"centro_x": e["px"], "valor": e["valor"]}
                       for e in textos_base.get("etiquetas_eje_x", [])]
    etiquetas_x = [_normalizar_etiqueta_x(e) for e in etiquetas_x]

    etiquetas_y = correcciones.get("etiquetas_y")
    if etiquetas_y is None:
        etiquetas_y = [{"centro_y": e["py"], "valor": e["valor"]}
                       for e in textos_base.get("etiquetas_eje_y", [])]
    etiquetas_y = [_normalizar_etiqueta_y(e) for e in etiquetas_y]

    etiquetas_dato = [
        {"valor": e["valor"], "px": int(e["px"]), "py": int(e["py"]), "pw": 0, "ph": 0}
        for e in textos_base.get("etiquetas_dato", [])
    ]

    series_base = base.get("series", [])
    if "series" in correcciones:
        # La pizarra no siempre manda el nombre de cada serie: se conserva el
        # que tenía la serie en la misma posición.
        series = [{"puntos_px": [(float(p[0]), float(p[1])) for p in s["puntos_px"]],
                   "hue": s.get("hue"), "modo": "manual",
                   "nombre": s.get("nombre") or (series_base[k].get("nombre", "")
                                                 if k < len(series_base) else "")}
                  for k, s in enumerate(correcciones["series"])]
    else:
        series = [{"puntos_px": [(p["px"], p["py"]) for p in s["puntos"]],
                   "hue": s.get("hue"), "modo": s.get("modo"), "nombre": s.get("nombre", "")}
                  for s in base.get("series", [])]

    txt = {
        "titulo": textos_base.get("titulo", ""),
        "titulo_bbox": textos_base.get("titulo_bbox"),
        "titulo_x": textos_base.get("titulo_eje_x", ""),
        "titulo_x_bbox": textos_base.get("titulo_eje_x_bbox"),
        "leyenda": textos_base.get("leyenda", ""),
        "leyenda_entradas": textos_base.get("leyenda_entradas", []),
        "categorias_x": textos_base.get("categorias_x") or [],
    }
    titulo_eje_y = textos_base.get("titulo_eje_y", "")
    titulo_eje_y_bbox = textos_base.get("titulo_eje_y_bbox")
    if "titulo_eje_x" in correcciones:
        txt["titulo_x"] = (correcciones["titulo_eje_x"] or "").strip()
        if not txt["titulo_x"]:
            txt["titulo_x_bbox"] = None
        elif correcciones.get("titulo_eje_x_pos"):
            txt["titulo_x_bbox"] = _bbox_desde_posicion(correcciones["titulo_eje_x_pos"], imagen.shape)
    if "titulo_eje_y" in correcciones:
        titulo_eje_y = (correcciones["titulo_eje_y"] or "").strip()
        if not titulo_eje_y:
            titulo_eje_y_bbox = None
        elif correcciones.get("titulo_eje_y_pos"):
            titulo_eje_y_bbox = _bbox_desde_posicion(correcciones["titulo_eje_y_pos"], imagen.shape)
    if "leyenda" in correcciones:
        txt["leyenda"] = correcciones["leyenda"] or ""
        txt["leyenda_entradas"] = []

    advertencias = ["Resultado corregido manualmente por el usuario en la pizarra."]
    uid = uuid.uuid4().hex[:8]

    return _calibrar_y_exportar(
        imagen=imagen, ruta_imagen=ruta_imagen, dir_resultados=dir_resultados, uid=uid,
        eje_x=eje_x, eje_y=eje_y, rect=rect, series=series, txt=txt,
        etiquetas_x=etiquetas_x, etiquetas_y=etiquetas_y, etiquetas_dato=etiquetas_dato,
        titulo_eje_y=titulo_eje_y, titulo_eje_y_bbox=titulo_eje_y_bbox,
        advertencias=advertencias, n_puntos=n_puntos,
    )


def procesar_imagen(ruta_imagen, dir_resultados, n_puntos=300, max_series=3, lang=None,
                    simplificar=False, palabras=None):
    """Pipeline completo. Devuelve rutas (csv, json, overlay), resumen, advertencias y series.

    `palabras`: texto real del PDF dentro de la figura, en píxeles de esta
    imagen (pipeline_rapido.extraer_palabras). Si trae al menos 2 números,
    se usa en lugar del OCR para ejes, títulos y leyenda: es exacto y no
    necesita Tesseract. Sin eso (imagen escaneada o pegada) se usa el OCR,
    con la detección de leyenda y de símbolos (Ω, μ...) por plantillas.
    """
    os.makedirs(dir_resultados, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    advertencias = []

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError(f"No se pudo abrir la imagen: {ruta_imagen}")

    alto_imagen, ancho_imagen = imagen.shape[:2]
    t = _tolerancias(imagen.shape)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    eje_x, eje_y = detectar_ejes(gris, tol=t)
    if eje_x is None:
        advertencias.append("No se detectó el eje X: no habrá calibración horizontal.")
    if eje_y is None:
        advertencias.append("No se detectó el eje Y: no habrá calibración vertical.")
    eje_x_fila = float((eje_x[1] + eje_x[3]) / 2) if eje_x else None
    eje_y_col = float((eje_y[0] + eje_y[2]) / 2) if eje_y else None
    rect = area_de_dibujo(eje_x, eje_y, imagen.shape)

    series = detectar_curvas(imagen, rect=rect, area_minima=t["area_min_curva"],
                             max_series=max_series)
    if not series:
        advertencias.append("No se detectó ninguna curva dentro del área de dibujo.")
    elif series[0]["modo"] == "intensidad":
        advertencias.append(
            "Curva detectada por intensidad (gráfica sin color): verificar el overlay, "
            "puede incluir rejilla o marcadores."
        )
    if len(series) > 1:
        advertencias.append(
            f"Se detectaron {len(series)} series; cada una necesita su propia textura BANA."
        )
    mascaras = [s["mascara"] for s in series]

    # --- ¿Texto real del PDF en vez de OCR? ---
    tokens_pdf = tokens_de_palabras_pdf(palabras)
    usa_pdf = sum(tk["valor"] is not None for tk in tokens_pdf) >= 2
    if not usa_pdf:
        tokens_pdf = None
        if not ocr_disponible():
            # Sin Tesseract ni texto en el PDF: se sigue sin texto (ejes y
            # curva igual se detectan) en vez de abortar toda la segmentación.
            usa_pdf = True
            tokens_pdf = tokens_de_palabras_pdf(palabras)
            advertencias.append(
                "No hay OCR (Tesseract) en el servidor y la imagen no trae texto: "
                "no se leyeron números ni títulos. Revisar o corregir en la pizarra."
            )

    # --- Números de los ejes: por regiones (OCR) o del PDF ---
    if usa_pdf:
        reg_x, reg_y = [], []
    else:
        reg_x, reg_y = detectar_etiquetas_ejes(gris, eje_x, eje_y, rect=rect, tol=t)
    usa_reg_x, usa_reg_y = len(reg_x) >= 2, len(reg_y) >= 2
    excluir = (reg_x if usa_reg_x else []) + (reg_y if usa_reg_y else [])

    txt = detectar_textos(gris, eje_x_fila, eje_y_col, rect=rect,
                          mascaras_curva=mascaras, tol=t, lang=lang,
                          excluir=excluir, tokens_pdf=tokens_pdf)
    etiquetas_x = reg_x if usa_reg_x else max(reg_x, txt["etiquetas_x"], key=len)
    etiquetas_y = reg_y if usa_reg_y else max(reg_y, txt["etiquetas_y"], key=len)
    etiquetas_dato = txt["etiquetas_dato"]

    # --- Multiplicador del eje ("1e7" arriba del eje Y, "×1e6" al final del X)
    # y segundo eje Y a la derecha ---
    todos = txt.get("tokens") or []
    if rect is not None:
        rx0, ry0, rx1, ry1 = rect
        ancho_r = rx1 - rx0
        patron_mult = re.compile(r"(?:[x×]\s*)?1e([+-]?\d+)")

        def _con_potencias(etqs, salvo):
            # un eje logarítmico escribe sus propios números como 1e0, 1e1...:
            # ahí un "1eN" es una marca más, no un multiplicador
            return any(patron_mult.fullmatch(str(e.get("texto", "")).replace("−", "-"))
                       for e in etqs if e is not salvo)

        for tk in todos:
            m_mult = patron_mult.fullmatch(tk["texto"].replace("−", "-"))
            if not m_mult:
                continue
            factor = 10.0 ** int(m_mult.group(1))
            if (tk["py"] + tk["ph"] <= ry0 + 0.25 * tk["ph"] and tk["centro_x"] < rx0 + 0.35 * ancho_r
                    and not _con_potencias(etiquetas_y, tk)):
                etiquetas_y = [{**e, "valor": e["valor"] * factor} for e in etiquetas_y if e is not tk]
                advertencias.append(f"Eje Y multiplicado por 10^{m_mult.group(1)} (indicado sobre el eje).")
            elif (eje_x_fila is not None and tk["centro_y"] > eje_x_fila
                  and tk["centro_x"] > rx1 - 0.35 * ancho_r
                  and not _con_potencias(etiquetas_x, tk)):
                etiquetas_x = [{**e, "valor": e["valor"] * factor} for e in etiquetas_x if e is not tk]
                advertencias.append(f"Eje X multiplicado por 10^{m_mult.group(1)} (indicado al final del eje).")
            etiquetas_dato = [e for e in etiquetas_dato if e is not tk]

        derecha = [tk for tk in todos if tk["valor"] is not None
                   and tk["centro_x"] > rx1 + t["margen_texto"] and ry0 <= tk["centro_y"] <= ry1]
        if len(derecha) >= 2:
            advertencias.append(
                "El gráfico tiene un segundo eje vertical a la derecha: solo se usó la escala "
                "de la izquierda, así que las curvas que se leen con el eje derecho quedan "
                "con valores incorrectos. Revisar antes de imprimir."
            )

    # --- Eje X de categorías ("Ene", "Feb"...): cada rótulo es la posición
    # 0, 1, 2...; la lámina y la narración usan el nombre ---
    if len(etiquetas_x) < 2 and txt.get("categorias_x"):
        etiquetas_x = [{"centro_x": c["centro_x"], "valor": float(i), "px": int(c["px"]),
                        "py": int(c["py"]), "pw": int(c["pw"]), "ph": int(c["ph"])}
                       for i, c in enumerate(txt["categorias_x"])]

    # --- Título del eje Y (vertical) ---
    x_limite_vertical = None
    if etiquetas_y:
        x_limite_vertical = min(e["px"] for e in etiquetas_y)
    elif eje_y is not None:
        x_limite_vertical = _limite_titulo_vertical(
            gris, eje_y, eje_x, int(np.median(gris)), t["margen_texto"])
    titulo_eje_y, titulo_eje_y_bbox = ("", None)
    if x_limite_vertical is not None:
        if usa_pdf:
            titulo_eje_y, titulo_eje_y_bbox = titulo_vertical_de_palabras(
                palabras, x_limite_vertical)
        else:
            titulo_eje_y, titulo_eje_y_bbox = detectar_texto_vertical(
                gris, x_limite_vertical, lang=lang)

    # --- Leyenda: nombre de cada serie ---
    if usa_pdf:
        # Texto exacto del PDF: cada nombre se empareja por color con la
        # muestra que tiene al lado (y las series quedan en el orden de la
        # leyenda).
        txt["leyenda_entradas"] = []
        if len(series) > 1:
            nombres, usados = asignar_nombres_de_leyenda(imagen, series, todos)
            # un número de un nombre ("Ventas 2023") no es un valor anotado
            etiquetas_dato = [e for e in etiquetas_dato if not any(e is u for u in usados)]
            if not all(nombres):
                advertencias.append(
                    "No se pudo leer en la leyenda el nombre de todas las series: las que "
                    "faltan se llaman 'Serie A', 'Serie B'... en la lámina y la narración."
                )
    else:
        # OCR: detección de leyenda con muestras de color y reconocimiento
        # de símbolos (Ω, μ, λ...) por plantillas.
        if not _cargar_plantillas():
            advertencias.append(
                "No hay plantillas de símbolos (falta Pillow o no se encontraron fuentes): "
                "los símbolos griegos como 'Ω' no se pueden reconocer. "
                "Instala Pillow (pip install pillow) y ejecuta segmentador.diagnostico()."
            )
        leyenda_entradas = detectar_leyenda(imagen, gris, rect, series, tol=t, lang=lang,
                                            advertencias=advertencias)
        cajas_leyenda = []
        for e in leyenda_entradas:
            for b in (e["bbox"], e["muestra_bbox"]):
                if b:
                    cajas_leyenda.append((b["px"], b["py"], b["pw"], b["ph"]))
        if not leyenda_entradas and not _texto_plausible(txt["leyenda"]):
            leyenda_entradas = detectar_leyenda_sin_muestra(
                gris, rect, tol=t, lang=lang,
                excluir_cajas=[(e["px"], e["py"], e["pw"], e["ph"]) for e in etiquetas_y])
        if leyenda_entradas:
            txt["leyenda"] = " | ".join(e["texto"] for e in leyenda_entradas)
            for e in leyenda_entradas:
                for serie in series:
                    if e["hue"] is not None and serie.get("hue") == e["hue"]:
                        serie["nombre"] = e["texto"]
            sin_verif = [e["texto"] for e in leyenda_entradas if e["metodo"] == "ocr_sin_verificar"]
            if sin_verif:
                advertencias.append(
                    "Leyenda: " + ", ".join(f"'{z}'" for z in sin_verif) + " se leyó solo con OCR, "
                    "sin comparar formas; si era un símbolo (Ω, μ, λ...) probablemente está mal."
                )
        txt["leyenda_entradas"] = leyenda_entradas

        if not txt["titulo_x"] and not txt.get("categorias_x"):
            tx, tx_bbox = detectar_titulo_eje_x_por_region(
                gris, eje_x, etiquetas_x, rect, tol=t, lang=lang, excluir_cajas=cajas_leyenda)
            if tx:
                txt["titulo_x"], txt["titulo_x_bbox"] = tx, tx_bbox

    for nombre, etqs, clave in (("X", etiquetas_x, lambda e: e["centro_x"]),
                                ("Y", etiquetas_y, lambda e: e["centro_y"])):
        for antes, despues in rescatar_etiquetas(etqs, clave):
            advertencias.append(
                f"Eje {nombre}: la etiqueta leída como '{antes}' se corrigió a '{despues}' "
                "para que sea coherente con el resto de la escala; verifícalo en el overlay."
            )

    return _calibrar_y_exportar(
        imagen=imagen, ruta_imagen=ruta_imagen, dir_resultados=dir_resultados, uid=uid,
        eje_x=eje_x, eje_y=eje_y, rect=rect, series=series, txt=txt,
        etiquetas_x=etiquetas_x, etiquetas_y=etiquetas_y, etiquetas_dato=etiquetas_dato,
        titulo_eje_y=titulo_eje_y, titulo_eje_y_bbox=titulo_eje_y_bbox,
        advertencias=advertencias, n_puntos=n_puntos, simplificar=simplificar,
    )


# ---------- 7) DIAGNÓSTICO ----------

def diagnostico(ruta_imagen=None, lang=None):
    """Muestra versión, Tesseract, Pillow, fuentes y qué detecta en la imagen."""
    print("segmentador.py :", os.path.abspath(__file__))
    print("versión        :", __version__)
    try:
        print("Tesseract      :", pytesseract.get_tesseract_version())
        print("idiomas        :", pytesseract.get_languages(config=""))
    except Exception as e:
        print("Tesseract      : ERROR ->", e)
    try:
        import PIL
        print("Pillow         :", PIL.__version__)
    except ImportError:
        print("Pillow         : NO INSTALADO  -> pip install pillow")
    extra = [r for r in os.environ.get("SEGMENTADOR_FUENTES", "").split(";") if r.strip()]
    fuentes = [r for r in extra + _FUENTES_CANDIDATAS if os.path.isfile(r)]
    print("fuentes        :", fuentes or "NINGUNA (define SEGMENTADOR_FUENTES)")
    plantillas = _cargar_plantillas()
    print("plantillas     :", len(plantillas),
          "| con Ω:", any(ch == "Ω" for ch, _ in plantillas))
    if not ruta_imagen:
        return

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print("No se pudo abrir", ruta_imagen)
        return
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    t = _tolerancias(imagen.shape)
    eje_x, eje_y = detectar_ejes(gris, tol=t)
    rect = area_de_dibujo(eje_x, eje_y, imagen.shape)
    series = detectar_curvas(imagen, rect=rect, area_minima=t["area_min_curva"])
    print("\nimagen         :", imagen.shape[1], "x", imagen.shape[0])
    print("ejes           :", eje_x, eje_y, "| rect:", rect)
    print("series (hue)   :", [s_.get("hue") for s_ in series])

    reg_x, _ = detectar_etiquetas_ejes(gris, eje_x, eje_y, rect=rect, tol=t)
    print("números eje X  :", [e["valor"] for e in reg_x])
    tx, tb = detectar_titulo_eje_x_por_region(gris, eje_x, reg_x, rect, tol=t, lang=lang)
    print("título eje X   :", repr(tx), tb)

    avisos = []
    ley = detectar_leyenda(imagen, gris, rect, series, tol=t, lang=lang, advertencias=avisos)
    for e in ley:
        print("leyenda        :", repr(e["texto"]), "| método:", e["metodo"],
              "| conf:", round(e["conf"], 1), "| muestra:", e["muestra_bbox"],
              "| texto:", e["bbox"])
        b = e["bbox"]
        x, y, w, h = b["px"], b["py"], b["pw"], b["ph"]
        crop = gris[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
        tinta, binaria = _tinta_de_recorte(crop)
        if tinta is not None:
            gl = _glifos(binaria)
            if gl:
                gx, gy, gw, gh = max(gl, key=lambda c: c[2] * c[3])
                print("   ranking     :", [(c, round(p, 3)) for c, p in
                                           _clasificar_glifo(tinta[gy:gy + gh, gx:gx + gw])[:5]])
    if not ley:
        print("leyenda        : no encontrada")
    for a in avisos:
        print("aviso          :", a)
