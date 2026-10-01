"""
segmentador.py 
  1) Detectar los ejes X e Y de una gráfica y el rectángulo del área de dibujo.
  2) Detectar UNA O VARIAS curvas de datos por color dominante.
  3) Detectar y clasificar el texto (título, título de ejes, números de eje,
     etiquetas de dato, leyenda) con Tesseract OCR.
  4) Calibrar píxel -> valor real con ajuste lineal ROBUSTO (descarta números
     mal leídos por el OCR).
  5) Exportar un CSV (compatible con la versión anterior) y un JSON
     estructurado, pensado como entrada directa de la etapa BANA/STL.

Salidas de procesar_imagen():
    ruta_csv, ruta_json, ruta_overlay, resumen, advertencias, series
"""

import os
import re
import csv
import json
import math
import uuid

import cv2
import numpy as np

# Cambia con cada entrega: sale en el JSON (resumen.version_segmentador) para
# confirmar que el servidor está usando ESTE archivo y no uno viejo en memoria.
__version__ = "2026-10-01-simbolos-v2"
import pytesseract
from pytesseract import Output

# --- Configuración de Tesseract OCR en Windows ---
_RUTA_TESSERACT_WINDOWS = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
if os.name == "nt" and os.path.isfile(_RUTA_TESSERACT_WINDOWS):
    pytesseract.pytesseract.tesseract_cmd = _RUTA_TESSERACT_WINDOWS


# ============================================================
# 0) UTILIDADES GENERALES
# ============================================================

def _tolerancias(forma):
    """
    Todas las tolerancias se derivan del tamaño de la imagen
    """
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


def _idioma_ocr(preferido=None):
    """
    Aquí usamos spa+eng 
    """
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


def _es_numero(texto):
    return bool(_RE_NUMERO.match(texto.strip()))


def _a_float(texto):
    """
    Convierte a float respetando notación española e inglesa. Devuelve None si
    no se puede (antes esto podía lanzar ValueError y tumbar el pipeline).
    """
    t = texto.strip().replace("−", "-").replace("–", "-").replace("—", "-")
    t = t.replace(" ", "")
    tiene_punto, tiene_coma = "." in t, "," in t
    if tiene_punto and tiene_coma:
        # el separador decimal es el que aparece más a la derecha
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif tiene_coma:
        entero, _, dec = t.partition(",")
        # "1,000" con 3 dígitos después -> miles; "3,5" -> decimal
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


# ============================================================
# 1) DETECCIÓN DE EJES Y DEL ÁREA DE DIBUJO
# ============================================================

def _fusionar_lineas(lineas, orientacion, tol):
    """
    Canny devuelve DOS bordes por cada trazo grueso, así que Hough entrega el
    mismo eje duplicado (y en trozos). Aquí se agrupan las líneas casi
    colineales y se unen sus extremos.
    """
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
    """
    Devuelve (eje_x, eje_y) como tuplas (x1,y1,x2,y2), o (None, None).

    Correcciones respecto a la versión anterior:
      - umbrales relativos al tamaño de la imagen;
      - clasificación horizontal/vertical con elif (antes una misma línea podía
        entrar en las dos listas);
      - fusión de líneas duplicadas por Canny;
      - se descarta el marco de la imagen (líneas pegadas al borde), que antes
        podía ganarle al eje X real por estar "más abajo";
      - el eje Y se elige por cercanía a la esquina inferior izquierda del eje
        X, considerando también que su extremo inferior toque esa fila.
    """
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

    # Un "marco" (borde del recorte de la figura, típico al extraer imágenes de
    # un PDF con algo de relleno) es MUY largo y toca las dos puntas de su
    # propio eje (columna/fila 0 y la última). Un eje real, aunque sea largo,
    # casi nunca toca el borde exacto: siempre queda margen para las etiquetas
    # y el título. El filtro anterior (por posición) no distinguía esto, así
    # que un marco le podía ganar al eje real en `_elegir_esquina` por ser más
    # largo, dejando los ejes detectados pegados al borde de toda la imagen en
    # vez de sobre las líneas reales del gráfico.
    margen_marco = max(3, int(round(0.01 * min(alto, ancho))))

    def _es_marco(g, dim):
        return g["a"] <= margen_marco and g["b"] >= dim - 1 - margen_marco

    gh_sin_marco = [g for g in gh if not _es_marco(g, ancho)]
    gv_sin_marco = [g for g in gv if not _es_marco(g, alto)]

    eje_x, eje_y = _elegir_esquina(gh_sin_marco, gv_sin_marco, t, alto, ancho)
    if eje_x is None and eje_y is None:
        # Ninguna L sin marco: se reintenta con las líneas originales (mejor un
        # eje pegado al borde que ninguno, por si la figura de verdad no tiene
        # margen alrededor del gráfico).
        eje_x, eje_y = _elegir_esquina(gh, gv, t, alto, ancho)
    if eje_x is not None or eje_y is not None:
        return eje_x, eje_y

    # ---- Respaldo (no hay ninguna L clara): heurística anterior ----
    # Igual que arriba: se prefieren las líneas sin marco si hay alguna.
    gh_resp = gh_sin_marco or gh
    gv_resp = gv_sin_marco or gv

    eje_x = None
    if gh_resp:
        # el eje X es la horizontal larga más baja del gráfico
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
    """
    Un par de ejes es una "L": el eje Y termina donde empieza el eje X.
    Antes el eje X era simplemente "la horizontal larga más baja", y el Y "la
    vertical más cercana a su extremo". Eso falla en dos casos reales:

      1) Hough alarga los segmentos a través de las marcas (ticks) y de las
         etiquetas del origen: el eje X arrancaba a la izquierda del eje Y y el
         eje Y bajaba por debajo del eje X (los ejes no se cortaban);
      2) una leyenda, un marco o una tabla debajo de la gráfica tiene una
         horizontal más baja que el eje y le ganaba.

    Aquí se prueban todos los pares (horizontal, vertical) y se elige el que
    mejor forma la esquina inferior-izquierda; luego cada eje se recorta a esa
    esquina. Si no hay ninguna L clara devuelve (None, None) y el llamador usa
    la heurística de respaldo.
    """
    if not gh or not gv:
        return None, None
    diag = math.hypot(alto, ancho)
    # cuánto pueden sobresalir los ejes por marcas/etiquetas del origen
    sobrepaso = max(t["tol_linea"] * 3, int(round(0.10 * min(alto, ancho))))

    mejor, mejor_costo = None, None
    for h in gh:
        for v in gv:
            vx, hy = v["pos"], h["pos"]
            # el eje Y debe caer cerca del EXTREMO IZQUIERDO del eje X ...
            if not (h["a"] - sobrepaso <= vx <= h["a"] + 0.25 * h["largo"]):
                continue
            # ... y el eje X cerca del EXTREMO INFERIOR del eje Y
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
    # La esquina ya fue validada (ambos extremos a menos de `sobrepaso`), así que
    # los dos ejes se hacen tocar exactamente en ella.
    return (col, fila, int(h["b"]), fila), (col, int(v["a"]), col, fila)


def area_de_dibujo(eje_x, eje_y, forma):
    """
    Rectángulo (x0, y0, x1, y1) donde vive la curva. Es clave: en la versión
    anterior la curva se buscaba en TODA la imagen, así que un logo de color,
    un recuadro de leyenda o un título coloreado podían ganar el concurso de
    "color dominante".
    """
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


# ============================================================
# 2) DETECCIÓN DE CURVAS (una o varias series)
# ============================================================

def _hues_dominantes(h, mask, n_max=3, tol=12, minimo_relativo=0.06):
    """
    Histograma circular de tonos. El `np.argmax` plano de la versión anterior
    partía en dos cualquier curva roja (tono ~0 y ~179 a la vez).
    """
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


def _limpiar_componentes(mask_u8, area_minima):
    """
    Cierra huecos (líneas punteadas, cortes por rejilla) y se queda con todos
    los trozos grandes de ese color, no solo con el mayor: antes, una curva
    cortada por las líneas de rejilla se truncaba al fragmento más largo.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cerrada = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
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
    """
    Devuelve una lista de dicts {"hue", "mascara", "modo"}. Para el objetivo
    BANA esto importa: cada serie necesita su propia textura en el STL, y la
    versión anterior solo podía devolver una.
    """
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

    # --- Respaldo: curva negra/gris (muy común en papers en blanco y negro) ---
    # Aquí los ejes son negros igual que la curva, así que el recorte se separa
    # unos píxeles más de los bordes: con un inset de 1 px, las primeras
    # columnas se comían el trazo del propio eje Y y el primer punto de la
    # polilínea salía desplazado.
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


def detectar_curva(imagen_bgr, **kw):
    """Compatibilidad con la API anterior: devuelve solo la primera máscara."""
    s = detectar_curvas(imagen_bgr, **kw)
    return s[0]["mascara"] if s else None


# ============================================================
# 3) OCR: DETECCIÓN Y CLASIFICACIÓN DE TEXTO
# ============================================================

def _mapa_distancia_curva(mascaras):
    if not mascaras:
        return None
    union = np.zeros_like(mascaras[0], dtype=bool)
    for m in mascaras:
        union |= m
    no_curva = (~union).astype(np.uint8) * 255
    return cv2.distanceTransform(no_curva, cv2.DIST_L2, 5)


def detectar_textos(gris, eje_x_fila, eje_y_col, rect=None, mascaras_curva=None,
                    tol=None, escala_ocr=2.0, lang=None, excluir=None):
    """
    Corre OCR una sola vez y clasifica en DOS pasadas (antes era una sola
    cadena de if/elif, y eso causaba dos errores reales):

      1) el "0" del origen, que está a la izquierda del eje Y pero a la altura
         del eje X, se clasificaba como etiqueta del eje X y arruinaba la
         calibración horizontal. Ahora una etiqueta de eje X además debe estar
         dentro del rango horizontal del gráfico, y una de eje Y dentro del
         rango vertical;
      2) cualquier texto no numérico dentro del gráfico (leyenda, anotaciones,
         unidades) se pegaba al título. Ahora el título solo puede estar por
         ENCIMA del área de dibujo; lo de adentro va a "leyenda".

    Además el título se arma respetando el orden de lectura de Tesseract
    (bloque, párrafo, línea, palabra). Antes se ordenaba solo por `cy`, así que
    las palabras de una misma línea salían barajadas.
    """
    t = tol or _tolerancias(gris.shape)
    margen = t["margen_texto"]
    radio_dato = t["radio_dato"]
    mapa_dist = _mapa_distancia_curva(mascaras_curva)

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

    # ---------- pasada 1: recolectar tokens ----------
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

    # Los números que ya se leyeron por regiones (`detectar_etiquetas_ejes`) no
    # se vuelven a clasificar: evita duplicados y que sus restos acaben en la
    # leyenda o en los títulos.
    if excluir:
        def _dentro(tk):
            for b in excluir:
                if (b["px"] - 3 <= tk["centro_x"] <= b["px"] + b["pw"] + 3
                        and b["py"] - 3 <= tk["centro_y"] <= b["py"] + b["ph"] + 3):
                    return True
            return False
        tokens = [tk for tk in tokens if not _dentro(tk)]

    # ---------- pasada 2: clasificar ----------
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

        # --- texto no numérico ---
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

    # El título del eje X es la línea de texto MÁS CERCANA debajo de los números
    # del eje (antes se tomaba todo lo que hubiera debajo, y una leyenda puesta
    # bajo la gráfica se pegaba al título: "Meses — Serie A"). Lo demás es leyenda.
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
    }


# ============================================================
# 3a) ETIQUETAS NUMÉRICAS DE LOS EJES (lectura por regiones)
# ============================================================
#
# Por qué existe: el OCR de la imagen completa (`psm 11`) es bueno para títulos
# pero malo para las marcas de los ejes: son números sueltos, diminutos y en
# negrita, y Tesseract se los salta o los lee como basura ("2 4 6 8 10" daba
# solo "10"). Aquí el eje ya se conoce, así que:
#   1) se recorta la franja donde DEBEN estar las etiquetas (debajo del eje X,
#      a la izquierda del eje Y);
#   2) se separa cada etiqueta por componentes conectados (sin depender del
#      OCR para saber dónde hay texto);
#   3) cada etiqueta se agranda y se lee sola, solo con dígitos;
#   4) se conserva la fila (X) o la columna (Y) que más etiquetas alinea.

_WHITELIST_NUM = "0123456789.,-+%"
_CONFUSIONES = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "l": "1", "I": "1", "|": "1", "i": "1", "!": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9", "q": "9",
})


def _normalizar_numero(texto):
    """Devuelve el texto listo para _a_float o None si no es un número plausible."""
    t = texto.strip().strip(".,:;")
    if t.endswith("%"):
        t = t[:-1].strip()
    if not t:
        return None
    if not _es_numero(t):
        t2 = t.replace(" ", "")
        if not _es_numero(t2) and any(c.isdigit() for c in t2):
            # confusiones típicas ("1O" -> "10"), solo si ya hay al menos un dígito real
            t2 = t2.translate(_CONFUSIONES)
        t = t2
    return t if _es_numero(t) else None


def _grosor_hasta_borde(gris, eje, lado, fondo):
    """
    Devuelve la fila (eje X, lado='abajo') o la columna (eje Y, lado='izq') donde
    TERMINA el trazo del eje. Las etiquetas se buscan a partir de ahí, así el
    propio eje (que puede medir 3-5 px) no se cuela en la franja.
    """
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
    """
    Cajas (x, y, w, h) en coordenadas de la imagen con posibles etiquetas dentro
    de `region`. `lado_eje` = 'arriba' (el eje X está justo encima de la franja)
    o 'derecha' (el eje Y está justo a la derecha).
    Devuelve (cajas, altura_tipica_de_glifo).
    """
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
        # Sin filtro de área mínima: el punto decimal de un texto chico mide 1-2 px
        # y, si se descarta, "1.0" se parte en "1" y "0".
        delgado = min(c["w"], c["h"]) <= 2
        if delgado and max(c["w"], c["h"]) >= 2.5 * h_ref:
            continue                       # resto del eje o del marco
        if c["toca"]:
            # marca (tick): pegada al eje, fina y más corta que un dígito
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
    # Une los dígitos de UNA misma etiqueta. Debe ser menor que la separación
    # entre los números y el título rotado del eje Y (si no, la caja de "20" se
    # ensancha hasta la letra del título).
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
    """
    Lee UN número en un recorte pequeño. Devuelve (texto_normalizado, confianza).

    `margen_px` es el contorno de fondo que se dejó alrededor de la tinta; sirve
    para calcular la altura real del glifo y agrandarlo hasta ~42 px, que es donde
    Tesseract lee bien (con glifos de 5 px de alto no lee ni un "0").
    """
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

    # Primero solo dígitos; si Tesseract no devuelve nada, se reintenta sin la
    # lista blanca (a veces con ella lee vacío) y `_normalizar_numero` filtra.
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
        # Tesseract casi nunca reconoce un dígito AISLADO (un "0" suelto sale
        # vacío). Se repite el recorte lado a lado ("0 0"), se lee esa línea y,
        # si sale un patrón duplicado, se toma una mitad.
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
            # Umbral estricto: una lectura falsa con confianza baja es peor que
            # no leer nada (en una gráfica táctil un número falso pasa inadvertido).
            if n >= 2 and n % 2 == 0 and crudo[:n // 2] == crudo[n // 2:] and conf_media >= 50:
                num = _normalizar_numero(crudo[:n // 2])
                if num is not None:
                    return num, conf_media
    return mejor, max(mejor_conf, 0.0)


def _elegir_alineacion(tokens, clave, tol, distancia_eje):
    """
    Entre todos los números leídos, se queda con los que están ALINEADOS: misma
    fila (etiquetas del eje X) o mismo borde derecho (etiquetas del eje Y). Así
    un número suelto de una leyenda o un título rotado mal leído no entra.
    """
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
    """
    Lee las marcas numéricas de los dos ejes por regiones.
    Devuelve (etiquetas_x, etiquetas_y): listas de dicts con el mismo formato que
    los tokens de `detectar_textos`.
    """
    t = tol or _tolerancias(gris.shape)
    alto, ancho = gris.shape[:2]
    margen = t["margen_texto"]
    fondo = int(np.median(gris))
    x0r, y0r, x1r, y1r = rect if rect else (0, 0, ancho - 1, alto - 1)

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

    # ---------- eje X: franja bajo el eje ----------
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

    # ---------- eje Y: franja a la izquierda del eje ----------
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


# Letras válidas en un título: latín (con tildes) y GRIEGO. Sin el griego, un
# título o una leyenda como "Ω", "λ (nm)" o "Δt" se descartaba por "no tener letras".
_RE_LETRA = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\u0370-\u03FF\u2126\u00B5]"


def _texto_plausible(texto):
    """
    Un título de verdad es mayoritariamente letras. El OCR sobre números
    rotados devuelve cosas como "» es 2 @ 8 ° 8 8 8": eso NO es un título.
    Quita los tokens sin ningún alfanumérico y exige que las letras sean al
    menos la mitad de lo que queda.
    """
    if not texto:
        return False
    toks = [w for w in texto.split() if re.search(_RE_LETRA[:-1] + r"0-9]", w)]
    limpio = "".join(toks)
    if not limpio:
        return False
    letras = len(re.findall(_RE_LETRA, limpio))
    return letras >= 1 and letras / len(limpio) >= 0.5


def _limite_titulo_vertical(gris, eje_y, eje_x, fondo, margen):
    """
    Cuando no se pudieron leer los números del eje Y, se localiza la banda de
    esos números por la tinta: la franja pegada al eje son las marcas y sus
    números, y lo que quede a su izquierda, separado por un hueco, es el título
    rotado. Devuelve la x donde termina el título (o None si no hay título).
    """
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
        return None                 # una sola banda: son las marcas, no hay título
    return runs[-1][0]              # inicio de la banda pegada al eje


# ============================================================
# 3b) TEXTO VERTICAL (título del eje Y, rotado 90°)
# ============================================================

def detectar_texto_vertical(gris, x_limite, margen=6, ancho_minimo=15, lang=None):
    """
    Lee el título del eje Y (texto rotado 90°) en la franja a la izquierda de
    `x_limite`.

    Cambios respecto a la versión anterior:
      - `x_limite` ahora es el borde izquierdo de los NÚMEROS del eje Y, no el
        eje: si se pasaba la columna del eje, la franja incluía los números
        (1000, 800, 600...) y el OCR los devolvía como "» es 2 @ 8 ° 8 8 8";
      - la franja se agranda antes de rotarla (el texto rotado suele ser chico);
      - el resultado se descarta si no parece un título (`_texto_plausible`);
      - la rotación ganadora se elige por confianza x sqrt(nº de caracteres).
    """
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


# ============================================================
# 3c) TEXTOS CORTOS Y SÍMBOLOS: título del eje X y leyenda
# ============================================================
#
# Por qué existe: el OCR global (`psm 11`) se salta los textos de UNA sola letra
# ("r", "t", "x") y no conoce los símbolos griegos si solo está instalado `eng`
# (la "Ω" de una leyenda salía como "—_—N" con confianza 0 y se descartaba).
# Igual que con los números de los ejes, aquí se usa la geometría ya conocida:
#   - título del eje X: la primera línea de tinta DEBAJO de los números del eje;
#   - leyenda: se busca la "muestra" de cada serie (el trazo corto del mismo
#     color que la curva, fuera del área de dibujo) y se lee lo que tiene a la
#     derecha.
# Cada texto se recorta, se agranda y se lee solo. Si es un único glifo, además
# se compara contra plantillas dibujadas con fuentes del sistema (latín +
# griego + símbolos), lo que permite reconocer "Ω" aunque Tesseract no pueda.

_SIMBOLOS_PLANTILLA = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "αβγδεζηθικλμνξπρστυφχψω"      # griego minúscula
    "ΓΔΘΛΞΠΣΦΨΩ"                   # griego mayúscula (las demás = latín)
    "°%±∞"
)

_FUENTES_CANDIDATAS = [
    # Windows (Excel usa Calibri por defecto)
    r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\segoeui.ttf",
    # Linux
    "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]

_CACHE_PLANTILLAS = None


def _cargar_plantillas(max_fuentes=6, tam=64):
    """
    Dibuja cada símbolo con las fuentes disponibles y guarda el glifo recortado
    (tinta=1.0, fondo=0.0). Se hace una sola vez por proceso.
    Variable de entorno opcional SEGMENTADOR_FUENTES="ruta1;ruta2" para añadir
    fuentes propias (p. ej. la que usa la editorial de los libros).
    """
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
        # glifo "tofu" (carácter inexistente): si un símbolo se dibuja igual,
        # la fuente no lo tiene y no se usa como plantilla.
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
    """
    `tinta`: recorte float (1 = tinta, 0 = fondo) ajustado a UN glifo.
    Devuelve [(simbolo, puntaje)] ordenado de mejor a peor (puntaje ~ -1..1).

    La plantilla se reduce a la MISMA altura en píxeles que el glifo real con
    INTER_AREA, así sufre el mismo emborronamiento que un texto de 8-10 px, y
    se prueban desplazamientos de ±1 px porque a ese tamaño medio píxel cambia
    mucho la correlación.
    """
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
    """Igual que _idioma_ocr, pero añade griego si está instalado (ell / grc)."""
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
    """Recorte de gris -> (tinta float 0..1, máscara binaria). Tolera fondo oscuro."""
    c = crop.astype(np.float32)
    fondo = float(np.percentile(c, 90)) if np.median(c) >= 127 else float(np.percentile(c, 10))
    tinta = np.abs(fondo - c)
    mx = float(tinta.max())
    if mx < 30:
        return None, None
    tinta /= mx
    return tinta, tinta > 0.35


def _glifos(binaria):
    """Cajas (x, y, w, h) de los glifos, uniendo trozos que se solapan en x (la 'i', el '%')."""
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
    """
    Lee un texto corto (una palabra, una letra o un símbolo) dentro de `caja`
    = (x, y, w, h). Devuelve (texto, confianza, metodo) o ("", 0, None).
    """
    alto, ancho = gris.shape[:2]
    x, y, w, h = caja
    c = 2
    crop = gris[max(0, y - c):min(alto, y + h + c), max(0, x - c):min(ancho, x + w + c)]
    if crop.size == 0:
        return "", 0.0, None
    tinta, binaria = _tinta_de_recorte(crop)
    if tinta is None:
        return "", 0.0, None
    # El margen de 2 px es solo contexto para el reescalado: la tinta que cae
    # ahí (la punta de la muestra de la leyenda, un tick) no es parte del texto.
    ox, oy = x - max(0, x - c), y - max(0, y - c)
    dentro = np.zeros_like(binaria)
    dentro[oy:oy + h, ox:ox + w] = True
    tinta = np.where(dentro, tinta, 0.0).astype(np.float32)
    binaria &= dentro
    glifos = _glifos(binaria)
    if not glifos:
        return "", 0.0, None

    # ---------- Tesseract sobre el recorte agrandado ----------
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

    # ---------- Varios glifos: Tesseract manda; plantillas solo de respaldo ----------
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

    # ---------- Un solo glifo: se contrasta Tesseract con las plantillas ----------
    gx, gy, gw, gh = glifos[0]
    ranking = _clasificar_glifo(tinta[gy:gy + gh, gx:gx + gw])
    if not ranking:
        # Sin plantillas no hay forma de distinguir "Ω" de "Q"/"O": se devuelve
        # la lectura del OCR pero marcada, para que salga una advertencia.
        return (ocr_txt, ocr_conf, "ocr_sin_verificar") if ocr_txt else ("", 0.0, None)
    mejor_ch, mejor_p = ranking[0]
    puntajes = dict(ranking)
    if len(ocr_txt) == 1 and ocr_conf >= 50:
        # Tesseract gana si la forma de SU lectura también encaja casi igual de
        # bien (así una "r" sigue siendo "r" y no "Γ"); si propone "Q" o "O"
        # para una "Ω", la plantilla de "Ω" encaja claramente mejor y gana ella.
        if puntajes.get(ocr_txt, -9) >= mejor_p - 0.03:
            return ocr_txt, ocr_conf, "ocr"
    if mejor_p >= umbral_plantilla:
        return mejor_ch, 100.0 * mejor_p, "plantilla"
    if ocr_txt and ocr_conf >= 60 and _texto_plausible(ocr_txt):
        return ocr_txt, ocr_conf, "ocr"
    return "", 0.0, None


def _lineas_de_texto(gris, region, fondo=None):
    """
    Agrupa la tinta de `region` = (x0, y0, x1, y1) en líneas de texto y, dentro
    de cada línea, en frases (separadas por huecos grandes). Descarta restos de
    marcos y rejillas (trazos finos y muy largos).
    Devuelve una lista de cajas (x, y, w, h) en coordenadas de la imagen,
    ordenadas de arriba abajo y de izquierda a derecha.
    """
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

    # líneas: componentes que se solapan verticalmente
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
    """
    Busca la muestra de color de cada serie FUERA de la curva (trazo corto
    horizontal o cuadradito del mismo tono) y lee el texto a su derecha.
    Devuelve [{"hue", "texto", "conf", "metodo", "bbox", "muestra_bbox"}].
    """
    t = tol or _tolerancias(gris.shape)
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
            # el texto de ESTA muestra: misma fila y pegado a su derecha
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
    """
    Respaldo para leyendas sin muestra de color reconocible (series negras):
    lee las líneas de texto que haya A LA DERECHA del área de dibujo.
    """
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
    """
    El título del eje X es la PRIMERA línea de texto debajo de los números del
    eje. Se lee por regiones porque el OCR global no detecta títulos de una sola
    letra ("r", "t", "x").
    Devuelve (texto, bbox) o ("", None).
    """
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


# ============================================================
# 4) CALIBRACIÓN LINEAL ROBUSTA
# ============================================================

def ajustar_lineal_robusto(pixeles, valores, umbral_rel=0.04):
    """
    np.polyfit con todos los puntos (versión anterior) significa que UN número
    mal leído por el OCR ("100" leído como "10") desplaza toda la escala sin
    aviso. Para una gráfica táctil destinada a un estudiante ciego eso es el
    peor error posible: la figura sale plausible pero mal.

    Aquí: consenso tipo RANSAC sobre todos los pares, luego ajuste final con
    los inliers y R² reportado.
    Devuelve (m, b, info) con info = {n_total, n_usados, r2, descartados}.
    """
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


def ajustar_lineal(pixeles, valores):
    """Compatibilidad con la firma anterior."""
    m, b, _ = ajustar_lineal_robusto(pixeles, valores)
    return m, b


def _variantes_numero(texto):
    """
    Lecturas alternativas plausibles de un número mal leído por el OCR:
    punto decimal perdido ("15" <- 1.5), punto de más ("1.5" <- 15), un cero
    perdido o de sobra ("10" <- "1", "100" <- "1000").
    """
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
    """
    Corrige etiquetas que no encajan con la recta que forman las demás, pero
    SOLO si alguna lectura alternativa sí encaja (dentro de la tolerancia del
    ajuste). Necesita al menos 3 etiquetas coherentes para poder juzgar.
    Devuelve [(texto_antes, texto_despues)] y modifica `etiquetas` en el sitio.
    """
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


# ============================================================
# 5) MÁSCARA -> POLILÍNEA
# ============================================================

def mascara_a_polilinea(mascara, n_puntos=300, ventana_mediana=5):
    """
    Devuelve (cols, filas_suavizadas, cols_remuestreadas, filas_remuestreadas).

    Cambios:
      - el promedio por columna se calcula vectorizado (antes era un bucle con
        `filas_c[cols_c == col]` dentro, es decir O(n_columnas × n_píxeles):
        en una figura grande eso son decenas de millones de comparaciones);
      - se aplica mediana móvil para quitar el ruido del antialiasing;
      - se remuestrea a un número fijo de puntos, que es lo que necesita la
        etapa STL (una polilínea limpia y uniforme, no un punto por píxel).
    """
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


# ============================================================
# 6) PIPELINE COMPLETO
# ============================================================

def _calibrar_y_exportar(imagen, ruta_imagen, dir_resultados, uid,
                          eje_x, eje_y, rect, series, txt,
                          etiquetas_x, etiquetas_y, etiquetas_dato,
                          titulo_eje_y, titulo_eje_y_bbox,
                          advertencias, n_puntos=300):
    """
    Segunda mitad del pipeline: a partir de ejes/etiquetas/curva(s) YA
    DECIDIDOS (por la detección automática, o por el usuario corrigiendo en
    la pizarra) hace la calibración píxel->valor y escribe CSV/JSON/overlay.

    Cada elemento de `series` puede venir de dos formas:
      - {"mascara": np.ndarray bool, "hue":.., "modo":..}      (detección automática)
      - {"puntos_px": [(px,py), ...], "hue":.., "modo":..}     (curva editada a mano)
    """
    alto_imagen, ancho_imagen = imagen.shape[:2]

    # --- Calibración ---
    m_x, b_x, info_x = ajustar_lineal_robusto(
        [e["centro_x"] for e in etiquetas_x], [e["valor"] for e in etiquetas_x])
    m_y, b_y, info_y = ajustar_lineal_robusto(
        [e["centro_y"] for e in etiquetas_y], [e["valor"] for e in etiquetas_y])

    # --- Respaldo: calibrar Y con las etiquetas pegadas a la curva ---
    # (solo tiene sentido si la curva viene como máscara; una curva ya editada
    # a mano por el usuario no necesita este respaldo)
    calibrado_y_por_datos = False
    mascaras = [s["mascara"] for s in series if "mascara" in s]
    if m_y is None and etiquetas_dato and mascaras:
        union = np.zeros_like(mascaras[0], dtype=bool)
        for m in mascaras:
            union |= m
        filas_c, cols_c = np.where(union)
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
                m_y, b_y, info_y = ajustar_lineal_robusto(pares_fila, pares_valor)
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
        if info["r2"] is not None and info["r2"] < 0.995:
            advertencias.append(
                f"El eje {nombre} no ajusta bien a una recta (R²={info['r2']:.3f}); "
                "¿escala logarítmica o números mal leídos?"
            )
        if info["descartados"]:
            advertencias.append(
                f"Eje {nombre}: se descartaron como erróneos los valores {info['descartados']}."
            )

    def pixel_a_x(x_px):
        return None if m_x is None else float(m_x * x_px + b_x)

    def pixel_a_y(y_px):
        return None if m_y is None else float(m_y * y_px + b_y)

    # --- Polilíneas por serie ---
    series_salida = []
    for i, s in enumerate(series, start=1):
        if "puntos_px" in s:
            puntos_px = s["puntos_px"]
        else:
            pl = mascara_a_polilinea(s["mascara"], n_puntos=n_puntos)
            if pl is None:
                continue
            _, _, cols_rs, filas_rs = pl
            puntos_px = list(zip(cols_rs, filas_rs))
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

    # --- Overlay de verificación ---
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
    for bbox, color in ((txt["titulo_bbox"], (255, 255, 0)),
                        (txt["titulo_x_bbox"], (0, 165, 255)),
                        (titulo_eje_y_bbox, (255, 0, 255))):
        if bbox:
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

    # --- JSON estructurado (entrada natural de la etapa BANA/STL) ---
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
            },
            "series": series_salida,
            "resumen": resumen,
            "advertencias": advertencias,
            "_ruta_imagen_original": os.path.abspath(ruta_imagen),
        }, f, ensure_ascii=False, indent=2)

    # --- CSV (mismo esquema de columnas que la versión anterior) ---
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
                desc = (f"{etiqueta} leído por OCR: '{texto}', "
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
    """
    Recalcula CSV/JSON/overlay a partir de lo que el usuario corrigió a mano
    en la pizarra. `correcciones` trae SOLO lo que el usuario tocó; lo que
    falte se completa con el resultado original (ruta_json_original), así el
    usuario no tiene que rehacer todo, solo arreglar lo que falló.

    Formato de `correcciones` (todas las claves son opcionales):
    {
      "eje_x": [x1, y1, x2, y2] | null,
      "eje_y": [x1, y1, x2, y2] | null,
      "etiquetas_x": [{"centro_x": num, "valor": num}, ...],
      "etiquetas_y": [{"centro_y": num, "valor": num}, ...],
      "series": [{"puntos_px": [[px, py], ...], "nombre": "Ω"}, ...],
      "titulo_eje_x": "r", "leyenda": "Ω"
    }
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

    if "series" in correcciones:
        series = [{"puntos_px": [(float(p[0]), float(p[1])) for p in s["puntos_px"]],
                   "hue": s.get("hue"), "modo": "manual", "nombre": s.get("nombre", "")}
                  for s in correcciones["series"]]
    else:
        series = [{"puntos_px": [(p["px"], p["py"]) for p in s["puntos"]],
                   "hue": s.get("hue"), "modo": s.get("modo"), "nombre": s.get("nombre", "")}
                  for s in base.get("series", [])]

    txt = {
        "titulo": textos_base.get("titulo", ""), "titulo_bbox": None,
        "titulo_x": textos_base.get("titulo_eje_x", ""), "titulo_x_bbox": None,
        "leyenda": textos_base.get("leyenda", ""),
        "leyenda_entradas": textos_base.get("leyenda_entradas", []),
    }
    titulo_eje_y = textos_base.get("titulo_eje_y", "")
    # textos corregidos a mano (p. ej. un símbolo que el OCR leyó mal)
    if "titulo_eje_x" in correcciones:
        txt["titulo_x"] = correcciones["titulo_eje_x"] or ""
    if "titulo_eje_y" in correcciones:
        titulo_eje_y = correcciones["titulo_eje_y"] or ""
    if "leyenda" in correcciones:
        txt["leyenda"] = correcciones["leyenda"] or ""
        txt["leyenda_entradas"] = []

    advertencias = ["Resultado corregido manualmente por el usuario en la pizarra."]
    uid = uuid.uuid4().hex[:8]

    return _calibrar_y_exportar(
        imagen=imagen, ruta_imagen=ruta_imagen, dir_resultados=dir_resultados, uid=uid,
        eje_x=eje_x, eje_y=eje_y, rect=rect, series=series, txt=txt,
        etiquetas_x=etiquetas_x, etiquetas_y=etiquetas_y, etiquetas_dato=etiquetas_dato,
        titulo_eje_y=titulo_eje_y, titulo_eje_y_bbox=None,
        advertencias=advertencias, n_puntos=n_puntos,
    )


def procesar_imagen(ruta_imagen, dir_resultados, n_puntos=300, max_series=3, lang=None):
    """
    Ejecuta el pipeline y devuelve un dict con:
      ruta_csv / nombre_csv, ruta_json / nombre_json, ruta_overlay /
      nombre_overlay, resumen, advertencias, series, puntos_curva.

    (El docstring anterior prometía un ZIP con 4 CSVs y devolvía un solo CSV;
    aquí la documentación y el retorno ya coinciden.)
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

    # --- Ejes ---
    eje_x, eje_y = detectar_ejes(gris, tol=t)
    if eje_x is None:
        advertencias.append("No se detectó el eje X: no habrá calibración horizontal.")
    if eje_y is None:
        advertencias.append("No se detectó el eje Y: no habrá calibración vertical.")
    eje_x_fila = float((eje_x[1] + eje_x[3]) / 2) if eje_x else None
    eje_y_col = float((eje_y[0] + eje_y[2]) / 2) if eje_y else None
    rect = area_de_dibujo(eje_x, eje_y, imagen.shape)

    # --- Curvas ---
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

    # --- Etiquetas numéricas de los ejes (lectura por regiones) ---
    reg_x, reg_y = detectar_etiquetas_ejes(gris, eje_x, eje_y, rect=rect, tol=t)
    usa_reg_x, usa_reg_y = len(reg_x) >= 2, len(reg_y) >= 2
    excluir = (reg_x if usa_reg_x else []) + (reg_y if usa_reg_y else [])

    # --- Texto (títulos, leyenda, etiquetas de dato) ---
    txt = detectar_textos(gris, eje_x_fila, eje_y_col, rect=rect,
                          mascaras_curva=mascaras, tol=t, lang=lang,
                          excluir=excluir)
    # Si la lectura por regiones no consiguió al menos 2 números en un eje, se
    # queda con lo mejor de las dos lecturas.
    etiquetas_x = reg_x if usa_reg_x else max(reg_x, txt["etiquetas_x"], key=len)
    etiquetas_y = reg_y if usa_reg_y else max(reg_y, txt["etiquetas_y"], key=len)
    etiquetas_dato = txt["etiquetas_dato"]

    # --- Título del eje Y (vertical) ---
    # El límite es el borde izquierdo de los NÚMEROS del eje, no el eje mismo.
    x_limite_vertical = None
    if etiquetas_y:
        x_limite_vertical = min(e["px"] for e in etiquetas_y)
    elif eje_y is not None:
        x_limite_vertical = _limite_titulo_vertical(
            gris, eje_y, eje_x, int(np.median(gris)), t["margen_texto"])
    titulo_eje_y, titulo_eje_y_bbox = ("", None)
    if x_limite_vertical is not None:
        titulo_eje_y, titulo_eje_y_bbox = detectar_texto_vertical(
            gris, x_limite_vertical, lang=lang)

    # --- Leyenda (muestra de color + texto) y título del eje X por regiones ---
    # Se hacen DESPUÉS del OCR global y solo completan/corrigen lo que este no
    # leyó: títulos de una letra ("r") y símbolos que `eng` no conoce ("Ω").
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
        por_forma = [e["texto"] for e in leyenda_entradas if e["metodo"] == "plantilla"]
        if por_forma:
            advertencias.append(
                "Leyenda: " + ", ".join(f"'{z}'" for z in por_forma) + " se leyó por "
                "comparación de formas (símbolo que el OCR no reconoce). Verifícalo en el overlay."
            )
    txt["leyenda_entradas"] = leyenda_entradas

    if not txt["titulo_x"]:
        tx, tx_bbox = detectar_titulo_eje_x_por_region(
            gris, eje_x, etiquetas_x, rect, tol=t, lang=lang, excluir_cajas=cajas_leyenda)
        if tx:
            txt["titulo_x"], txt["titulo_x_bbox"] = tx, tx_bbox
            if tx_bbox.get("metodo") == "plantilla":
                advertencias.append(
                    f"Título del eje X leído por comparación de formas: '{tx}'. Verifícalo."
                )

    # --- Corrección por consistencia (punto decimal perdido, cero de más...) ---
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
        advertencias=advertencias, n_puntos=n_puntos,
    )

# ============================================================
# 7) DIAGNÓSTICO
# ============================================================

def diagnostico(ruta_imagen=None, lang=None):
    """
    Muestra qué está fallando cuando no salen el título del eje X o la leyenda:
        python -c "import segmentador; segmentador.diagnostico('grafica.png')"
    """
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
