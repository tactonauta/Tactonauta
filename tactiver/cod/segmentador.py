# -*- coding: utf-8 -*-
"""
segmentador.py  (versión corregida)
===================================
Pipeline 100% automático (sin clics manuales, sin SAM2) para:
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
    Todas las tolerancias se derivan del tamaño de la imagen. En la versión
    anterior estaban fijas en píxeles (largo_minimo=150, dy<15, radio_dato=18),
    lo que hacía que el mismo código fallara en imágenes chicas y fuera
    demasiado permisivo en imágenes grandes (un PDF a 300 dpi da páginas de
    2000+ px de ancho).
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
    El código anterior llamaba a Tesseract sin `lang`, o sea en inglés. Los
    títulos en español con tildes ("Número de salidas") salían mal leídos.
    Aquí se usa spa+eng si el paquete de español está instalado.
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

    eje_x = None
    if gh:
        # el eje X es la horizontal larga más baja del gráfico
        mejor = max(gh, key=lambda g: (round(g["pos"]), g["largo"]))
        fila = int(round(mejor["pos"]))
        eje_x = (int(mejor["a"]), fila, int(mejor["b"]), fila)

    eje_y = None
    if gv:
        if eje_x is not None:
            x_esq, y_esq = eje_x[0], eje_x[1]
            mejor = min(
                gv,
                key=lambda g: abs(g["pos"] - x_esq) + 0.5 * abs(g["b"] - y_esq) - 0.2 * g["largo"],
            )
        else:
            mejor = min(gv, key=lambda g: g["pos"])
        col = int(round(mejor["pos"]))
        eje_y = (col, int(mejor["a"]), col, int(mejor["b"]))

    return eje_x, eje_y


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
                    tol=None, escala_ocr=2.0, lang=None):
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

    # El título del eje X es la línea de texto MÁS BAJA de la zona inferior;
    # lo que quede a la altura de los números es parte de las etiquetas.
    if tok_titulo_x and etiquetas_x:
        base_num = max(e["py"] + e["ph"] for e in etiquetas_x)
        bajo = [z for z in tok_titulo_x if z["centro_y"] > base_num]
        if bajo:
            tok_titulo_x = bajo

    titulo, titulo_bbox = _unir(tok_titulo)
    titulo_x, titulo_x_bbox = _unir(tok_titulo_x)
    leyenda, _ = _unir(tok_leyenda)

    return {
        "etiquetas_x": etiquetas_x,
        "etiquetas_y": etiquetas_y,
        "etiquetas_dato": etiquetas_dato,
        "titulo": titulo, "titulo_bbox": titulo_bbox,
        "titulo_x": titulo_x, "titulo_x_bbox": titulo_x_bbox,
        "leyenda": leyenda,
    }


# ============================================================
# 3b) TEXTO VERTICAL (título del eje Y, rotado 90°)
# ============================================================

def detectar_texto_vertical(gris, x_limite, margen=6, ancho_minimo=15, lang=None):
    """
    Igual que antes (el mapeo de coordenadas rotadas estaba bien), pero la
    rotación ganadora ya no se elige solo por confianza promedio: una rotación
    equivocada produce basura corta con confianza alta. Ahora el puntaje pondera
    la cantidad de caracteres y exige al menos una letra.
    """
    x_limite = int(x_limite) - margen
    if x_limite < ancho_minimo:
        return "", None

    franja = gris[:, :x_limite]
    alto_franja, ancho_franja = franja.shape[:2]
    if alto_franja == 0 or ancho_franja == 0:
        return "", None

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
        if not re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", texto_final):
            continue

        puntaje = (sum(confs) / len(confs)) * math.sqrt(len(texto_final))
        if puntaje > mejor_puntaje:
            xs1 = [c[0] for c in cajas]
            ys1 = [c[1] for c in cajas]
            xs2 = [c[0] + c[2] for c in cajas]
            ys2 = [c[1] + c[3] for c in cajas]
            mejor_bbox = {"px": int(min(xs1)), "py": int(min(ys1)),
                          "pw": int(max(xs2) - min(xs1)), "ph": int(max(ys2) - min(ys1))}
            mejor_texto, mejor_puntaje = texto_final, puntaje

    return mejor_texto, mejor_bbox


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
    info = {"n_total": int(p.size), "n_usados": 0, "r2": None, "descartados": []}

    if p.size < 2 or np.unique(p).size < 2 or np.unique(v).size < 2:
        return None, None, info

    if p.size == 2:
        m, b = np.polyfit(p, v, 1)
        info.update(n_usados=2, r2=1.0)
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
    )
    return float(m), float(b), info


def ajustar_lineal(pixeles, valores):
    """Compatibilidad con la firma anterior."""
    m, b, _ = ajustar_lineal_robusto(pixeles, valores)
    return m, b


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

    # --- Texto ---
    txt = detectar_textos(gris, eje_x_fila, eje_y_col, rect=rect,
                          mascaras_curva=mascaras, tol=t, lang=lang)
    etiquetas_x = txt["etiquetas_x"]
    etiquetas_y = txt["etiquetas_y"]
    etiquetas_dato = txt["etiquetas_dato"]

    # --- Título del eje Y (vertical) ---
    if etiquetas_y:
        x_limite_vertical = min(e["px"] for e in etiquetas_y)
    else:
        x_limite_vertical = eje_y_col
    titulo_eje_y, titulo_eje_y_bbox = ("", None)
    if x_limite_vertical is not None:
        titulo_eje_y, titulo_eje_y_bbox = detectar_texto_vertical(
            gris, x_limite_vertical, lang=lang)

    # --- Calibración ---
    m_x, b_x, info_x = ajustar_lineal_robusto(
        [e["centro_x"] for e in etiquetas_x], [e["valor"] for e in etiquetas_x])
    m_y, b_y, info_y = ajustar_lineal_robusto(
        [e["centro_y"] for e in etiquetas_y], [e["valor"] for e in etiquetas_y])

    # --- Respaldo: calibrar Y con las etiquetas pegadas a la curva ---
    calibrado_y_por_datos = False
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
        pl = mascara_a_polilinea(s["mascara"], n_puntos=n_puntos)
        if pl is None:
            continue
        _, _, cols_rs, filas_rs = pl
        puntos = [{
            "px": float(c), "py": float(f),
            "valor_x": pixel_a_x(c), "valor_y": pixel_a_y(f),
        } for c, f in zip(cols_rs, filas_rs)]
        series_salida.append({
            "id": f"serie_{i}", "hue": s["hue"], "modo": s["modo"],
            "n_puntos": len(puntos), "puntos": puntos,
        })

    puntos_curva = series_salida[0]["puntos"] if series_salida else []

    # --- Overlay de verificación ---
    overlay = imagen.copy()
    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (200, 200, 200), 1)
    colores = [(0, 255, 0), (0, 200, 255), (255, 200, 0)]
    for i, s in enumerate(series):
        overlay[s["mascara"]] = colores[i % len(colores)]
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

    nombre_overlay = f"overlay_{uid}.png"
    ruta_overlay = os.path.join(dir_resultados, nombre_overlay)
    cv2.imwrite(ruta_overlay, overlay)

    resumen = {
        "eje_x_detectado": eje_x is not None,
        "eje_y_detectado": eje_y is not None,
        "rect_grafico": list(rect),
        "n_series": len(series_salida),
        "n_etiquetas_x": len(etiquetas_x),
        "n_etiquetas_y": len(etiquetas_y),
        "n_etiquetas_dato": len(etiquetas_dato),
        "titulo": txt["titulo"],
        "titulo_eje_x": txt["titulo_x"],
        "titulo_eje_y": titulo_eje_y,
        "leyenda": txt["leyenda"],
        "n_puntos_curva": len(puntos_curva),
        "calibrado_x": m_x is not None,
        "calibrado_y": m_y is not None,
        "calibrado_y_por_datos": calibrado_y_por_datos,
        "r2_x": info_x["r2"], "r2_y": info_y["r2"],
        "ancho_imagen": ancho_imagen, "alto_imagen": alto_imagen,
        # La etapa BANA/STL solo debería correr si esto es True.
        "listo_para_stl": bool(series_salida and m_x is not None and m_y is not None),
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
                "etiquetas_eje_x": [{"valor": e["valor"], "px": e["centro_x"]} for e in etiquetas_x],
                "etiquetas_eje_y": [{"valor": e["valor"], "py": e["centro_y"]} for e in etiquetas_y],
                "etiquetas_dato": [{"valor": e["valor"], "px": e["centro_x"],
                                    "py": e["centro_y"]} for e in etiquetas_dato],
            },
            "series": series_salida,
            "resumen": resumen,
            "advertencias": advertencias,
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

        if txt["leyenda"]:
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
