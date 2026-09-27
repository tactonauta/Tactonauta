import cv2
import numpy as np

def contar_paneles(image_path, umbral_gris=120, frac_largo=0.4):
    img = cv2.imread(image_path)
    if img is None:
        return 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, bw = cv2.threshold(gray, umbral_gris, 255, cv2.THRESH_BINARY_INV)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * frac_largo), 10)))
    verticales = cv2.erode(bw, kernel_v, iterations=1)
    verticales = cv2.dilate(verticales, kernel_v, iterations=1)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * frac_largo), 10), 1))
    horizontales = cv2.erode(bw, kernel_h, iterations=1)
    horizontales = cv2.dilate(horizontales, kernel_h, iterations=1)

    def clusters_de_posiciones(mask, eje):
        proyeccion = mask.sum(axis=eje)
        if proyeccion.max() == 0:
            return []
        posiciones = np.where(proyeccion > proyeccion.max() * 0.3)[0]
        if len(posiciones) == 0:
            return []
        clusters = [[posiciones[0]]]
        for p in posiciones[1:]:
            if p - clusters[-1][-1] <= 15:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return clusters

    clusters_v = clusters_de_posiciones(verticales, eje=0)
    clusters_h = clusters_de_posiciones(horizontales, eje=1)

    n_columnas = max(len(clusters_v) - 1, 1)
    n_filas = max(len(clusters_h) - 1, 1)

    return n_columnas * n_filas

def es_figura_compuesta(image_path, umbral_paneles=2):
    return contar_paneles(image_path) >= umbral_paneles


def clasificar_geometria(image_path, umbral_extent_relleno=0.45, area_min_relativa=0.003):
    img = cv2.imread(image_path)
    if img is None:
        return "indeterminado", 0.0, {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_area_img = h * w

    mask = ((gray < 200) & (gray > 20)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    formas = []
    for i in range(1, n_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < total_area_img * area_min_relativa:
            continue
        bbox_w = stats[i, cv2.CC_STAT_WIDTH]
        bbox_h = stats[i, cv2.CC_STAT_HEIGHT]
        bbox_area = bbox_w * bbox_h
        if bbox_area == 0:
            continue
        extent = area_px / bbox_area
        formas.append((extent, area_px / total_area_img))

    if not formas:
        return "indeterminado", 0.0, {}

    formas.sort(key=lambda x: -x[1])
    extent_principal, area_rel = formas[0]

    tipo = "barra_o_torta" if extent_principal >= umbral_extent_relleno else "linea"
    return tipo, extent_principal, {"n_formas": len(formas), "area_relativa": round(area_rel, 3)}


def es_dispersion(image_path, min_marcadores=15):
    """Gráfico de DISPERSIÓN (puntos sueltos, sin línea que los una): tiene
    trazos finos como uno de líneas, así que `clasificar_geometria` lo daba
    por "linea", y el segmentador unía los puntos en una curva que no
    existe. Se reconoce por sus marcadores de color: muchos, chicos, sin
    ningún trazo largo, redondos (no los guiones de una línea discontinua) y
    repartidos sin el espaciado regular de una línea punteada.
    (Un gráfico de dispersión en negro no se distingue por este camino.)"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)).astype(np.uint8)
    n, _, stats, centros = cv2.connectedComponentsWithStats(color, connectivity=8)
    comps = [(stats[i], centros[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if len(comps) < min_marcadores:
        return False
    anchos = np.array([s[cv2.CC_STAT_WIDTH] for s, _ in comps], float)
    altos = np.array([s[cv2.CC_STAT_HEIGHT] for s, _ in comps], float)
    if anchos.max() > 0.15 * img.shape[1]:
        return False          # hay un trazo largo: es una línea (con o sin marcadores)
    if np.median(np.maximum(anchos, altos) / np.maximum(1.0, np.minimum(anchos, altos))) > 2.0:
        return False          # guiones alargados: línea discontinua
    pts = np.array([c for _, c in comps], float)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    vecino = d.min(axis=1)
    if vecino.std() < 0.3 * vecino.mean():
        return False          # puntos equiespaciados: línea punteada
    return True


def es_grafico_lineal(image_path, umbral_extent_relleno=0.45):
    if es_figura_compuesta(image_path):
        return False, 0.0, "compuesta"
    tipo, extent, detalle = clasificar_geometria(image_path, umbral_extent_relleno)
    if tipo == "linea" and es_dispersion(image_path):
        return False, extent, "dispersion"
    confianza = 1 - extent if tipo == "linea" else extent
    return tipo == "linea", confianza, tipo


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        es_lineal, conf, etiqueta = es_grafico_lineal(path)
        print(f"{path}: {etiqueta} (conf {conf:.2f}) -> {'LINEAL' if es_lineal else 'descartado'}")