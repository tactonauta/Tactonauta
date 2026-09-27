import hashlib
import json
import os
import re
import shutil
import sys

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from classify_charts import es_grafico_lineal, es_figura_compuesta, clasificar_geometria, es_dispersion

AREA_MINIMA = 15000
CARPETA_RESULTADOS = "resultados_rapido"
BLACKLIST_PATH = os.path.join(CARPETA_RESULTADOS, "hashes_no_son_graficos.json")

_deplot_model = None
_deplot_processor = None


def cargar_deplot():
    global _deplot_model, _deplot_processor
    if _deplot_model is None:
        import torch
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        print("Cargando modelo DePlot (solo una vez)...")
        torch.set_default_device("cpu")
        _deplot_processor = Pix2StructProcessor.from_pretrained("google/deplot")
        _deplot_model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
    return _deplot_model, _deplot_processor


def extraer_datos_del_grafico(image_path):
    model, processor = cargar_deplot()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        images=image,
        text="Generate underlying data table of the figure below:",
        return_tensors="pt",
    )
    predictions = model.generate(**inputs, max_new_tokens=512)
    resultado = processor.decode(predictions[0], skip_special_tokens=True)
    return resultado.replace("<0x0A>", "\n")


def hash_archivo(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def es_proporcion_de_logo(w, h, tolerancia=0.25, area_maxima_logo=60000):
    area = w * h
    if area > area_maxima_logo:
        return False  # muy grande para ser un ícono/logo, sea cuadrada o no
    ratio = w / h if h else 0
    return (1 - tolerancia) <= ratio <= (1 + tolerancia)


def cargar_blacklist():
    if os.path.exists(BLACKLIST_PATH):
        with open(BLACKLIST_PATH) as f:
            return set(json.load(f))
    return set()


def guardar_blacklist(hashes):
    with open(BLACKLIST_PATH, "w") as f:
        json.dump(sorted(hashes), f, indent=2)


# Un pie de figura empieza con su rótulo y número: "Figura 3.", "Fig. 2:",
# "Gráfico 1 -", "Figure 4", "Ilustración IV"...
_PATRON_PIE = re.compile(
    r"^\s*(fig(ura|ure)?s?\.?|gr[aá]fic[oa]s?|chart|imagen|ilustraci[oó]n|diagrama)"
    r"\s*(n[°º.]\s*)?([0-9]+[a-z]?|[IVXivx]+)\b",
    re.IGNORECASE,
)
PIE_DISTANCIA_MAX = 60     # puntos PDF (~2 cm) entre la figura y su pie
PIE_LARGO_MAX = 600        # caracteres: un pie no es un párrafo entero


def extraer_pie_de_figura(page, bbox):
    """Texto del pie (o título) de la figura que ocupa `bbox` en la página:
    el bloque de texto más cercano, debajo o encima, que empiece con
    "Figura N" / "Gráfico N" / "Fig. N"... y se solape horizontalmente con
    la figura. Se prefiere el de abajo (convención habitual en papers); el
    de arriba cubre el estilo "Gráfico 1: título" de informes. "" si no hay.
    """
    r = fitz.Rect(bbox)
    mejor, mejor_puntaje = "", None
    for bloque in page.get_text("blocks"):
        x0, y0, x1, y1, texto = bloque[:5]
        if len(bloque) > 6 and bloque[6] != 0:   # bloque de imagen, no de texto
            continue
        texto = " ".join(str(texto).split())
        if not texto or not _PATRON_PIE.match(texto):
            continue
        solape = min(x1, r.x1) - max(x0, r.x0)
        if solape < 0.3 * min(x1 - x0, r.width):
            continue
        if y0 >= r.y1 - 2:              # debajo
            puntaje = y0 - r.y1
        elif y1 <= r.y0 + 2:            # encima
            puntaje = (r.y0 - y1) + 10
        else:                           # dentro del recuadro (dibujo vectorial)
            puntaje = 5
        if puntaje - 10 > PIE_DISTANCIA_MAX:
            continue
        if mejor_puntaje is None or puntaje < mejor_puntaje:
            mejor, mejor_puntaje = texto, puntaje
    if len(mejor) > PIE_LARGO_MAX:
        mejor = mejor[:PIE_LARGO_MAX].rsplit(" ", 1)[0] + "…"
    return mejor


# Texto alrededor de un dibujo vectorial que se considera parte del gráfico
# (números de los ejes, títulos, leyenda) y no del cuerpo del documento.
TEXTO_GRAFICO_DISTANCIA = 12   # pt entre un texto y lo que ya es parte del gráfico
TEXTO_GRAFICO_MAX_PALABRAS = 12
ESCALA_VECTORIAL = 2           # los dibujos vectoriales se rasterizan a 144 ppp


def _lineas_de_texto(page):
    """(recuadro, texto) de cada línea de texto de la página."""
    lineas = []
    for bloque in page.get_text("dict")["blocks"]:
        if bloque.get("type") != 0:
            continue
        for linea in bloque["lines"]:
            texto = " ".join(s["text"] for s in linea["spans"]).strip()
            if texto:
                lineas.append((fitz.Rect(linea["bbox"]), " ".join(texto.split())))
    return lineas


def _ampliar_con_texto(page, rect):
    """Agranda el recuadro de un dibujo vectorial para incluir SU texto.

    Los trazos de un gráfico (ejes, curvas, marco de la leyenda) son
    dibujos, pero los números de los ejes, los títulos y los nombres de la
    leyenda son TEXTO del PDF: si el recorte solo abarca los trazos, esos
    textos quedan afuera de la imagen y el segmentador no tiene con qué
    calibrar los ejes ni nombrar las curvas. Se suman las líneas de texto
    cortas pegadas al dibujo, creciendo de a poco (los números del eje
    están pegados a los trazos, y el título del eje pegado a los números);
    no el pie de figura ("Figura N. ...") ni oraciones del cuerpo del
    documento.
    """
    candidatas = []
    for caja, texto in _lineas_de_texto(page):
        palabras = texto.split()
        if _PATRON_PIE.match(texto) or len(palabras) > TEXTO_GRAFICO_MAX_PALABRAS:
            continue
        if len(palabras) >= 5 and texto.endswith("."):   # una oración, no un rótulo
            continue
        if caja.width > 1.3 * rect.width:
            continue
        candidatas.append(caja)

    ampliado = fitz.Rect(rect)
    for _ in range(4):
        d = TEXTO_GRAFICO_DISTANCIA
        zona = fitz.Rect(ampliado.x0 - d, ampliado.y0 - d, ampliado.x1 + d, ampliado.y1 + d)
        nuevas = [c for c in candidatas if c.intersects(zona) and not ampliado.contains(c)]
        if not nuevas:
            break
        for caja in nuevas:
            ampliado |= caja
    return ampliado


def extraer_palabras(page, clip, ancho_px, alto_px):
    """Palabras del PDF dentro de `clip`, en píxeles de la imagen extraída
    (ancho_px x alto_px). Son el texto EXACTO del documento: el segmentador
    las usa en vez del OCR cuando existen (gráficos vectoriales, o imágenes
    con el texto superpuesto como texto real). "vertical" marca las palabras
    rotadas (título del eje Y)."""
    clip = fitz.Rect(clip)
    if clip.width <= 0 or clip.height <= 0:
        return []
    sx, sy = ancho_px / clip.width, alto_px / clip.height
    # Recuadros de las líneas de texto ROTADAS según el propio PDF (su
    # dirección de escritura no es horizontal): así se reconoce también una
    # palabra corta como "mm" o "%", que por su forma no se distingue.
    rotadas = []
    # Potencias de diez escritas con exponente elevado y más chico ("10" +
    # "³", típico de un eje logarítmico o de un multiplicador "×10⁶"): como
    # palabra salen pegadas, "103", que se leía como ciento tres.
    potencias = []   # (recuadro, texto_pegado, texto_correcto)
    for bloque in page.get_text("dict", clip=clip)["blocks"]:
        for linea in bloque.get("lines", []):
            dx, _ = linea.get("dir", (1, 0))
            if abs(dx) < 0.5:
                rotadas.append(fitz.Rect(linea["bbox"]))
            spans = linea.get("spans", [])
            for base, exp in zip(spans, spans[1:]):
                b_txt, e_txt = base["text"].strip(), exp["text"].strip().replace("−", "-")
                if (b_txt.endswith("10") and re.fullmatch(r"[-+]?\d+", e_txt)
                        and exp["size"] < 0.85 * base["size"]
                        and exp["origin"][1] < base["origin"][1] - 0.5):
                    potencias.append((fitz.Rect(base["bbox"]) | fitz.Rect(exp["bbox"]),
                                      b_txt + exp["text"].strip(), b_txt[:-2] + "1e" + e_txt))
    palabras = []
    for x0, y0, x1, y1, texto, bloque, linea, num in page.get_text("words", clip=clip):
        texto = texto.strip()
        if not texto:
            continue
        for recuadro, pegado, correcto in potencias:
            if texto == pegado and recuadro.intersects(fitz.Rect(x0, y0, x1, y1)):
                texto = correcto
                break
        w, h = x1 - x0, y1 - y0
        palabras.append({
            "texto": texto,
            "px": round((x0 - clip.x0) * sx, 1), "py": round((y0 - clip.y0) * sy, 1),
            "pw": round(w * sx, 1), "ph": round(h * sy, 1),
            "vertical": (len(texto) > 1 and h > 1.5 * w)
                        or any(r.contains(fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)) for r in rotadas),
            "orden": [bloque, linea, num],
        })
    # Una palabra corta ("de") no se distingue sola: si otra de su misma
    # línea está rotada, la línea entera es vertical.
    lineas_verticales = {tuple(p["orden"][:2]) for p in palabras if p["vertical"]}
    for p in palabras:
        p["vertical"] = tuple(p["orden"][:2]) in lineas_verticales
    return palabras


def _agrupar_rects(rects, umbral=15):
    """Agrupa rectángulos de trazos vectoriales cercanos/superpuestos en un solo bloque."""
    grupos = [fitz.Rect(r) for r in rects]
    cambiado = True
    while cambiado:
        cambiado = False
        nuevos = []
        usados = [False] * len(grupos)
        for i in range(len(grupos)):
            if usados[i]:
                continue
            actual = fitz.Rect(grupos[i])
            for j in range(i + 1, len(grupos)):
                if usados[j]:
                    continue
                otro = grupos[j]
                cercano = (
                    actual.x0 - umbral <= otro.x1 and otro.x0 - umbral <= actual.x1 and
                    actual.y0 - umbral <= otro.y1 and otro.y0 - umbral <= actual.y1
                )
                if cercano:
                    actual |= otro
                    usados[j] = True
                    cambiado = True
            nuevos.append(actual)
            usados[i] = True
        grupos = nuevos
    return grupos


def extraer_dibujos_vectoriales(page, page_num, output_dir, contador_inicial):
    """Detecta gráficos dibujados directamente con vectores (matplotlib/R/LaTeX),
    agrupando los trazos sueltos (líneas, ejes, leyenda) en bloques y renderizándolos
    como imagen."""
    extraidas = []
    drawings = page.get_drawings()
    if not drawings:
        return extraidas, contador_inicial

    rects = [d["rect"] for d in drawings if d.get("rect") and d["rect"].width > 2 and d["rect"].height > 2]
    if not rects:
        return extraidas, contador_inicial

    grupos = _agrupar_rects(rects, umbral=15)

    contador = contador_inicial
    for g in grupos:
        if g.width < 80 or g.height < 60:  # descarta subrayados/viñetas sueltas
            continue
        contador += 1
        pad = 3
        g = _ampliar_con_texto(page, g)
        clip = fitz.Rect(g.x0 - pad, g.y0 - pad, g.x1 + pad, g.y1 + pad) & page.rect
        pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(ESCALA_VECTORIAL, ESCALA_VECTORIAL))
        nombre = f"page{page_num+1}_drawing{contador}.png"
        ruta = os.path.join(output_dir, nombre)
        pix.save(ruta)
        extraidas.append({"file_path": ruta, "page": page_num + 1, "width": clip.width, "height": clip.height,
                          "origen": "vectorial",
                          "pie_figura": extraer_pie_de_figura(page, clip),
                          "palabras": extraer_palabras(page, clip, pix.width, pix.height)})

    return extraidas, contador


def _pixmap_sobre_blanco(doc, xref):
    """La imagen `xref` en RGB y sin transparencia, sobre fondo BLANCO.

    Un PNG con fondo transparente (lo habitual al exportar un gráfico desde
    Excel o matplotlib) se guarda en el PDF como la imagen + una máscara de
    transparencia aparte ("SMask"). Leyendo solo la imagen, el fondo sale
    con el color que tenga debajo de la máscara —casi siempre negro— y el
    gráfico entero queda negro: el segmentador no encuentra ni ejes ni curva.
    """
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:                      # CMYK -> RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    smask = doc.extract_image(xref).get("smask") or 0
    if smask and not pix.alpha:
        try:
            pix = fitz.Pixmap(pix, fitz.Pixmap(doc, smask))
        except (RuntimeError, ValueError):
            pass                                    # máscara ilegible: se sigue sin ella
    if pix.alpha:
        muestras = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        color = muestras[:, :, :pix.n - 1].astype(np.float32)
        alfa = muestras[:, :, pix.n - 1:].astype(np.float32) / 255.0
        mezcla = (color * alfa + 255.0 * (1.0 - alfa)).round().astype(np.uint8)
        espacio = fitz.csRGB if pix.n - 1 == 3 else fitz.csGRAY
        pix = fitz.Pixmap(espacio, pix.width, pix.height, np.ascontiguousarray(mezcla).tobytes(), False)
    return pix


def extraer_imagenes_crudas(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    extraidas = []

    for page_num, page in enumerate(doc):
        # 1. Imágenes raster embebidas (fotos/capturas pegadas, ej. exportadas de Excel)
        for i, im in enumerate(page.get_image_info(xrefs=True)):
            xref = im["xref"]
            w = im["bbox"][2] - im["bbox"][0]
            h = im["bbox"][3] - im["bbox"][1]
            pix = _pixmap_sobre_blanco(doc, xref)
            nombre = f"page{page_num+1}_img{i+1}.png"
            ruta = os.path.join(output_dir, nombre)
            pix.save(ruta)
            extraidas.append({"file_path": ruta, "page": page_num + 1, "width": w, "height": h,
                              "origen": "imagen",
                              "pie_figura": extraer_pie_de_figura(page, im["bbox"]),
                              "palabras": extraer_palabras(page, im["bbox"], pix.width, pix.height)})

        # 2. Dibujos vectoriales (gráficos hechos con matplotlib/R/LaTeX directo en el PDF)
        vectoriales, _ = extraer_dibujos_vectoriales(page, page_num, output_dir, 0)
        extraidas.extend(vectoriales)

    return extraidas


def detectar_graficos(pdf_path, output_dir):
    """
    Versión "liviana" de procesar_pdf pensada para la API web: hace los
    pasos baratos (extracción + filtros + clasificación geométrica línea
    vs. barra/torta vs. compuesta) pero NO corre DePlot — ese modelo de
    1.1 GB ya no hace falta aquí porque, para lo que resulte "línea", el
    dato real se extrae después con segmentador.py (ejes + curva + OCR),
    que es más preciso que la reconstrucción aproximada de DePlot.

    Devuelve una lista de dicts (uno por figura que pasó los filtros
    baratos de tamaño/duplicado/logo), cada uno con:
      file_path, page, width, height, origen, pie_figura, palabras,
      tipo, confianza, es_lineal
    "pie_figura" es el texto "Figura N. ..." que el PDF pone junto a la
    figura ("" si no se encontró): es la base de la descripción narrada.
    "palabras" es el texto real del PDF dentro de la figura, en píxeles de
    la imagen (ver extraer_palabras): el segmentador lo prefiere al OCR.
    "tipo" es uno de: "linea", "barra_o_torta", "compuesta", "dispersion",
    "indeterminado".
    """
    os.makedirs(output_dir, exist_ok=True)
    blacklist = cargar_blacklist()

    extraidas = extraer_imagenes_crudas(pdf_path, output_dir)

    vistos = set()
    candidatos = []
    for r in extraidas:
        area = r["width"] * r["height"]
        if area < AREA_MINIMA:
            continue

        h = hash_archivo(r["file_path"])
        if h in vistos or h in blacklist:
            continue
        vistos.add(h)

        if es_proporcion_de_logo(r["width"], r["height"]):
            continue

        candidatos.append(r)

    resultados = []
    for r in candidatos:
        if es_figura_compuesta(r["file_path"]):
            r["tipo"] = "compuesta"
            r["confianza"] = 0.0
            r["es_lineal"] = False
        else:
            tipo, extent, _ = clasificar_geometria(r["file_path"])
            if tipo == "linea" and es_dispersion(r["file_path"]):
                tipo = "dispersion"   # puntos sueltos: no hay curva que unir
            confianza = round((1 - extent) if tipo == "linea" else extent, 2)
            r["tipo"] = tipo
            r["confianza"] = confianza
            r["es_lineal"] = tipo == "linea"
        resultados.append(r)

    return resultados


def procesar_pdf(pdf_path):
    nombre_base = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(CARPETA_RESULTADOS, f"extracted_{nombre_base}")
    lineales_dir = os.path.join(CARPETA_RESULTADOS, f"lineales_{nombre_base}")

    print(f"\n{'='*50}\nProcesando: {pdf_path}\n{'='*50}")

    if not os.path.exists(pdf_path):
        print("  ⚠️  No se encontró el archivo.")
        return

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    blacklist = cargar_blacklist()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 1. Extracción barata (sin IA): imágenes raster + dibujos vectoriales
    extraidas = extraer_imagenes_crudas(pdf_path, output_dir)
    print(f"Extraídas en bruto: {len(extraidas)}")

    # 2. Filtros baratos: tamaño, duplicados, blacklist, proporción (CON DIAGNÓSTICO)
    vistos = set()
    candidatos = []
    nuevos_logos = set()

    for r in extraidas:
        area = r["width"] * r["height"]
        nombre_img = os.path.basename(r["file_path"])

        if area < AREA_MINIMA:
            print(f"  ⏭️  {nombre_img}: muy chica ({r['width']:.0f}x{r['height']:.0f}={area:.0f}px²), descartada")
            continue

        h = hash_archivo(r["file_path"])
        if h in vistos:
            print(f"  ⏭️  {nombre_img}: duplicada dentro del mismo PDF, descartada")
            continue
        vistos.add(h)

        if h in blacklist:
            print(f"  ⏭️  {nombre_img}: en blacklist de logos (de un paper anterior), descartada")
            continue

        if es_proporcion_de_logo(r["width"], r["height"]):
            print(f"  🔲 {nombre_img}: proporción cuadrada ({r['width']:.0f}x{r['height']:.0f}) -> logo, descartado")
            nuevos_logos.add(h)
            continue

        print(f"  ✅ {nombre_img}: pasa filtros baratos ({r['width']:.0f}x{r['height']:.0f})")
        candidatos.append(r)

    print(f"Después de filtros baratos: {len(candidatos)}")

    # 3. Clasificación geométrica (barata, sin IA)
    lineales = []
    for r in candidatos:
        es_lineal, confianza, etiqueta = es_grafico_lineal(r["file_path"])
        estado = "LINEAL" if es_lineal else f"descartado ({etiqueta})"
        print(f"  {os.path.basename(r['file_path'])}: {estado} (conf {confianza:.2f})")
        if es_lineal:
            r["clasificacion_confianza"] = confianza
            lineales.append(r)

    print(f"Gráficos lineales: {len(lineales)} de {len(extraidas)} extraídos")

    # 4. SOLO ahora, sobre lo que sobrevivió, corremos el modelo pesado
    for r in lineales:
        print(f"  Extrayendo datos/ejes de {os.path.basename(r['file_path'])} con DePlot...")
        r["data_table"] = extraer_datos_del_grafico(r["file_path"])

    if os.path.exists(lineales_dir):
        shutil.rmtree(lineales_dir)
    os.makedirs(lineales_dir)
    for r in lineales:
        shutil.copy(r["file_path"], os.path.join(lineales_dir, os.path.basename(r["file_path"])))

    blacklist.update(nuevos_logos)
    guardar_blacklist(blacklist)

    with open(os.path.join(CARPETA_RESULTADOS, f"resultado_{nombre_base}.json"), "w") as f:
        json.dump(lineales, f, indent=2, ensure_ascii=False)

    print(f"\nGuardados en: {lineales_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python pipeline_rapido.py /ruta/al/paper.pdf")
        sys.exit(1)
    procesar_pdf(sys.argv[1])