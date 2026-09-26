# -*- coding: utf-8 -*-
"""
narracion.py
============
Descripción en texto de una gráfica ya segmentada, para que un software la
narre en voz alta, y exportación para Hand_Tracking (el programa que narra lo
que hay bajo el dedo mientras se recorre la lámina impresa).

Dos salidas:

describir_grafica(datos, pie_figura, diseno)
    Un párrafo en español: pie de figura del PDF (si se encontró), título,
    ejes y su rango, y cada curva con su tendencia, máximo y mínimo.

exportar_hand_tracking(datos, diseno, descripcion)
    Un dict JSON con los puntos y tramos de cada curva, los ejes y la
    leyenda, en milímetros sobre la placa (origen arriba a la izquierda,
    y hacia abajo, igual que la imagen de la cámara), cada uno con el texto
    que hay que narrar. Hand_Tracking/rastreo_gesto_pinza_grafica_autocalibrada.py
    lo carga con:  python rastreo_gesto_pinza_grafica_autocalibrada.py narracion.json

`datos` es el JSON que escribe segmentador.py (claves ejes, textos, series,
resumen). `diseno` es el dict que llena generador_stl.generar_modelo_desde_recta
con dónde quedó cada cosa en la placa; sin él solo se puede describir, no
ubicar.
"""
import math

from generador_stl import nombre_de_serie, simplificar_polilinea

FORMATO = "tactiverso-narracion"
VERSION = 1

# Puntos narrables por curva en Hand_Tracking: pocos y en los cambios de
# dirección (la polilínea simplificada), no los ~300 del segmentador.
PUNTOS_CLAVE_MAX = 12
PUNTOS_CLAVE_TOLERANCIA_MM = 3.0


# ------------------------------------------------------------------
# Números en español
# ------------------------------------------------------------------
def _decimales_para(rango):
    """Decimales razonables para leer valores de un eje con este rango
    (0-100 -> 0 decimales, 0-5 -> 1, 0-1 -> 2): más que eso es ruido de
    píxeles, no dato."""
    if not rango or not math.isfinite(rango) or rango <= 0:
        return 2
    return max(0, min(3, 1 - int(math.floor(math.log10(rango)))))


def _fmt(valor, decimales):
    if valor is None or not math.isfinite(valor):
        return "?"
    texto = f"{valor:.{decimales}f}"
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    if texto in ("-0", ""):
        texto = "0"
    return texto.replace(".", ",")


# ------------------------------------------------------------------
# Lectura del JSON del segmentador
# ------------------------------------------------------------------
def _rango_ejes(datos):
    """(x_min, x_max, y_min, y_max) en valores reales, o None por eje sin
    calibración."""
    ejes = datos.get("ejes") or {}
    rect = ejes.get("rect_grafico") or (datos.get("resumen") or {}).get("rect_grafico")
    cx, cy = ejes.get("calibracion_x") or {}, ejes.get("calibracion_y") or {}
    rx = ry = None
    if rect and len(rect) == 4:
        if cx.get("m") is not None and cx.get("b") is not None:
            a, b = cx["m"] * rect[0] + cx["b"], cx["m"] * rect[2] + cx["b"]
            rx = (min(a, b), max(a, b))
        if cy.get("m") is not None and cy.get("b") is not None:
            a, b = cy["m"] * rect[1] + cy["b"], cy["m"] * rect[3] + cy["b"]
            ry = (min(a, b), max(a, b))
    return rx, ry


def _marcas(etiquetas):
    """(menor, mayor) de los números leídos en un eje, o None si hay menos de 2."""
    valores = [e["valor"] for e in etiquetas or []
               if e.get("valor") is not None and math.isfinite(e["valor"])]
    return (min(valores), max(valores)) if len(valores) >= 2 else None


def _valores_serie(serie):
    return [(p["valor_x"], p["valor_y"]) for p in serie.get("puntos") or []
            if p.get("valor_x") is not None and p.get("valor_y") is not None]


def _tendencia(delta, rango, umbral=0.05):
    if rango and abs(delta) > umbral * rango:
        return "aumenta" if delta > 0 else "disminuye"
    return "se mantiene aproximadamente estable"


# ------------------------------------------------------------------
# Descripción en texto
# ------------------------------------------------------------------
def describir_grafica(datos, pie_figura=None, diseno=None):
    """Párrafo en español que describe la gráfica, listo para narrar."""
    textos = datos.get("textos") or {}
    series = [s for s in datos.get("series") or [] if s.get("puntos")]
    rx, ry = _rango_ejes(datos)
    dec_x = _decimales_para(rx[1] - rx[0]) if rx else 2
    dec_y = _decimales_para(ry[1] - ry[0]) if ry else 2
    tit_x = (textos.get("titulo_eje_x") or "").strip()
    tit_y = (textos.get("titulo_eje_y") or "").strip()

    frases = []
    pie = (pie_figura or "").strip()
    if pie:
        frases.append(pie if pie.endswith((".", "!", "?")) else pie + ".")

    titulo = (textos.get("titulo") or "").strip()
    n = len(series)
    tipo = "Gráfica de líneas" if n <= 1 else f"Gráfica de líneas con {n} curvas"
    frases.append(f"{tipo} titulada «{titulo}»." if titulo else f"{tipo}.")

    for nombre_eje, tit, rango, dec, marcas in (
            ("horizontal", tit_x, rx, dec_x, _marcas(textos.get("etiquetas_eje_x"))),
            ("vertical", tit_y, ry, dec_y, _marcas(textos.get("etiquetas_eje_y")))):
        partes = [f"Eje {nombre_eje}"]
        if tit:
            partes.append(f": {tit}")
        if marcas:
            # Lo que dicen los números del eje, no el borde del recuadro
            # (que en muchos gráficos queda en valores como 0,4 o 12,6).
            partes.append(f", con marcas de {_fmt(marcas[0], 3)} a {_fmt(marcas[1], 3)}")
        elif rango:
            partes.append(f", de {_fmt(rango[0], dec)} a {_fmt(rango[1], dec)}")
        else:
            partes.append(", sin escala legible")
        frases.append("".join(partes) + ".")

    texturas = {}
    if diseno:
        texturas = {s["indice"]: s["textura"] for s in diseno.get("series") or []}

    for i, serie in enumerate(series):
        nombre = nombre_de_serie(serie.get("nombre"), i)
        sujeto = f"La curva «{nombre}»" if n > 1 else "La curva"
        if i in texturas and n > 1:
            sujeto += f" (textura {texturas[i]} en la lámina)"
        vals = _valores_serie(serie)
        if len(vals) < 2:
            frases.append(f"{sujeto} no tiene escala legible para leer sus valores.")
            continue
        vals.sort()
        (x0, y0), (x1, y1) = vals[0], vals[-1]
        x_max, y_max = max(vals, key=lambda v: v[1])
        x_min, y_min = min(vals, key=lambda v: v[1])
        rango = (ry[1] - ry[0]) if ry else (y_max - y_min)
        frases.append(
            f"{sujeto} empieza en {_fmt(y0, dec_y)} cuando {tit_x or 'x'} vale {_fmt(x0, dec_x)}"
            f" y termina en {_fmt(y1, dec_y)} cuando vale {_fmt(x1, dec_x)}; en general "
            f"{_tendencia(y1 - y0, rango)}. Su valor máximo es {_fmt(y_max, dec_y)}, cuando "
            f"{tit_x or 'x'} vale {_fmt(x_max, dec_x)}, y el mínimo es {_fmt(y_min, dec_y)}, "
            f"cuando vale {_fmt(x_min, dec_x)}."
        )
    return " ".join(frases)


# ------------------------------------------------------------------
# Exportación para Hand_Tracking
# ------------------------------------------------------------------
def exportar_hand_tracking(datos, diseno, descripcion):
    """Dict JSON con puntos y tramos narrables ubicados en la placa (mm,
    origen arriba a la izquierda). Ver el docstring del módulo."""
    textos = datos.get("textos") or {}
    ancho = float(diseno["placa"]["ancho_mm"])
    alto = float(diseno["placa"]["alto_mm"])
    area, dom = diseno["area"], diseno["dominio"]
    calibrado = dom.get("calibrado")
    rango_x = (dom["x_max"] - dom["x_min"]) or 1.0
    rango_y = (dom["y_max"] - dom["y_min"]) or 1.0
    dec_x, dec_y = _decimales_para(rango_x), _decimales_para(rango_y)
    tit_x = (textos.get("titulo_eje_x") or "").strip() or "x"
    tit_y = (textos.get("titulo_eje_y") or "").strip() or "valor"
    varias = len(diseno.get("series") or []) > 1

    def a_hoja(x, y):
        return round(float(x), 1), round(alto - float(y), 1)

    def a_valor(x, y):
        return (dom["x_min"] + (x - area["izquierda"]) / area["ancho"] * rango_x,
                dom["y_min"] + (y - area["abajo"]) / area["alto"] * rango_y)

    puntos, segmentos, resumen_series = [], [], []
    for s in diseno.get("series") or []:
        nombre = s["nombre"]
        prefijo = f"{nombre}. " if varias else ""
        clave = simplificar_polilinea([tuple(p) for p in s["puntos_mm"]],
                                      tolerancia=PUNTOS_CLAVE_TOLERANCIA_MM,
                                      max_puntos=PUNTOS_CLAVE_MAX)
        resumen_series.append({"nombre": nombre, "textura": s["textura"]})
        anteriores = []
        for k, (x, y) in enumerate(clave):
            vx, vy = a_valor(x, y)
            hx, hy = a_hoja(x, y)
            if calibrado:
                texto = f"{prefijo}{tit_x} {_fmt(vx, dec_x)}: {tit_y} {_fmt(vy, dec_y)}."
            else:
                texto = f"{prefijo}Punto {k + 1} de {len(clave)} de la curva."
            punto = {"id": f"s{s['indice'] + 1}_p{k + 1}", "serie": nombre,
                     "x": hx, "y": hy, "texto": texto}
            if calibrado:
                punto.update({"valor_x": vx, "valor_y": vy})
            puntos.append(punto)
            anteriores.append((punto, vx, vy))
        for (p1, vx1, vy1), (p2, vx2, vy2) in zip(anteriores[:-1], anteriores[1:]):
            delta = vy2 - vy1
            if abs(delta) <= 0.01 * rango_y:
                tendencia, verbo = "estable", "se mantiene"
            elif delta > 0:
                tendencia, verbo = "aumento", "sube"
            else:
                tendencia, verbo = "disminución", "baja"
            if calibrado:
                texto = (f"{prefijo}Entre {tit_x} {_fmt(vx1, dec_x)} y {_fmt(vx2, dec_x)}, "
                         f"{verbo} de {_fmt(vy1, dec_y)} a {_fmt(vy2, dec_y)}.")
            else:
                texto = f"{prefijo}En este tramo la curva {verbo}."
            segmentos.append({
                "id": f"{p1['id']}-{p2['id'].split('_')[-1]}", "tipo": "curva", "serie": nombre,
                "x1": p1["x"], "y1": p1["y"], "x2": p2["x"], "y2": p2["y"],
                "tendencia": tendencia, "texto": texto,
            })

    for entrada in diseno.get("leyenda") or []:
        x0, y0, x1, y1 = entrada["recuadro"]
        (ax, ay), (bx, by) = a_hoja(x0, (y0 + y1) / 2), a_hoja(x1, (y0 + y1) / 2)
        segmentos.append({
            "id": f"leyenda_{entrada['indice'] + 1}", "tipo": "leyenda", "serie": entrada["nombre"],
            "x1": ax, "y1": ay, "x2": bx, "y2": by,
            "texto": f"Leyenda: la textura {entrada['textura']} es {entrada['nombre']}.",
        })

    for clave_eje, nombre_eje, tit, d0, d1, dec in (
            ("x", "horizontal", textos.get("titulo_eje_x"), dom["x_min"], dom["x_max"], dec_x),
            ("y", "vertical", textos.get("titulo_eje_y"), dom["y_min"], dom["y_max"], dec_y)):
        (ax, ay), (bx, by) = [a_hoja(*p) for p in diseno["ejes"][clave_eje]]
        texto = f"Eje {nombre_eje}" + (f": {tit.strip()}" if (tit or "").strip() else "")
        marcas = _marcas(textos.get(f"etiquetas_eje_{clave_eje}"))
        if marcas:
            texto += f", con marcas de {_fmt(marcas[0], 3)} a {_fmt(marcas[1], 3)}."
        else:
            texto += f", de {_fmt(d0, dec)} a {_fmt(d1, dec)}." if calibrado else ", sin escala."
        segmentos.append({"id": f"eje_{clave_eje}", "tipo": "eje", "serie": None,
                          "x1": ax, "y1": ay, "x2": bx, "y2": by, "texto": texto})

    return {
        "formato": FORMATO,
        "version": VERSION,
        "descripcion": descripcion,
        "placa": {"ancho_mm": ancho, "alto_mm": alto,
                  "origen": "esquina superior izquierda, y hacia abajo"},
        "series": resumen_series,
        "puntos": puntos,
        "segmentos": segmentos,
    }
