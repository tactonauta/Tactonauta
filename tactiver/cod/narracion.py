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

exportar_csv_hand_tracking(exportacion, ruta)
    Lo mismo en un CSV (una fila por elemento), para abrirlo en una hoja de
    cálculo o cargarlo en Hand_Tracking igual que el JSON.

exportar_datos_lamina(datos, diseno, descripcion, exportacion, figura)
    Todo lo que se sabe de la lámina, en un solo dict: de qué documento y
    página sale la figura, su pie y su contexto en el texto, los valores de
    cada curva y el mapa de la placa de exportar_hand_tracking. Es lo que
    abre el QR impreso (api.py, /q/<token>).

`datos` es el JSON que escribe segmentador.py (claves ejes, textos, series,
resumen). `diseno` es el dict que llena generador_stl.generar_modelo_desde_recta
con dónde quedó cada cosa en la placa; sin él solo se puede describir, no
ubicar.
"""
import csv
import io
import math
import numbers
import re
import time
import unicodedata

import numpy as np

from generador_stl import nombre_de_serie, simplificar_polilinea

FORMATO = "tactiverso-narracion"
VERSION = 1

# Puntos narrables por curva en Hand_Tracking: pocos y en los cambios de
# dirección (la polilínea simplificada), no los ~300 del segmentador.
PUNTOS_CLAVE_MAX = 12
PUNTOS_CLAVE_TOLERANCIA_MM = 3.0

# Lo que se narra del QR en relieve (generador_stl.agregar_qr).
TEXTO_QR = ("En la esquina superior izquierda hay un código QR en relieve: no es parte del "
            "gráfico. Identifica esta lámina: al escanearlo se abren sus datos (documento y "
            "página de donde sale, contexto y valores de cada curva).")

FORMATO_DATOS = "tactiverso-lamina"
VERSION_DATOS = 1
# Valores por curva en los datos del QR: suficientes para reconstruirla (el
# segmentador da ~300, casi todos redundantes).
VALORES_SERIE_MAX = 200


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


def _formato_lineal(rango):
    """Formato para un eje lineal: decimales según el rango y, en rangos
    grandes, redondeo a unas 3 cifras (1536725 -> 1540000 en un eje de
    0 a 12 millones): más precisión que eso es ruido de píxeles."""
    dec = _decimales_para(rango)
    paso = 1.0
    if rango and math.isfinite(rango) and rango > 0:
        exponente = int(math.floor(math.log10(rango))) - 2
        paso = 10.0 ** exponente if exponente > 0 else 1.0
    return lambda v: _fmt(round(v / paso) * paso if (v is not None and math.isfinite(v)) else v, dec)


def _valor_con_nombre(valor_txt, nombre):
    """Cómo se dice un valor del eje Y en un punto: "Miles de soles 78" si
    el eje tiene un nombre, "3 mm" / "20 %" si el nombre es una unidad corta,
    y solo "78" si el eje no tiene nombre."""
    if not nombre or nombre == "valor":
        return valor_txt
    if len(nombre) <= 5 and " " not in nombre:
        return f"{valor_txt} {nombre}"
    return f"{nombre} {valor_txt}"


def _fmt_significativo(valor):
    """Para ejes logarítmicos, donde un mismo eje va de 0,01 a 10 000: tres
    cifras significativas en vez de una cantidad fija de decimales."""
    if valor is None or not math.isfinite(valor) or valor == 0:
        return _fmt(valor, 0)
    return _fmt(valor, max(0, min(4, 2 - int(math.floor(math.log10(abs(valor)))))))


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

    def real(v, cal):
        return 10.0 ** v if cal.get("escala") == "log" else v

    if rect and len(rect) == 4:
        if cx.get("m") is not None and cx.get("b") is not None:
            a, b = real(cx["m"] * rect[0] + cx["b"], cx), real(cx["m"] * rect[2] + cx["b"], cx)
            rx = (min(a, b), max(a, b))
        if cy.get("m") is not None and cy.get("b") is not None:
            a, b = real(cy["m"] * rect[1] + cy["b"], cy), real(cy["m"] * rect[3] + cy["b"], cy)
            ry = (min(a, b), max(a, b))
    return rx, ry


def _es_log(datos, eje):
    cal = ((datos.get("ejes") or {}).get(f"calibracion_{eje}") or {})
    return cal.get("escala") == "log"


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
# Eje X: numérico o de categorías ("Ene", "Feb", ...)
# ------------------------------------------------------------------
def _categorias(textos):
    return [str(c) for c in (textos.get("categorias_x") or [])]


def _categoria_de(valor_x, categorias, tolerancia=0.25):
    """Nombre de la categoría en la posición `valor_x` (0, 1, 2...), o None
    si el valor no cae sobre una categoría."""
    if not categorias or valor_x is None or not math.isfinite(valor_x):
        return None
    i = int(round(valor_x))
    if abs(valor_x - i) <= tolerancia and 0 <= i < len(categorias):
        return categorias[i]
    return None


class _EjeX:
    """Cómo nombrar una posición del eje X al narrar: "cuando Mes vale 3" en
    un eje numérico, "en Mar" en uno de categorías."""

    def __init__(self, titulo, formato, categorias):
        self.titulo = titulo
        self.fmt = formato
        self.cats = categorias

    def valor(self, x):
        return _categoria_de(x, self.cats) or self.fmt(x)

    def en(self, x):
        cat = _categoria_de(x, self.cats)
        if cat:
            return f"en {cat}"
        return f"cuando {self.titulo or 'x'} vale {self.fmt(x)}"


def _forma(vals, rango_y, eje, fmt_y):
    """Frase con la forma de la curva: si sube y luego baja (o al revés), lo
    dice; "en general disminuye" no describe bien una curva en U."""
    (x0, y0), (x1, y1) = vals[0], vals[-1]
    x_max, y_max = max(vals, key=lambda v: v[1])
    x_min, y_min = min(vals, key=lambda v: v[1])
    rango = rango_y or (y_max - y_min) or 1.0
    ancho = (x1 - x0) or 1.0
    margen = 0.05 * ancho
    max_interior = (x0 + margen < x_max < x1 - margen) and (y_max - max(y0, y1)) > 0.1 * rango
    min_interior = (x0 + margen < x_min < x1 - margen) and (min(y0, y1) - y_min) > 0.1 * rango
    f = fmt_y
    inicio = f"empieza en {f(y0)} {eje.en(x0)}"
    if max_interior and not min_interior:
        return (f"{inicio}, sube hasta su máximo de {f(y_max)} {eje.en(x_max)} "
                f"y luego baja hasta {f(y1)} {eje.en(x1)}.")
    if min_interior and not max_interior:
        return (f"{inicio}, baja hasta su mínimo de {f(y_min)} {eje.en(x_min)} "
                f"y luego sube hasta {f(y1)} {eje.en(x1)}.")
    if max_interior and min_interior:
        return (f"{inicio} y termina en {f(y1)} {eje.en(x1)}, subiendo y bajando: su máximo "
                f"es {f(y_max)} {eje.en(x_max)} y su mínimo {f(y_min)} {eje.en(x_min)}.")
    return (f"{inicio} y termina en {f(y1)} {eje.en(x1)}; en general "
            f"{_tendencia(y1 - y0, rango)}. Su valor máximo es {f(y_max)} {eje.en(x_max)}, "
            f"y el mínimo es {f(y_min)} {eje.en(x_min)}.")


# ------------------------------------------------------------------
# Descripción en texto
# ------------------------------------------------------------------
def describir_grafica(datos, pie_figura=None, diseno=None):
    """Párrafo en español que describe la gráfica, listo para narrar."""
    textos = datos.get("textos") or {}
    series = [s for s in datos.get("series") or [] if s.get("puntos")]
    rx, ry = _rango_ejes(datos)
    log_x, log_y = _es_log(datos, "x"), _es_log(datos, "y")
    dec_x = _decimales_para(rx[1] - rx[0]) if rx else 2
    dec_y = _decimales_para(ry[1] - ry[0]) if ry else 2
    fmt_x = _fmt_significativo if log_x else _formato_lineal(rx[1] - rx[0] if rx else None)
    fmt_y = _fmt_significativo if log_y else _formato_lineal(ry[1] - ry[0] if ry else None)
    tit_x = (textos.get("titulo_eje_x") or "").strip()
    tit_y = (textos.get("titulo_eje_y") or "").strip()
    cats = _categorias(textos)
    eje = _EjeX(tit_x, fmt_x, cats)

    frases = []
    pie = (pie_figura or "").strip()
    if pie:
        frases.append(pie if pie.endswith((".", "!", "?")) else pie + ".")

    titulo = (textos.get("titulo") or "").strip()
    n = len(series)
    tipo = "Gráfica de líneas" if n <= 1 else f"Gráfica de líneas con {n} curvas"
    frases.append(f"{tipo} titulada «{titulo}»." if titulo else f"{tipo}.")

    for nombre_eje, tit, rango, dec, marcas, log in (
            ("horizontal", tit_x, rx, dec_x, _marcas(textos.get("etiquetas_eje_x")), log_x),
            ("vertical", tit_y, ry, dec_y, _marcas(textos.get("etiquetas_eje_y")), log_y)):
        partes = [f"Eje {nombre_eje}"]
        if tit:
            partes.append(f": {tit}")
        if log:
            partes.append(", en escala logarítmica (cada marca multiplica el valor)")
        if nombre_eje == "horizontal" and cats:
            partes.append(f", con {len(cats)} categorías: " + ", ".join(cats))
        elif marcas:
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
            sujeto += f" (Leyenda {i + 1} en la lámina, textura {texturas[i]})"
        vals = _valores_serie(serie)
        if len(vals) < 2:
            frases.append(f"{sujeto} no tiene escala legible para leer sus valores.")
            continue
        vals.sort()
        frases.append(f"{sujeto} {_forma(vals, (ry[1] - ry[0]) if ry else None, eje, fmt_y)}")

    # Textos que en la lámina van con una letra (no entraban en su lugar).
    abreviaturas = (diseno or {}).get("abreviaturas") or []
    if abreviaturas:
        # agrupadas por lugar: "números del eje X: A es 2015, B es 2016..."
        grupos = {}
        for a in abreviaturas:
            grupos.setdefault(a["donde"] or "texto", []).append(a)
        partes = []
        for donde, lista in grupos.items():
            if donde.startswith(("número del eje", "categoría del eje")):
                plural = donde.replace("número", "números").replace("categoría", "categorías")
                partes.append(plural + ": "
                              + ", ".join(f"{a['identificador']} es {a['texto']}" for a in lista))
            else:
                partes += [f"{a['identificador']}, {donde}: «{a['texto']}»" for a in lista]
        explicacion = "; ".join(partes)
        frases.append("En la lámina, algunos textos van con una letra. "
                      + explicacion[:1].upper() + explicacion[1:] + ".")
    if diseno:
        frases.append("En la lámina, el eje horizontal lleva la letra X y el vertical la letra Y; "
                      "la leyenda de abajo dice qué es cada uno.")
    if (diseno or {}).get("qr"):
        frases.append(TEXTO_QR)
    return " ".join(frases)


# ------------------------------------------------------------------
# Exportación para Hand_Tracking
# ------------------------------------------------------------------
def exportar_hand_tracking(datos, diseno, descripcion):
    """Dict JSON con puntos y tramos narrables ubicados en la placa (mm,
    origen arriba a la izquierda). Ver el docstring del módulo.

    Puntos narrables de cada curva: en un eje de categorías, uno por
    categoría (como "Enero, Febrero..." del ejemplo original de
    Hand_Tracking); en un eje numérico, los cambios de dirección de la curva
    (su polilínea simplificada)."""
    textos = datos.get("textos") or {}
    ancho = float(diseno["placa"]["ancho_mm"])
    alto = float(diseno["placa"]["alto_mm"])
    area, dom = diseno["area"], diseno["dominio"]
    calibrado = dom.get("calibrado")
    rango_x = (dom["x_max"] - dom["x_min"]) or 1.0
    rango_y = (dom["y_max"] - dom["y_min"]) or 1.0
    # el dominio está en el espacio del eje: log10(valor) si la escala es log
    log_x, log_y = dom.get("escala_x") == "log", dom.get("escala_y") == "log"
    dec_x, dec_y = _decimales_para(rango_x), _decimales_para(rango_y)
    fmt_x = _fmt_significativo if log_x else _formato_lineal(rango_x)
    fmt_y = _fmt_significativo if log_y else _formato_lineal(rango_y)
    tit_x = (textos.get("titulo_eje_x") or "").strip() or "x"
    tit_y = (textos.get("titulo_eje_y") or "").strip() or "valor"
    cats = _categorias(textos) if calibrado else []
    varias = len(diseno.get("series") or []) > 1
    series_datos = [s for s in datos.get("series") or []
                    if isinstance(s.get("puntos"), list) and len(s["puntos"]) >= 2]

    def a_hoja(x, y):
        return round(float(x), 1), round(alto - float(y), 1)

    def a_valor(x, y):
        ex = dom["x_min"] + (x - area["izquierda"]) / area["ancho"] * rango_x
        ey = dom["y_min"] + (y - area["abajo"]) / area["alto"] * rango_y
        return (10.0 ** ex if log_x else ex, 10.0 ** ey if log_y else ey)

    def a_placa(vx, vy):
        ex = math.log10(vx) if log_x else vx
        ey = math.log10(vy) if (log_y and vy > 0) else vy
        return (area["izquierda"] + (ex - dom["x_min"]) / rango_x * area["ancho"],
                area["abajo"] + (ey - dom["y_min"]) / rango_y * area["alto"])

    def puntos_clave(s):
        """[(x_mm, y_mm, valor_x, valor_y, nombre_x)] de una curva."""
        if cats and s["indice"] < len(series_datos):
            vals = sorted(_valores_serie(series_datos[s["indice"]]))
            if len(vals) >= 2:
                xs, ys = [v[0] for v in vals], [v[1] for v in vals]
                clave = []
                for i, cat in enumerate(cats):
                    if xs[0] - 0.25 <= i <= xs[-1] + 0.25:
                        vy = float(np.interp(i, xs, ys))
                        clave.append((*a_placa(i, vy), float(i), vy, cat))
                if len(clave) >= 2:
                    return clave
        clave = simplificar_polilinea([tuple(p) for p in s["puntos_mm"]],
                                      tolerancia=PUNTOS_CLAVE_TOLERANCIA_MM,
                                      max_puntos=PUNTOS_CLAVE_MAX)
        resultado = []
        for x, y in clave:
            vx, vy = a_valor(x, y)
            # "Mes 3: ..." si el eje tiene nombre; si no, solo "2015: ..."
            nombre_x = f"{tit_x} {fmt_x(vx)}" if tit_x != "x" else fmt_x(vx)
            resultado.append((x, y, vx, vy, nombre_x))
        return resultado

    puntos, segmentos, resumen_series = [], [], []
    for s in diseno.get("series") or []:
        nombre = s["nombre"]
        prefijo = f"{nombre}. " if varias else ""
        clave = puntos_clave(s)
        resumen_series.append({"nombre": nombre, "textura": s["textura"]})
        anteriores = []
        for k, (x, y, vx, vy, nombre_x) in enumerate(clave):
            hx, hy = a_hoja(x, y)
            if calibrado:
                texto = f"{prefijo}{nombre_x}: {_valor_con_nombre(fmt_y(vy), tit_y)}."
            else:
                texto = f"{prefijo}Punto {k + 1} de {len(clave)} de la curva."
            punto = {"id": f"s{s['indice'] + 1}_p{k + 1}", "serie": nombre,
                     "x": hx, "y": hy, "texto": texto}
            if calibrado:
                punto.update({"valor_x": vx, "valor_y": vy})
            puntos.append(punto)
            anteriores.append((punto, vx, vy, nombre_x))
        for (p1, vx1, vy1, n1), (p2, vx2, vy2, n2) in zip(anteriores[:-1], anteriores[1:]):
            delta = (math.log10(vy2) - math.log10(vy1)) if (log_y and vy1 > 0 and vy2 > 0) else vy2 - vy1
            if abs(delta) <= 0.01 * rango_y:
                tendencia, verbo = "estable", "se mantiene"
            elif delta > 0:
                tendencia, verbo = "aumento", "sube"
            else:
                tendencia, verbo = "disminución", "baja"
            if calibrado and cats:
                texto = f"{prefijo}De {n1} a {n2}, {verbo} de {fmt_y(vy1)} a {fmt_y(vy2)}."
            elif calibrado:
                eje_txt = f"{tit_x} " if tit_x != "x" else ""
                texto = (f"{prefijo}Entre {eje_txt}{fmt_x(vx1)} y {fmt_x(vx2)}, "
                         f"{verbo} de {fmt_y(vy1)} a {fmt_y(vy2)}.")
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
            "texto": f"{entrada.get('etiqueta') or 'Leyenda'}, textura {entrada['textura']}: "
                     f"{entrada['nombre']}.",
        })
    for entrada in diseno.get("leyenda_ejes") or []:
        x0, y0, x1, y1 = entrada["recuadro"]
        (ax, ay), (bx, by) = a_hoja(x0, (y0 + y1) / 2), a_hoja(x1, (y0 + y1) / 2)
        nombre_eje = "horizontal" if entrada["eje"] == "X" else "vertical"
        segmentos.append({
            "id": f"leyenda_eje_{entrada['eje'].lower()}", "tipo": "leyenda", "serie": None,
            "x1": ax, "y1": ay, "x2": bx, "y2": by,
            "texto": f"Leyenda: {entrada['eje']} es el eje {nombre_eje}, {entrada['texto']}.",
        })

    # Textos escritos en la lámina (títulos, categorías, números de los ejes,
    # valores anotados) y los que van con una letra: al tocar la "A" se
    # narra el texto completo; al tocar un número del eje, ese número.
    def _eje(nombre, tit):
        tit = (tit or "").strip()
        return f"Eje {nombre} ({tit})" if tit else f"Eje {nombre}"

    for k, txt in enumerate(diseno.get("textos") or [], start=1):
        x0, y0, x1, y1 = txt["recuadro"]
        (ax, ay), (bx, by) = a_hoja(x0, (y0 + y1) / 2), a_hoja(x1, (y0 + y1) / 2)
        que = {"titulo": "Título", "categoria": "Categoría del eje horizontal",
               "numero_x": _eje("horizontal", textos.get("titulo_eje_x")),
               "numero_y": _eje("vertical", textos.get("titulo_eje_y")),
               "dato": "Valor anotado junto a la curva"}.get(txt["tipo"], "Texto")
        completo = txt["texto_completo"]
        if txt["tipo"] in ("titulo_eje_x", "titulo_eje_y"):
            # en la placa, solo la letra del eje
            nombre_eje = "horizontal" if txt["tipo"] == "titulo_eje_x" else "vertical"
            texto = f"Letra {txt['texto']}: eje {nombre_eje}" + (f", {completo}." if completo else ".")
        elif txt.get("identificador"):
            texto = f"Letra {txt['identificador']}, {que.lower()}: {completo}."
        else:
            texto = f"{que}: {completo}."
        segmentos.append({"id": f"texto_{k}", "tipo": "texto", "serie": None,
                          "x1": ax, "y1": ay, "x2": bx, "y2": by, "texto": texto})
    for a in diseno.get("abreviaturas") or []:
        if not a.get("recuadro"):
            continue
        x0, y0, x1, y1 = a["recuadro"]
        (ax, ay), (bx, by) = a_hoja(x0, y1), a_hoja(x1, y0)
        segmentos.append({
            "id": f"leyenda_{a['identificador']}", "tipo": "leyenda", "serie": None,
            "x1": ax, "y1": ay, "x2": bx, "y2": by,
            "texto": f"Leyenda: {a['identificador']} es {a['texto']}.",
        })

    # El QR en relieve: al tocarlo se dice qué es, para que no se confunda
    # con parte del gráfico. x1,y1 -> x2,y2 = su diagonal (como la leyenda).
    qr = diseno.get("qr")
    if qr:
        (ax, ay) = a_hoja(qr["x"], qr["y"] + qr["lado"])
        (bx, by) = a_hoja(qr["x"] + qr["lado"], qr["y"])
        segmentos.append({"id": "qr", "tipo": "texto", "serie": None,
                          "x1": ax, "y1": ay, "x2": bx, "y2": by, "texto": TEXTO_QR})

    for clave_eje, nombre_eje, tit, d0, d1, dec in (
            ("x", "horizontal", textos.get("titulo_eje_x"), dom["x_min"], dom["x_max"], dec_x),
            ("y", "vertical", textos.get("titulo_eje_y"), dom["y_min"], dom["y_max"], dec_y)):
        (ax, ay), (bx, by) = [a_hoja(*p) for p in diseno["ejes"][clave_eje]]
        texto = f"Eje {nombre_eje}" + (f": {tit.strip()}" if (tit or "").strip() else "")
        marcas = _marcas(textos.get(f"etiquetas_eje_{clave_eje}"))
        if clave_eje == "x" and cats:
            texto += f", de {cats[0]} a {cats[-1]}."
        elif marcas:
            es_log = (clave_eje == "x" and log_x) or (clave_eje == "y" and log_y)
            texto += (", en escala logarítmica" if es_log else "") + \
                f", con marcas de {_fmt(marcas[0], 3)} a {_fmt(marcas[1], 3)}."
        else:
            if calibrado and ((clave_eje == "x" and log_x) or (clave_eje == "y" and log_y)):
                texto += f", en escala logarítmica, de {_fmt_significativo(10 ** d0)} a {_fmt_significativo(10 ** d1)}."
            else:
                texto += f", de {_fmt(d0, dec)} a {_fmt(d1, dec)}." if calibrado else ", sin escala."
        segmentos.append({"id": f"eje_{clave_eje}", "tipo": "eje", "serie": None,
                          "x1": ax, "y1": ay, "x2": bx, "y2": by, "texto": texto})

    return {
        "formato": FORMATO,
        "version": VERSION,
        "descripcion": descripcion,
        "placa": {"ancho_mm": ancho, "alto_mm": alto,
                  "chaflan_mm": diseno["placa"].get("chaflan_mm", 0),
                  "origen": "esquina superior izquierda, y hacia abajo "
                            "(la esquina recortada es la superior derecha)"},
        "series": resumen_series,
        "puntos": puntos,
        "segmentos": segmentos,
    }


# ------------------------------------------------------------------
# Exportación para Hand_Tracking en CSV
# ------------------------------------------------------------------
COLUMNAS_CSV = ["tipo", "id", "serie", "x1_mm", "y1_mm", "x2_mm", "y2_mm",
                "valor_x", "valor_y", "tendencia", "texto"]


def exportar_csv_hand_tracking(exportacion, ruta):
    """Escribe en CSV lo mismo que `exportar_hand_tracking`: todo lo que
    Hand_Tracking necesita para narrar la lámina. Coordenadas en mm sobre la
    placa, con el origen en la esquina superior IZQUIERDA y la y hacia abajo
    (como la imagen de la cámara). Una fila por elemento, columna "tipo":

      placa        x2_mm/y2_mm = ancho y alto de la placa; texto = origen,
                   esquina recortada y distribución.
      descripcion  texto = descripción completa (tecla "d" en Hand_Tracking).
      serie        una por curva: serie = nombre, texto = su textura.
      punto        x1_mm/y1_mm = dónde está; valor_x/valor_y = sus valores;
                   texto = lo que se narra al tocarlo.
      curva        tramo entre dos puntos de una curva (x1,y1 -> x2,y2),
                   con tendencia (aumento / disminución / estable).
      eje          el eje X o el Y, de punta a punta.
      texto        un texto en Braille (título, categoría, número de un eje,
                   valor anotado): x1,y1 -> x2,y2 = su renglón; si va con una
                   letra (A, B...), lo dice.
      leyenda      una entrada de la leyenda de abajo: la textura de una
                   serie o qué significa una letra (x1,y1 -> x2,y2 = su
                   recuadro, de la esquina superior izquierda a la inferior
                   derecha, si ocupa más de un renglón).

    Se guarda en UTF-8 con BOM para que Excel respete las tildes."""
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
        f.write(csv_hand_tracking(exportacion))


def csv_hand_tracking(exportacion):
    """El texto del CSV de `exportar_csv_hand_tracking` (sin BOM)."""
    placa = exportacion["placa"]

    def r(v):
        # valores con 4 decimales (más es ruido de píxeles); vacío si no hay
        return round(v, 4) if isinstance(v, float) else ("" if v is None else v)

    f = io.StringIO(newline="")
    w = csv.writer(f)
    w.writerow(COLUMNAS_CSV)
    w.writerow(["placa", "placa", "", 0, 0, placa["ancho_mm"], placa["alto_mm"], "", "", "",
                f"{placa['origen']}; esquina recortada de {placa.get('chaflan_mm', 0)} mm; "
                "gráfico arriba y leyenda abajo"])
    w.writerow(["descripcion", "descripcion", "", "", "", "", "", "", "", "",
                exportacion.get("descripcion", "")])
    for i, s in enumerate(exportacion.get("series") or [], start=1):
        w.writerow(["serie", f"serie_{i}", s["nombre"], "", "", "", "", "", "", "",
                    f"Textura {s['textura']}"])
    for p in exportacion.get("puntos") or []:
        w.writerow(["punto", p["id"], p.get("serie") or "", p["x"], p["y"], "", "",
                    r(p.get("valor_x")), r(p.get("valor_y")), "", p["texto"]])
    for s in exportacion.get("segmentos") or []:
        w.writerow([s["tipo"], s["id"], s.get("serie") or "", s["x1"], s["y1"], s["x2"], s["y2"],
                    "", "", s.get("tendencia", ""), s["texto"]])
    return f.getvalue()


# ------------------------------------------------------------------
# Datos completos de la lámina (lo que abre su QR)
# ------------------------------------------------------------------
def _num(v):
    """Un valor leído, con 6 cifras significativas (más es ruido de píxeles),
    como float de Python (los de numpy no se pueden escribir en JSON)."""
    if isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v):
        return float(f"{float(v):.6g}")
    return None


def _eje_datos(datos, eje, rango):
    textos = datos.get("textos") or {}
    marcas = [e["valor"] for e in textos.get(f"etiquetas_eje_{eje}") or []
              if e.get("valor") is not None and math.isfinite(e["valor"])]
    return {
        "titulo": (textos.get(f"titulo_eje_{eje}") or "").strip(),
        "escala": "log" if _es_log(datos, eje) else "lineal",
        "min": _num(rango[0]) if rango else None,
        "max": _num(rango[1]) if rango else None,
        # los números que el gráfico original tenía escritos en el eje
        "marcas": sorted(_num(v) for v in marcas),
    }


def nombre_color(serie):
    """Color de la curva en el gráfico original, en palabras ("verde"), a
    partir del tono que midió el segmentador (OpenCV: 0-179). None si no se
    sabe; "negra" si se separó por intensidad (sin color)."""
    if serie.get("modo") == "intensidad":
        return "negra"
    hue = serie.get("hue")
    if not isinstance(hue, numbers.Real):
        return None
    for limite, nombre in ((8, "roja"), (19, "naranja"), (33, "amarilla"), (80, "verde"),
                           (100, "celeste"), (130, "azul"), (150, "morada"), (169, "rosada")):
        if hue <= limite:
            return nombre
    return "roja"


def leyenda_lamina(datos, diseno):
    """La leyenda completa de la lámina, lo que se escucha en el modo
    exploración y en Hand_Tracking: lo impreso abajo ("X- ...", "Y- ...",
    "Leyenda 1"...) con el nombre y el color de cada curva, y lo que en la
    placa va con una letra y NO se imprime ("A es España").

    Devuelve {"titulo", "ejes", "series", "letras", "texto"}; "texto" es todo
    junto, listo para leer en voz alta."""
    textos = datos.get("textos") or {}
    figura = (diseno.get("figura") or "").strip()
    titulo = (textos.get("titulo") or "").strip()
    titulo_completo = ". ".join(t for t in (figura, titulo) if t)
    ejes = [{"letra": letra, "eje": nombre, "texto": (textos.get(clave) or "").strip()}
            for letra, nombre, clave in (("X", "horizontal", "titulo_eje_x"),
                                         ("Y", "vertical", "titulo_eje_y"))]
    datos_series = [s for s in datos.get("series") or [] if s.get("puntos")]
    series = []
    for s in diseno.get("series") or []:
        original = datos_series[s["indice"]] if s["indice"] < len(datos_series) else {}
        series.append({"etiqueta": f"Leyenda {s['indice'] + 1}", "nombre": s["nombre"],
                       "textura": s["textura"], "color": nombre_color(original)})
    letras = [{"letra": a["identificador"], "texto": a["texto"], "donde": a.get("donde") or ""}
              for a in diseno.get("abreviaturas") or []]

    frases = [titulo_completo + "." if titulo_completo else "Lámina táctil."]
    for e in ejes:
        frases.append(f"{e['letra']}: eje {e['eje']}" + (f", {e['texto']}." if e["texto"] else "."))
    if len(series) > 1:
        for s in series:
            color = f", línea {s['color']} en el original" if s["color"] else ""
            frases.append(f"{s['etiqueta']}, textura {s['textura']}: {s['nombre']}{color}.")
    for l in letras:
        frases.append(f"{l['letra']}: {l['texto']}.")
    return {"titulo": titulo_completo, "ejes": ejes, "series": series, "letras": letras,
            "texto": " ".join(frases)}


def exportar_datos_lamina(datos, diseno, descripcion, exportacion, figura=None):
    """Dict con todo lo que se sabe de una lámina (ver el docstring del módulo).

    `figura`: lo que /api/clasificar sacó del PDF para esta figura
    (api._extraccion): "pagina", "pie_figura", "contexto", "documento",
    "origen". Puede faltar (imagen subida suelta por el supervisor).

    Coordenadas de la placa como en exportar_hand_tracking: mm, origen en
    la esquina superior izquierda, y hacia abajo. "lamina.area_grafico_mm" y
    "lamina.dominio" permiten pasar cualquier punto de la placa a valores
    del gráfico (y al revés) sin los puntos narrables."""
    figura = figura or {}
    textos = datos.get("textos") or {}
    rx, ry = _rango_ejes(datos)
    texturas = {s["indice"]: s["textura"] for s in diseno.get("series") or []}
    series = []
    for i, serie in enumerate(s for s in datos.get("series") or [] if s.get("puntos")):
        vals = sorted(_valores_serie(serie))
        if len(vals) > VALORES_SERIE_MAX:
            paso = (len(vals) - 1) / (VALORES_SERIE_MAX - 1)
            vals = [vals[round(k * paso)] for k in range(VALORES_SERIE_MAX)]
        entrada = {"nombre": nombre_de_serie(serie.get("nombre"), i),
                   "textura": texturas.get(i),
                   "valores": [[_num(x), _num(y)] for x, y in vals]}
        if vals:
            x_max, y_max = max(vals, key=lambda v: v[1])
            x_min, y_min = min(vals, key=lambda v: v[1])
            entrada["maximo"] = {"x": _num(x_max), "y": _num(y_max)}
            entrada["minimo"] = {"x": _num(x_min), "y": _num(y_min)}
        series.append(entrada)

    alto_placa = float(diseno["placa"]["alto_mm"])
    area = diseno["area"]
    dom = diseno["dominio"]
    documento = dict(figura.get("documento") or {})
    return {
        "formato": FORMATO_DATOS,
        "version": VERSION_DATOS,
        "generado_en": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "documento": documento,
        "figura": {
            "pagina": figura.get("pagina"),
            "pie_figura": (figura.get("pie_figura") or "").strip(),
            "origen": figura.get("origen"),
        },
        "contexto": figura.get("contexto") or {"antes": "", "despues": "", "menciones": []},
        "descripcion": descripcion,
        # lo que se escucha en el modo exploración (ver leyenda_lamina)
        "leyenda": leyenda_lamina(datos, diseno),
        "grafica": {
            "titulo": (textos.get("titulo") or "").strip(),
            "calibrada": bool(dom.get("calibrado")),
            "eje_x": {**_eje_datos(datos, "x", rx), "categorias": _categorias(textos)},
            "eje_y": _eje_datos(datos, "y", ry),
            "series": series,
        },
        "lamina": {
            **exportacion,
            # recuadro del gráfico en la placa y los valores en sus bordes
            # (en escala log, el dominio está en log10 del valor)
            "area_grafico_mm": {
                "izquierda": round(float(area["izquierda"]), 2),
                "arriba": round(alto_placa - float(area["abajo"] + area["alto"]), 2),
                "ancho": round(float(area["ancho"]), 2),
                "alto": round(float(area["alto"]), 2),
            },
            "dominio": {k: (bool(v) if k == "calibrado" else _num(v) if k.endswith(("_min", "_max")) else v)
                        for k, v in dom.items()},
        },
    }


# ------------------------------------------------------------------
# Audio del modo exploración
# ------------------------------------------------------------------
# Lo que se escucha al escanear el QR de la lámina: todo lo que la lámina no
# puede mostrar en relieve, en cuatro partes que se pueden repetir por
# separado (botones en la app y en la página del QR):
#   1. Qué es esta gráfica: pie de figura, documento y página.
#   2. Leyenda: ejes, curvas (con su color en el original) y las letras A, B...
#   3. Qué muestra: forma de cada curva, máximos y mínimos.
#   4. Qué dice el documento: los párrafos que citan la figura.
# Se arma con los datos ya guardados de la lámina (exportar_datos_lamina), así
# que sirve también para las láminas generadas antes de este cambio.
#
# El texto leído de imágenes (OCR) a veces sale roto ("Tim e(h(;"): para quien
# no ve la lámina, escucharlo confunde más de lo que ayuda, así que lo que no
# parece texto legible no se lee (_legible).

AUDIO_CONTEXTO_MAX = 700        # caracteres del documento que se leen
_RE_PALABRA = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][a-záéíóúüñ'’]+[.,;:!?]?|[A-ZÁÉÍÓÚÑ]{2,}[.,;:]?")
_RE_NUMERO = re.compile(r"[-−+]?\d+([.,]\d+)*%?[.,;:]?|\d+%?[A-Za-z]{1,4}")
_RE_UNIDAD = re.compile(r"\([^()\s]{1,12}\)[.,;:]?")
_SUELTAS_OK = {"a", "e", "o", "y", "u", "x", "i"}
_CORTAS_OK = {"de", "la", "el", "en", "un", "of", "in", "to", "on", "by", "at", "an", "vs", "per", "and", "the"}
_ENCABEZADOS = {"introduction", "introducción", "abstract", "resumen", "contents", "índice",
                "references", "referencias", "conclusion", "conclusiones", "methods", "métodos"}


def _normalizar(texto):
    """Ligaduras del PDF (ﬁ -> fi), guiones de corte de línea ("man- agement")
    y espacios repetidos."""
    texto = unicodedata.normalize("NFKC", str(texto or "")).replace("\u200b", "").replace("\xad", "")
    texto = re.sub(r"([a-záéíóúñ])- ([a-záéíóúñ])", r"\1\2", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _limpiar(texto):
    """Quita restos de OCR en los bordes ("gu e Sophie's..." -> "Sophie's...",
    "swelling (%) >" -> "swelling (%)") y repeticiones ("Time (min) Time (min)")."""
    texto = _normalizar(texto).strip(" -—–_|•■●▪*>~=.,;:")
    palabras = texto.split()
    while len(palabras) > 2 and len(palabras[0]) <= 2 and palabras[0].lower() not in _CORTAS_OK \
            and palabras[0].lower() not in _SUELTAS_OK and palabras[0].islower():
        palabras.pop(0)
    while len(palabras) > 2 and len(palabras[0]) == 1 and palabras[0].islower():
        palabras.pop(0)
    while len(palabras) > 1 and palabras[0] in _CORTAS_OK:     # "of SPEED" -> "SPEED"
        palabras.pop(0)
    mitad = len(palabras) // 2
    if len(palabras) % 2 == 0 and mitad and palabras[:mitad] == palabras[mitad:]:
        palabras = palabras[:mitad]
    return " ".join(palabras)


def _legible(texto, corto_ok=True):
    """¿Parece texto de verdad y no un resto de OCR? Un nombre corto (r, Ω,
    pH) vale; "OO OOO", "Tim e(h(;" o "Condition Condition Condition" no.
    corto_ok="mayuscula": una sola letra solo si es mayúscula o griega (el
    nombre de una curva "e" suele ser un resto de OCR; el de un eje "r", no)."""
    texto = (texto or "").strip()
    if not texto:
        return False
    if corto_ok == "mayuscula" and len(texto) == 1 and texto.islower() and texto.isascii():
        return False
    if len(texto) <= 3 and " " not in texto:
        return corto_ok and (texto.isascii() and texto.isalnum() or
                             all(unicodedata.category(c).startswith("L") and not c.isascii()
                                 and unicodedata.name(c, "").startswith("GREEK") for c in texto))
    palabras = texto.split()
    buenas = 0
    for w in palabras:
        if len(w) == 1:
            buenas += w.lower() in _SUELTAS_OK and len(palabras) <= 6
        elif len(set(w.lower())) == 1 and w.isalpha():
            continue                        # "OO", "ooo"
        elif _RE_PALABRA.fullmatch(w) or _RE_NUMERO.fullmatch(w) or _RE_UNIDAD.fullmatch(w):
            buenas += 1
        elif "-" in w and all(_RE_PALABRA.fullmatch(p) or _RE_NUMERO.fullmatch(p) for p in w.split("-") if p):
            buenas += 1                     # "HA-g-pHEA", "PEGMA-PEGDA"
    if buenas / len(palabras) < 0.75:
        return False
    # la misma palabra una y otra vez: varias etiquetas pegadas por el OCR
    mas_repetida = max(palabras.count(w) for w in palabras)
    return not (len(palabras) >= 4 and mas_repetida / len(palabras) > 0.3)


def _texto_legible(texto, corto_ok=True):
    """El texto limpio si es legible; "" si no."""
    limpio = _limpiar(texto)
    return limpio if _legible(limpio, corto_ok) else ""


def _para_leer(texto):
    """Ajustes para la voz: "Fig. 2" se lee "Figura 2"."""
    texto = re.sub(r"\bFigs?\.\s*(?=\d)", "Figura ", texto)
    return re.sub(r"\bFig\s+(?=\d)", "Figura ", texto)


def _oracion(texto):
    texto = texto.strip()
    return texto if not texto or texto.endswith((".", "!", "?", "…")) else texto + "."


def _parrafo_de_texto(texto):
    """Un párrafo del documento y no un encabezado de página, una referencia
    o una fila de tabla: mayoría de palabras comunes en minúscula."""
    palabras = texto.split()
    if len(palabras) < 15:          # encabezados ("3.3. Laponite nanocomposite...")
        return False
    comunes = sum(1 for w in palabras if re.fullmatch(r"[a-záéíóúñ]{3,}[.,;:]?", w))
    return comunes / len(palabras) >= 0.45


def _audio_que_es(d):
    figura, doc = d.get("figura") or {}, d.get("documento") or {}
    frases = []
    pie = _normalizar(figura.get("pie_figura"))
    titulo = _texto_legible((d.get("grafica") or {}).get("titulo"), corto_ok=False)
    if pie:
        frases.append(_oracion(_para_leer(pie)))
        if titulo and titulo.lower() not in pie.lower():
            frases.append(f"Título de la gráfica: {_oracion(titulo)}")
    elif titulo:
        frases.append(f"Gráfica titulada: {_oracion(titulo)}")
    else:
        frases.append("Gráfica de líneas sin pie de figura.")
    titulo_doc = _limpiar(re.sub(r"^(research|original|review)\s+article\s+", "", _normalizar(doc.get("titulo")),
                                 flags=re.IGNORECASE))
    if not _legible(titulo_doc, corto_ok=False) or titulo_doc.lower() in _ENCABEZADOS:
        titulo_doc = ""
    archivo = re.sub(r"\.pdf$", "", _normalizar(doc.get("archivo")), flags=re.IGNORECASE)
    pagina, paginas = figura.get("pagina"), doc.get("paginas")
    donde = f"en la página {pagina}" + (f" de {paginas}" if paginas else "") if pagina else ""
    if titulo_doc:
        frases.append(f"Es del documento «{titulo_doc}»" + (f", {donde}." if donde else "."))
    elif archivo:
        frases.append(f"Es del archivo {archivo}" + (f", {donde}." if donde else "."))
    elif donde:
        frases.append(f"Está {donde} del documento.")
    return " ".join(frases)


def _nombre_curva(nombre):
    """Nombre legible de una curva, o "" (también para los genéricos "Serie A")."""
    nombre = _texto_legible(nombre, corto_ok="mayuscula")
    return "" if re.fullmatch(r"Serie [A-Z]", nombre) else nombre


def _audio_eje(info, letra):
    nombre = "Eje horizontal, letra X" if letra == "X" else "Eje vertical, letra Y"
    titulo = _texto_legible(info.get("titulo"))
    partes = [nombre + (f": {titulo}" if titulo else ", sin título legible")]
    if info.get("escala") == "log":
        partes.append("en escala logarítmica")
    cats = [c for c in info.get("categorias") or [] if c]
    marcas = info.get("marcas") or []
    if cats:
        partes.append(f"con {len(cats)} categorías, de {cats[0]} a {cats[-1]}" if len(cats) > 1
                      else f"con la categoría {cats[0]}")
    elif len(marcas) >= 2:
        fmt = _fmt_significativo if info.get("escala") == "log" else (lambda v: _fmt(v, 3))
        partes.append(f"con marcas de {fmt(marcas[0])} a {fmt(marcas[-1])}")
    return ", ".join(partes) + "."


def _audio_leyenda(d):
    ley, graf = d.get("leyenda") or {}, d.get("grafica") or {}
    frases = [_audio_eje(graf.get("eje_x") or {}, "X"), _audio_eje(graf.get("eje_y") or {}, "Y")]
    series = ley.get("series") or []
    if len(series) == 1:
        s = series[0]
        nombre = _nombre_curva(s.get("nombre"))
        frases.append("Hay una sola curva" + (f", «{nombre}»" if nombre else "")
                      + (f"; en el original es una línea {s['color']}" if s.get("color") else "") + ".")
    elif series:
        frases.append(f"Hay {len(series)} curvas, cada una con su textura.")
        for s in series:
            nombre = _nombre_curva(s.get("nombre")) or "sin nombre legible"
            frases.append(f"{s['etiqueta']}, textura {s['textura']}: {nombre}"
                          + (f", línea {s['color']} en el original" if s.get("color") else "") + ".")
    grupos = {}
    for l in ley.get("letras") or []:
        grupos.setdefault(l.get("donde") or "", []).append(l)
    for donde, lista in grupos.items():
        if donde.startswith(("número del eje", "categoría del eje")):
            plural = donde.replace("número", "Números").replace("categoría", "Categorías")
            frases.append(f"{plural}, que en la lámina van con letras: "
                          + ", ".join(f"{l['letra']} es {l['texto']}" for l in lista) + ".")
        else:
            for l in lista:
                texto = _texto_legible(l.get("texto")) or "un texto que no se pudo leer bien"
                frases.append(f"Letra {l['letra']}" + (f", {donde}" if donde else "") + f": {_oracion(texto)}")
    return " ".join(frases)


def _audio_que_muestra(d):
    graf = d.get("grafica") or {}
    ex, ey = graf.get("eje_x") or {}, graf.get("eje_y") or {}
    log_y = ey.get("escala") == "log"
    rango_y = (ey["max"] - ey["min"]) if ey.get("max") is not None and ey.get("min") is not None else None
    # decimales según cuánto varían los valores, no solo el rango del eje
    # (una curva entre 9,5 y 10,2 en un eje de 0 a 20 no es "de 10 a 10")
    ys = [y for s in graf.get("series") or [] for _, y in s.get("valores") or [] if y is not None]
    variacion = (max(ys) - min(ys)) if len(ys) >= 2 else None
    rangos = [r for r in (rango_y, variacion) if r]
    fmt_y = _fmt_significativo if log_y else _formato_lineal(min(rangos) if rangos else None)
    fmt_x = _fmt_significativo if ex.get("escala") == "log" else _formato_lineal(
        (ex["max"] - ex["min"]) if ex.get("max") is not None and ex.get("min") is not None else None)
    eje = _EjeX(_texto_legible(ex.get("titulo")) or "el eje horizontal", fmt_x, ex.get("categorias") or [])
    series = graf.get("series") or []
    frases = []
    if not graf.get("calibrada"):
        frases.append("La escala de la gráfica no se pudo leer bien: los valores son aproximados.")
    for i, s in enumerate(series):
        vals = sorted((x, y) for x, y in s.get("valores") or [] if x is not None and y is not None)
        nombre = _nombre_curva(s.get("nombre"))
        sujeto = "La curva" if len(series) == 1 else f"La curva {i + 1}" + (f", «{nombre}»," if nombre else "")
        if len(vals) < 2:
            frases.append(f"{sujeto} no tiene una escala legible para leer sus valores.")
            continue
        frases.append(f"{sujeto} {_forma(vals, rango_y, eje, fmt_y)}")
    if not series:
        frases.append("No se pudieron leer los valores de la gráfica.")
    texto = " ".join(frases)
    # "cuando el eje horizontal vale 5" se entiende; "cuando x vale 5", también
    return texto.replace("cuando el eje horizontal vale", "cuando el eje horizontal está en")


_RE_CITA_FIGURA = re.compile(r"\bfig(?:ura|ure)?s?\.?\s*(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+))?", re.IGNORECASE)


def _cita_otra_figura(texto, numero):
    """True si el párrafo cita figuras y ninguna es `numero` ("5.3"): la
    búsqueda de menciones a veces trae el párrafo de la figura 5.1 para la 5.3."""
    if not numero:
        return False
    citadas = set()
    for desde, hasta in _RE_CITA_FIGURA.findall(texto):
        citadas.add(desde)
        if hasta and desde.isdigit() and int(hasta) - int(desde) < 20:
            citadas.update(str(n) for n in range(int(desde), int(hasta) + 1))
    return bool(citadas) and numero not in citadas


def _audio_documento(d):
    ctx = d.get("contexto") or {}
    m_num = _RE_CITA_FIGURA.match(_normalizar((d.get("figura") or {}).get("pie_figura")))
    numero = m_num.group(1) if m_num else None
    partes, total = [], 0
    for m in ctx.get("menciones") or []:
        texto = _normalizar(m.get("texto"))
        if not _parrafo_de_texto(texto) or _cita_otra_figura(texto, numero):
            continue
        if texto[:1].islower():
            texto = "…" + texto         # empieza a mitad de frase (cambio de columna o de página)
        if total + len(texto) > AUDIO_CONTEXTO_MAX:
            texto = texto[:max(0, AUDIO_CONTEXTO_MAX - total)].rsplit(" ", 1)[0].rstrip(",;:") + "…"
        pag = m.get("pagina")
        partes.append((f"En la página {pag}, el documento dice: " if pag else "El documento dice: ") + texto)
        total += len(texto)
        if total >= AUDIO_CONTEXTO_MAX:
            break
    if not partes:
        for clave, intro in (("antes", "Antes de la figura, el documento dice: "),
                             ("despues", "Después de la figura, el documento dice: ")):
            texto = _normalizar(ctx.get(clave))
            if _parrafo_de_texto(texto):
                partes.append(intro + (texto[:AUDIO_CONTEXTO_MAX].rsplit(" ", 1)[0] + "…"
                                       if len(texto) > AUDIO_CONTEXTO_MAX else texto))
                break
    if not partes:
        return "No se encontró en el documento un párrafo que hable de esta figura."
    return " ".join(_oracion(p) for p in partes)


def audio_exploracion(datos):
    """El audio del modo exploración de una lámina, a partir de sus datos
    (exportar_datos_lamina). Devuelve {"secciones": [{"id", "titulo",
    "texto"}], "texto": todo junto, con el nombre de cada parte}."""
    secciones = [
        {"id": "que_es", "titulo": "Qué es esta gráfica", "texto": _audio_que_es(datos)},
        {"id": "leyenda", "titulo": "Leyenda", "texto": _audio_leyenda(datos)},
        {"id": "que_muestra", "titulo": "Qué muestra", "texto": _audio_que_muestra(datos)},
        {"id": "documento", "titulo": "Qué dice el documento", "texto": _audio_documento(datos)},
    ]
    texto = " ".join([secciones[0]["texto"]] + [f"{s['titulo']}. {s['texto']}" for s in secciones[1:]])
    return {"secciones": secciones, "texto": texto}

