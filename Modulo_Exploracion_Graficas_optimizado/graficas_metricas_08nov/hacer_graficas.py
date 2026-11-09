"""Gráficas del informe / pitch a partir de la carpeta metricas (medir_metricas.py 1.2)."""
import csv
import collections
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle

MET = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# ---- estilo (paleta de referencia validada; ver dataviz/references/palette.md) ----
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = {"morado_g2": "#2a78d6", "salmon_g2": "#eb6834", "turquesa_g3": "#1baf7a"}   # slots 1-3
ANTES = "#c3c2b7"                     # de-énfasis
GOOD, CRIT = "#0ca30c", "#d03b3b"
NOMBRE = {"morado_g2": "Morada · gráfica 2", "salmon_g2": "Salmón · gráfica 2", "turquesa_g3": "Turquesa · gráfica 3"}
PLACAS = ["morado_g2", "salmon_g2", "turquesa_g3"]

for f in font_manager.findSystemFonts():
    if "Inter" in os.path.basename(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Inter", "Segoe UI", "Arial", "DejaVu Sans"], "font.size": 13, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 12,
})
W, H = 13.333, 7.5          # 16:9


def coma(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


def cargar(placa, prueba):
    f = sorted(glob.glob(os.path.join(MET, placa, f"*_{prueba}.json")))
    return json.load(open(f[-1], encoding="utf-8")) if f else None


def figura(titulo, subtitulo, fuente="Fuente: medir_metricas.py 1.2 · sesiones del 8 nov 2026 · cámara WN CAM-K211L"):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.055, 0.93, titulo, fontsize=24, fontweight="semibold", color=INK, va="top")
    fig.text(0.055, 0.865, subtitulo, fontsize=14, color=INK2, va="top")
    fig.text(0.055, 0.035, fuente, fontsize=10.5, color=MUTED)
    return fig


def guardar(fig, nombre):
    fig.savefig(os.path.join(OUT, nombre + ".png"), dpi=150)
    plt.close(fig)
    print("ok", nombre)


def eje_limpio(ax, grid_x=False, grid_y=True):
    ax.tick_params(length=0, pad=8)
    if grid_y:
        ax.yaxis.grid(True, color=GRID, linewidth=1)
    if grid_x:
        ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def barra_h(ax, y, x0, x1, alto, color):
    """Barra horizontal con el extremo de datos redondeado (4 px) y la base recta."""
    if x1 - x0 <= 0:
        return
    ax.add_patch(FancyBboxPatch((x0, y - alto / 2), x1 - x0, alto, boxstyle="round,pad=0,rounding_size=0.0",
                                facecolor=color, edgecolor="none"))


# =====================================================================
# 1. Panorama: criterios cumplidos por grupo (las 3 placas)
# =====================================================================
filas = list(csv.DictReader(open(os.path.join(MET, "resumen_metricas.csv"), encoding="utf-8-sig")))
cnt = collections.OrderedDict()
for r in filas:
    if r["cumple"] in ("sí", "no"):
        g = cnt.setdefault(r["grupo"], [0, 0])
        g[1] += 1
        g[0] += r["cumple"] == "sí"
nombres_g = {"Pinza": "Pinza (gesto)", "Placa": "Detección de la placa", "Registro": "Calibración (registro)",
             "Mano": "Seguimiento de la mano", "Geometría": "Geometría de la lámina", "QR": "Lectura del QR",
             "Selección": "Selección del elemento"}
orden = sorted(cnt.items(), key=lambda kv: kv[1][0] / kv[1][1])
fig = figura("La pinza y la detección de la placa ya cumplen; falta acertar el elemento",
             f"Criterios cumplidos por grupo, sumando las 3 placas · {sum(v[0] for v in cnt.values())} de "
             f"{sum(v[1] for v in cnt.values())} criterios medidos")
ax = fig.add_axes([0.25, 0.17, 0.58, 0.63])
for i, (g, (a, b)) in enumerate(orden):
    pct = 100 * a / b
    ax.barh(i, pct, height=0.42, color=C["morado_g2"], edgecolor="none")
    ax.barh(i, 100 - pct, left=pct, height=0.42, color="#e8eef8", edgecolor="none")
    ax.text(101.5, i, f"{coma(pct, 0)} %", va="center", fontsize=14, fontweight="semibold", color=INK)
    ax.text(111, i, f"{a} de {b}", va="center", fontsize=12, color=INK2)
ax.set_yticks(range(len(orden)), [nombres_g.get(g, g) for g, _ in orden], fontsize=14, color=INK)
ax.set_xlim(0, 100)
ax.set_xticks([])
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
eje_limpio(ax, grid_x=False, grid_y=False)
fig.text(0.055, 0.09, "En «Seguimiento de la mano» solo fallan los fps (15, meta 25); la detección es del 100 %.",
         fontsize=11.5, color=INK2)
guardar(fig, "01_panorama_criterios")

# =====================================================================
# 2. Pinza: umbral anterior vs pinza actual (cortes por minuto)
# =====================================================================
fal = {p: cargar(p, "falsos")["metricas"] for p in PLACAS}
fig = figura("La nueva pinza eliminó los cortes del audio: de 28–52 por minuto a 0",
             "Cortes con la pinza cerrada, por minuto · misma grabación evaluada con el umbral anterior "
             "(distancia fija en px) y con la pinza actual")
ax = fig.add_axes([0.08, 0.2, 0.86, 0.56])
x = range(len(PLACAS))
bw = 0.26
for i, p in enumerate(PLACAS):
    a, n = fal[p]["cortes_px_min"], fal[p]["cortes_min"]
    ax.bar(i - bw / 2 - 0.01, a, width=bw, color=ANTES, edgecolor="none")
    ax.bar(i + bw / 2 + 0.01, max(n, 0.4), width=bw, color=C["morado_g2"], edgecolor="none")
    ax.text(i - bw / 2 - 0.01, a + 1.2, coma(a, 0), ha="center", fontsize=15, fontweight="semibold", color=INK)
    ax.text(i + bw / 2 + 0.01, n + 1.2, coma(n, 0), ha="center", fontsize=15, fontweight="semibold", color=INK)
ax.set_xticks(list(x), [NOMBRE[p] for p in PLACAS], fontsize=13, color=INK)
ax.set_ylim(0, 60)
ax.set_ylabel("Cortes por minuto")
eje_limpio(ax)
ax.spines["left"].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ANTES, label="Umbral anterior (px fijo)"),
                   Patch(color=C["morado_g2"], label="Pinza actual (palma + histéresis + 250 ms de gracia)")],
          loc="upper right", ncol=1, fontsize=12.5)
sens_a = [fal[p]["sensibilidad_px_pct"] for p in PLACAS]
fig.text(0.055, 0.09, f"Sensibilidad con la pinza mantenida: {coma(min(sens_a), 0)}–{coma(max(sens_a), 0)} % antes → 100 % ahora. "
         "Activaciones falsas con los dedos separados: 0 por minuto en las 3 placas.", fontsize=11.5, color=INK2)
guardar(fig, "02_pinza_antes_despues")

# =====================================================================
# 3. Registro: calibración guardada vs autodetección
# =====================================================================
reg = {p: cargar(p, "registro")["metricas"] for p in PLACAS}
fig = figura("Autodetectar la placa baja el error de calibración de ~10 mm a 2 mm o menos",
             "Error de ubicación sobre el gráfico (mediana, mm), medido con clics en 10–12 puntos de referencia")
ax = fig.add_axes([0.08, 0.2, 0.86, 0.56])
for i, p in enumerate(PLACAS):
    a, n = reg[p]["error_mediana_mm"], reg[p]["error_auto_mediana_grafico_mm"]
    ax.bar(i - bw / 2 - 0.01, a, width=bw, color=ANTES, edgecolor="none")
    ax.bar(i + bw / 2 + 0.01, n, width=bw, color=C["morado_g2"], edgecolor="none")
    ax.text(i - bw / 2 - 0.01, a + 0.25, coma(a), ha="center", fontsize=15, fontweight="semibold", color=INK)
    ax.text(i + bw / 2 + 0.01, n + 0.25, coma(n), ha="center", fontsize=15, fontweight="semibold", color=INK)
ax.axhline(3, color=INK2, linewidth=1)
ax.text(2.45, 3.25, "Meta ≤ 3 mm", fontsize=12, color=INK2, ha="right")
ax.set_xticks(list(range(3)), [NOMBRE[p] for p in PLACAS], fontsize=13, color=INK)
ax.set_ylim(0, 12.5)
ax.set_ylabel("Error mediano (mm)")
eje_limpio(ax)
ax.spines["left"].set_visible(False)
ax.legend(handles=[Patch(color=ANTES, label="Calibración guardada (de otra colocación)"),
                   Patch(color=C["morado_g2"], label="Autodetección al empezar (tecla A)")], loc="upper left")
fig.text(0.055, 0.09, "La calibración guardada se hizo con la placa morada: con otra placa queda ~10 mm desplazada. "
         "La autodetección se ajusta a cada colocación.", fontsize=11.5, color=INK2)
guardar(fig, "03_calibracion_autodeteccion")

# =====================================================================
# 4. Yema: desplazamiento de cada intento (dispersión) + meta 4 mm
# =====================================================================
obj = {p: cargar(p, "objetivos") for p in PLACAS}
fig = figura("La yema cae desplazada de forma sistemática, no al azar",
             "Desplazamiento de la yema respecto del objetivo en cada intento (mm de la placa) · "
             "círculo = meta de 4 mm")
axs = [fig.add_axes([0.06 + i * 0.31, 0.17, 0.27, 0.6]) for i in range(3)]
for ax, p in zip(axs, PLACAS):
    m = obj[p]["metricas"]
    pts = [t["sesgo_mm"] for t in obj[p]["detalle"] if t.get("sesgo_mm")]
    ax.add_patch(Circle((0, 0), 4, facecolor=C["morado_g2"], alpha=0.06, edgecolor=INK2, linewidth=1))
    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=GRID, linewidth=1, zorder=0)
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    ax.scatter(xs, ys, s=70, color=C[p], edgecolor=SURF, linewidth=2, zorder=3)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    ax.scatter([mx], [my], s=260, marker="X", color=INK, edgecolor=SURF, linewidth=2, zorder=4)
    ax.set_xlim(-14, 14)
    ax.set_ylim(14, -14)          # y hacia abajo, como en la placa
    ax.set_aspect("equal")
    ax.set_xticks([-10, -5, 0, 5, 10], ["10", "5", "0", "5", "10"])
    ax.set_yticks([-10, -5, 0, 5, 10], ["10", "5", "0", "5", "10"])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(NOMBRE[p], fontsize=15, fontweight="semibold", color=INK, pad=24)
    ax.text(0, 1.01, f"n = {len(pts)} · error mediano {coma(m['error_mediana_mm'])} mm · sesgo {coma(m['sesgo_modulo_mm'])} mm (X)",
            transform=ax.transAxes, fontsize=11, color=INK2, va="bottom")
axs[0].set_ylabel("↓ abajo · arriba ↑ (mm)")
axs[1].set_xlabel("← izquierda · derecha → (mm)")
fig.text(0.055, 0.09, "Salmón y turquesa se midieron con la calibración guardada (desfasada ~10 mm). En la morada, bien calibrada,\n"
         "queda un sesgo de ~5 mm: el punto del dedo que sigue la cámara no es la yema que toca la placa.",
         fontsize=11.5, color=INK2)
guardar(fig, "04_yema_desplazamiento")

# =====================================================================
# 5. Acierto de selección con IC95 (elemento exacto y serie)
# =====================================================================
fig = figura("Hoy se acierta el elemento exacto en 1 de cada 4 intentos (meta: 9 de 10)",
             "Acierto por placa con intervalo de confianza del 95 % (Wilson) · prueba «objetivos», 8–15 intentos por placa")
paneles = [("acierto_estricto", "Elemento exacto", 90), ("acierto_serie", "Serie correcta", 95)]
for j, (clave, nombre, meta) in enumerate(paneles):
    ax = fig.add_axes([0.20 + j * 0.42, 0.2, 0.34, 0.54])
    for i, p in enumerate(PLACAS):
        d = obj[p]["detalle"]
        k, n = sum(bool(t[clave]) for t in d), len(d)
        lo, hi = wilson(k, n)
        y = len(PLACAS) - 1 - i
        ax.plot([100 * lo, 100 * hi], [y, y], color=C[p], linewidth=2, solid_capstyle="round")
        ax.scatter([100 * k / n], [y], s=110, color=C[p], edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(100 * k / n, y + 0.22, f"{coma(100 * k / n, 0)} %", ha="center", fontsize=13,
                fontweight="semibold", color=INK)
    ax.axvline(meta, color=INK2, linewidth=1)
    ax.text(meta - 1.5, 2.55, f"Meta {meta} %", ha="right", fontsize=12, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 2.8)
    ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_yticks(range(3), [NOMBRE[p] for p in reversed(PLACAS)] if j == 0 else [""] * 3, fontsize=13, color=INK)
    ax.set_title(nombre, fontsize=15, fontweight="semibold", color=INK, pad=12)
    eje_limpio(ax, grid_x=True, grid_y=False)
    ax.spines["left"].set_visible(False)
fig.text(0.055, 0.09, "Causa principal: el desplazamiento de la yema (gráfica 4). Con autodetección y corrección del punto del dedo,\n"
         "la geometría de la lámina permite ~90 % (gráfica 6).", fontsize=11.5, color=INK2)
guardar(fig, "05_acierto_seleccion")

# =====================================================================
# 6. Geometría: acierto esperado según el error de la yema + error medido
# =====================================================================
geo = {"Gráfica 2 (3 curvas que se cruzan)": (cargar("morado_g2", "geometria")["metricas"], C["morado_g2"]),
       "Gráfica 3 (1 curva)": (cargar("turquesa_g3", "geometria")["metricas"], C["turquesa_g3"])}
fig = figura("Con la yema a 2 mm del objetivo se acertaría ~90 %; hoy cae a 5–9 mm",
             "Acierto esperado en puntos y líneas aislados según el error de la yema · simulación sobre el CSV")
ax = fig.add_axes([0.08, 0.2, 0.62, 0.56])
for nombre, (m, col) in geo.items():
    xs = [2, 4, 6]
    ys = [m[f"esperado_{s}mm_aislado_pct"] for s in xs]
    ax.plot(xs, ys, color=col, linewidth=2, solid_capstyle="round", marker="o", markersize=9,
            markeredgecolor=SURF, markeredgewidth=2)
    ax.text(6.25, ys[-1], nombre, va="center", fontsize=12.5, color=INK)
    ax.text(2, ys[0] + 3.5, f"{coma(ys[0], 0)} %", ha="center", fontsize=13, fontweight="semibold", color=INK)
ax.axhline(90, color=INK2, linewidth=1)
ax.text(9.9, 91.5, "Meta 90 %", ha="right", fontsize=12, color=INK2)
for p in PLACAS:
    e = obj[p]["metricas"]["error_mediana_mm"]
    ax.scatter([e], [3], marker="v", s=150, color=C[p], edgecolor=SURF, linewidth=2, zorder=4, clip_on=False)
ax.text(5.3, 9, "error medido", fontsize=11.5, color=INK2, ha="center")
ax.set_xlim(1, 10)
ax.set_ylim(0, 105)
ax.set_xticks([2, 4, 6, 8, 10], ["2 mm", "4 mm", "6 mm", "8 mm", "10 mm"])
ax.set_yticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
ax.set_xlabel("Error típico de la yema (σ)")
eje_limpio(ax)
ax.spines["left"].set_visible(False)
leg = [plt.Line2D([], [], marker="v", linestyle="", markersize=11, color=C[p], label=f"{NOMBRE[p]}: "
                  f"{coma(obj[p]['metricas']['error_mediana_mm'])} mm") for p in PLACAS]
fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.76, 0.62), title="Error mediano medido",
           title_fontproperties={"weight": "semibold", "size": 12.5}, alignment="left")
guardar(fig, "06_geometria_vs_error")

# =====================================================================
# 7. QR: lectura por placa y condición con IC95
# =====================================================================
qr = {p: cargar(p, "qr")["metricas"] for p in PLACAS}
fig = figura("El QR actual casi no se lee con la cámara del módulo (meta: 80 %)",
             "Lecturas por intento, con intervalo de confianza del 95 % · QR de 25,8 mm con el enlace completo")
conds = ["en la bandeja", "en la mano a unos 20 cm"]
for j, cond in enumerate(conds):
    ax = fig.add_axes([0.20 + j * 0.42, 0.2, 0.34, 0.54])
    for i, p in enumerate(PLACAS):
        c = next(c for c in qr[p]["condiciones"] if c["condicion"] == cond)
        k, n = c["lecturas"], c["intentos"]
        lo, hi = wilson(k, n)
        y = len(PLACAS) - 1 - i
        ax.plot([100 * lo, 100 * hi], [y, y], color=C[p], linewidth=2, solid_capstyle="round")
        ax.scatter([100 * k / n], [y], s=110, color=C[p], edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(100 * hi + 2, y, f"{coma(100 * k / n, 0)} %  ({k}/{n})", va="center", fontsize=12.5, color=INK)
    ax.axvline(80, color=INK2, linewidth=1)
    ax.text(78.5, 2.55, "Meta 80 %", ha="right", fontsize=12, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 2.8)
    ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_yticks(range(3), [NOMBRE[p] for p in reversed(PLACAS)] if j == 0 else [""] * 3, fontsize=13, color=INK)
    ax.set_title("Placa " + cond, fontsize=15, fontweight="semibold", color=INK, pad=12)
    eje_limpio(ax, grid_x=True, grid_y=False)
    ax.spines["left"].set_visible(False)
fig.text(0.055, 0.09, "Desde la bandeja el QR queda con menos de 2 px por módulo. Alternativas ya implementadas: QR corto "
         "(solo el código, 32 mm) y escaneo con el celular (F5).", fontsize=11.5, color=INK2)
guardar(fig, "07_lectura_qr")

# =====================================================================
# 8. Indicadores clave (pitch)
# =====================================================================
man = {p: cargar(p, "mano")["metricas"] for p in PLACAS}
hoj = {p: cargar(p, "hoja")["metricas"] for p in PLACAS}
lat = max(man[p]["latencia_vision_p95_ms"] for p in PLACAS)
det = min(man[p]["deteccion_pct"] for p in PLACAS)
eauto = max(reg[p]["error_auto_max_grafico_mm"] for p in PLACAS)
fps = max(man[p]["fps_captura"] for p in PLACAS)
sel = [obj[p]["metricas"]["acierto_pct"] for p in PLACAS]
qrp = [qr[p]["tasa_pct"] for p in PLACAS]
tiles = [
    (True, f"{coma(det, 0)} %", "mano detectada\nen los frames", "meta ≥ 95 %"),
    (True, "0", "pinzas falsas y cortes\npor minuto", "antes: 28–52 cortes/min"),
    (True, "100 %", "placas autodetectadas\n(135 de 135 imágenes)", "3 colores de placa"),
    (True, f"≤ {coma(eauto)} mm", "error máximo de calibración\ncon autodetección", "meta ≤ 6 mm"),
    (True, f"{coma(lat, 0)} ms", "latencia de visión\n(percentil 95)", "meta ≤ 100 ms"),
    (False, f"{coma(min(sel), 0)}–{coma(max(sel), 0)} %", "acierto del elemento\nexacto", "meta 90 % · causa: la yema"),
    (False, f"{coma(fps, 0)} fps", "imágenes por segundo\nde la cámara", "meta 25 fps"),
    (False, f"{coma(min(qrp), 0)}–{coma(max(qrp), 0)} %", "lectura del QR con la\ncámara del módulo", "meta 80 % · plan: celular"),
]
fig = figura("Explorador de gráficas táctiles: lo que ya funciona y lo que falta",
             "Indicadores de las pruebas técnicas con 3 placas impresas · 8 nov 2026")
cols, filas_t = 4, 2
for i, (ok, valor, texto, nota) in enumerate(tiles):
    r, c = divmod(i, cols)
    x0, y0, w, h = 0.055 + c * 0.228, 0.47 - r * 0.36, 0.21, 0.31
    axt = fig.add_axes([x0, y0, w, h])
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.axis("off")
    axt.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.04",
                                 facecolor="#ffffff", edgecolor="#e1e0d9", linewidth=1, transform=axt.transAxes))
    col = GOOD if ok else CRIT
    axt.add_patch(Circle((0.11, 0.84), 0.055, color=col, transform=axt.transAxes))
    axt.text(0.11, 0.835, "✓" if ok else "!", ha="center", va="center", fontsize=13, fontweight="bold",
             color="#ffffff", transform=axt.transAxes)
    axt.text(0.2, 0.835, "Cumple" if ok else "Pendiente", va="center", fontsize=12, color=INK2,
             transform=axt.transAxes)
    axt.text(0.07, 0.56, valor, fontsize=34, fontweight="semibold", color=INK, va="center", transform=axt.transAxes)
    axt.text(0.07, 0.27, texto, fontsize=12, color=INK, va="center", transform=axt.transAxes, linespacing=1.3)
    axt.text(0.07, 0.08, nota, fontsize=10.5, color=MUTED, va="center", transform=axt.transAxes)
guardar(fig, "08_indicadores_pitch")
