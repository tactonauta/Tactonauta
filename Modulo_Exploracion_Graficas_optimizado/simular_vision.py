#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULAR VISIÓN  ·  pruebas de la Persona A sin cámara
=====================================================

Dibuja escenas sintéticas (la placa del CSV sobre la bandeja azul, con el inserto
claro o sobre negro mate; con giro, perspectiva, sombra, luz desigual, ruido y JPEG)
y, como se conoce la posición exacta de la placa, mide:

  · autodetección: si se encuentra la placa y el error del registro en mm
    (antes: rectángulo con Otsu en gris · ahora: 4 esquinas con vision.detectar_placa)
  · QR: tasa de lectura y ms por intento, a 640×480 y a más resolución

Los colores de las placas salen de las mediciones del 20 de octubre (Lab de la placa
morada) y de los grises de las tres placas; son aproximados. Un buen resultado aquí
no sustituye la prueba con cámara: sirve para comparar versiones del código.

    python simular_vision.py                               (todas las escenas, gráfica 2)
    python simular_vision.py --datos datos_graficas/ejemplo_temperatura.csv
    python simular_vision.py --guardar carpeta_imagenes    (guarda cada escena dibujada)
    python simular_vision.py --qr                          (añade la prueba del QR)
"""

import argparse
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
if AQUI not in sys.path:
    sys.path.insert(0, AQUI)

import vision as vi  # noqa: E402

# Colores Lab (L 0-100, a, b). Grises resultantes: placas ~115-165, bandeja ~98, inserto ~163.
PLACAS = {"morado": (49.6, 19.3, -35.8), "salmon": (67.0, 24.0, 21.0), "turquesa": (60.0, -27.0, -9.0)}
FONDOS = {"bandeja": (42.0, 4.0, -36.0), "negro": (13.0, 0.0, 0.0)}
INSERTO = (66.5, 0.0, 2.0)
PX_MM_LIENZO = 4.0


def lab_a_bgr(lab):
    L, a, b = lab
    p = np.array([[[L, a, b]]], np.float32)
    return (cv2.cvtColor(p, cv2.COLOR_Lab2BGR)[0, 0] * 255).clip(0, 255)


def lienzo_placa(datos, color, qr_relieve, rng):
    """La placa dibujada en su lienzo (4 px/mm): fondo, relieve de los elementos y QR."""
    W, H = datos.ancho_ref, datos.alto_ref
    k = PX_MM_LIENZO
    base = lab_a_bgr(color)
    img = np.empty((int(H * k), int(W * k), 3), np.float32)
    img[:] = base
    sombra, luz = base * 0.78, np.minimum(base * 1.12 + 8, 255)

    def g(x, y):
        return int(round(x * k)), int(round(y * k))

    for e in datos.elementos:
        if str(e["id"]).lower() == "qr":
            continue
        if e["forma"] == "punto":
            cv2.circle(img, g(e["x1"], e["y1"]), int(2.2 * k), sombra.tolist(), -1, cv2.LINE_AA)
            cv2.circle(img, g(e["x1"] - 0.3, e["y1"] - 0.3), int(1.6 * k), luz.tolist(), -1, cv2.LINE_AA)
        elif e["forma"] == "rect":
            cv2.rectangle(img, g(e["x1"], e["y1"]), g(e["x2"], e["y2"]), sombra.tolist(), max(1, int(0.6 * k)))
        else:
            grosor = int((1.2 if e["tipo"] == "eje" else 1.6) * k)
            cv2.line(img, g(e["x1"], e["y1"]), g(e["x2"], e["y2"]), sombra.tolist(), grosor, cv2.LINE_AA)
            cv2.line(img, g(e["x1"] - 0.3, e["y1"] - 0.3), g(e["x2"] - 0.3, e["y2"] - 0.3), luz.tolist(),
                     max(1, grosor // 3), cv2.LINE_AA)
    qr_mm = vi.esquinas_qr_mm(datos)
    if qr_mm is not None:
        codigo = cv2.QRCodeEncoder.create().encode("https://tactonauta.onrender.com/q/54D7E01D5C")
        x0, y0 = qr_mm[0]
        x1, y1 = qr_mm[2]
        modulo = cv2.resize((codigo < 128).astype(np.uint8), (int((x1 - x0) * k), int((y1 - y0) * k)),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        zona = img[int(y0 * k):int(y0 * k) + modulo.shape[0], int(x0 * k):int(x0 * k) + modulo.shape[1]]
        if qr_relieve == "blanco":   # módulos pintados de blanco sobre la placa (como la morada), sin defectos
            zona[modulo] = zona[modulo] * 0.15 + 225
        elif qr_relieve:    # mismo color, solo sombra (como la placa turquesa)
            zona[modulo] = zona[modulo] * 0.86
        else:
            zona[modulo] = zona[modulo] * 0.35
    return img


def mascara_placa(datos):
    k = PX_MM_LIENZO
    poli = (vi.contorno_placa_mm(datos) * k).astype(np.int32)
    m = np.zeros((int(datos.alto_ref * k), int(datos.ancho_ref * k)), np.uint8)
    cv2.fillPoly(m, [poli], 255)
    return m


def homografia_verdadera(datos, w, h, rng, giro, centro, escala, persp_px):
    W, H = datos.ancho_ref, datos.alto_ref
    esq = np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]]) * escala
    c, s = math.cos(math.radians(giro)), math.sin(math.radians(giro))
    esq = esq @ np.array([[c, s], [-s, c]]) + centro
    esq += rng.uniform(-persp_px, persp_px, esq.shape)
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    return cv2.getPerspectiveTransform(src, esq.astype(np.float32))


def escena(datos, placa, fondo, rng, w=640, h=480, inserto=False, giro=0.0, despl=(0, 0),
           escala=1.66, persp_px=3.0, qr_relieve=False, jpeg=85):
    """(imagen BGR, homografía verdadera mm → px)."""
    Hm = homografia_verdadera(datos, w, h, rng, giro, np.array([w / 2 + despl[0], h / 2 + despl[1]]),
                              escala * w / 640, persp_px * w / 640)
    k = PX_MM_LIENZO
    S = np.diag([1 / k, 1 / k, 1.0])
    fondo_bgr = lab_a_bgr(FONDOS[fondo])
    img = np.empty((h, w, 3), np.float32)
    img[:] = fondo_bgr
    img += rng.normal(0, 3, img.shape) * (fondo == "bandeja")
    if inserto:
        # Inserto claro de 240 × 240 mm, fijo en la bandeja (la placa va encima, algo descentrada).
        W2 = 240
        Hi = homografia_verdadera(datos, w, h, rng, giro * 0.3, np.array([w / 2, h / 2]),
                                  escala * w / 640 * W2 / datos.ancho_ref, 1.0)
        mi = cv2.warpPerspective(np.full((int(datos.alto_ref * k), int(datos.ancho_ref * k)), 255, np.uint8),
                                 Hi @ S, (w, h))
        img[mi > 127] = lab_a_bgr(INSERTO)
    m = mascara_placa(datos)
    mw = cv2.warpPerspective(m, Hm @ S, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    # Sombra de la placa (un poco desplazada y difusa).
    T = np.array([[1, 0, 3 * w / 640], [0, 1, 4 * w / 640], [0, 0, 1]])
    sombra = cv2.GaussianBlur(cv2.warpPerspective(m, T @ Hm @ S, (w, h)).astype(np.float32) / 255, (0, 0), 3)
    img *= (1 - 0.35 * sombra)[..., None]
    lienzo = lienzo_placa(datos, PLACAS[placa], qr_relieve, rng)
    k_px = np.hypot(*(Hm @ S)[:2, 0])                 # px de la imagen por px del lienzo
    if k_px < 1:                                      # antialias antes de reducir (como la óptica)
        lienzo = cv2.GaussianBlur(lienzo, (0, 0), 0.5 / k_px)
    lp = cv2.warpPerspective(lienzo, Hm @ S, (w, h), flags=cv2.INTER_LINEAR)
    img = img * (1 - mw[..., None]) + lp * mw[..., None]
    # Luz desigual y viñeteado.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    luz = 0.88 + 0.22 * (xx / w) * rng.uniform(0.5, 1.0) + 0.06 * (yy / h)
    luz *= 1 - 0.18 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    img *= luz[..., None]
    img = cv2.GaussianBlur(img, (0, 0), 0.8)          # desenfoque de la cámara, en px del sensor
    img += rng.normal(0, 2.5, img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), Hm


# ------------------------------------------------------------
# El método anterior (para comparar): Otsu en gris + rectángulo envolvente
# ------------------------------------------------------------
def detectar_hoja_antigua(frame):
    h, w = frame.shape[:2]
    gris = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for img in (bw, 255 - bw):
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in sorted(contornos, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area < 0.15 * h * w or area > 0.97 * h * w:
                continue
            aprox = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(aprox) in (4, 5) and cv2.isContourConvex(aprox):
                return cv2.boundingRect(c)
    return None


def caja_grafico(datos):
    ejes = [e for e in datos.elementos if e["tipo"] == "eje"] or datos.elementos
    xs = [v for e in ejes for v in (e["x1"], e["x2"])]
    ys = [v for e in ejes for v in (e["y1"], e["y2"])]
    return min(xs), min(ys), max(xs), max(ys)


def error_registro(datos, verdad, est):
    """(mediana, máximo) del error en el área del gráfico, en mm."""
    if est is None:
        return None, None
    x0, y0, x1, y1 = caja_grafico(datos)
    return verdad.diferencia_mm(est, x0, y0, x1, y1, n=21)


def escenarios():
    """(nombre, kwargs). 5 posiciones por placa y fondo, como en la simulación del 26 de octubre."""
    res = []
    for placa in PLACAS:
        for fondo, inserto in (("bandeja", False), ("bandeja", True), ("negro", False)):
            for i in range(5):
                giro = (-4, -1.5, 0, 2, 5)[i]
                despl = ((0, 0), (-25, 8), (20, -10), (10, 12), (-12, -6))[i]
                nombre = f"{placa:<8} {fondo + (' + inserto' if inserto else ''):<17} giro {giro:+.1f}°"
                res.append((nombre, dict(placa=placa, fondo=fondo, inserto=inserto, giro=giro, despl=despl)))
    # Placa cortada por arriba (los últimos ~15 mm fuera del encuadre, como en la turquesa).
    for placa in PLACAS:
        res.append((f"{placa:<8} {'bandeja, cortada':<17} arriba   ",
                    dict(placa=placa, fondo="bandeja", inserto=False, giro=1.0, despl=(0, -(240 - 182 + 25)))))
    return res


def probar_autodeteccion(datos, guardar=None, semilla=3):
    rng = np.random.default_rng(semilla)
    filas = []
    print(f"{'escena':<45} | {'antes: rect gris':<26} | ahora: 4 esquinas")
    print("-" * 110)
    for nombre, kw in escenarios():
        img, Hm = escena(datos, rng=rng, **kw)
        h, w = img.shape[:2]
        verdad = vi.Registro(Hm, datos.ancho_ref, datos.alto_ref, "verdad")
        rect = detectar_hoja_antigua(img)
        r_ant = vi.Registro.desde_rect(rect, datos.ancho_ref, datos.alto_ref, datos.estirar) if rect else None
        e_ant = error_registro(datos, verdad, r_ant)
        t0 = time.perf_counter()
        det = vi.detectar_placa(img, datos)
        ms = 1000 * (time.perf_counter() - t0)
        r_nuevo = det.registro(datos) if det else None
        e_nuevo = error_registro(datos, verdad, r_nuevo)
        fuera = verdad.bordes_fuera(w, h)

        def fmt(e):
            if e[0] is None:
                return "no la encuentra"
            return f"med {e[0]:4.1f} · máx {e[1]:5.1f} mm"

        extra = ""
        if det:
            extra = (f" [{det.metodo}, {det.confianza}" + (", reconstruida" if det.reconstruida else "")
                     + (f", recorte {det.esquina_vista}" if det.esquina_vista else "") + f", {ms:.0f} ms]")
            if r_nuevo is not None and r_nuevo.bordes_fuera(w, h):
                extra += " avisa: borde " + "/".join(r_nuevo.bordes_fuera(w, h))
        print(f"{nombre:<45} | {fmt(e_ant):<26} | {fmt(e_nuevo)}{extra}")
        filas.append({"escena": nombre, "kw": kw, "antes": e_ant, "ahora": e_nuevo, "ms": ms,
                      "cortada": bool(fuera), "det": det})
        if guardar:
            os.makedirs(guardar, exist_ok=True)
            vista = img.copy()
            if rect:
                x, y, rw, rh = rect
                cv2.rectangle(vista, (x, y), (x + rw, y + rh), (0, 0, 255), 1)
            if det:
                cv2.polylines(vista, [det.esquinas.round().astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
            nombre_archivo = "_".join(nombre.split()).replace("°", "").replace("+", "p").replace(",", "")
            cv2.imwrite(os.path.join(guardar, nombre_archivo + ".png"), vista)
    print("-" * 110)
    for clave in ("antes", "ahora"):
        normales = [f for f in filas if not f["cortada"]]
        ok = [f for f in normales if f[clave][1] is not None and f[clave][1] <= 6.0]
        enc = [f for f in normales if f[clave][1] is not None]
        print(f"{clave:>5}: encontrada en {len(enc)}/{len(normales)} escenas · con error máx ≤ 6 mm en "
              f"{len(ok)}/{len(normales)}", end="")
        meds = [f[clave][0] for f in enc]
        print(f" · mediana de las medianas {np.median(meds):.1f} mm" if meds else "")
        for placa in PLACAS:
            for fondo in ("bandeja", "bandeja + inserto", "negro"):
                sub = [f for f in normales if f["kw"]["placa"] == placa and
                       (f["kw"]["fondo"] + (" + inserto" if f["kw"]["inserto"] else "")) == fondo]
                k = sum(1 for f in sub if f[clave][1] is not None and f[clave][1] <= 6.0)
                print(f"        {placa:<9} {fondo:<18} {k}/{len(sub)}")
    cort = [f for f in filas if f["cortada"]]
    print(f"cortadas: avisa del borde en {sum(1 for f in cort if f['det'] and f['det'].registro(datos).bordes_fuera(640, 480))}"
          f"/{len(cort)}; error máx (ahora): " + ", ".join(
              "—" if f["ahora"][1] is None else f"{f['ahora'][1]:.1f} mm" for f in cort))
    print(f"tiempo de detectar_placa: mediana {np.median([f['ms'] for f in filas]):.0f} ms")
    return filas


def probar_qr(datos, semilla=11, n=6, intentos=15):
    """Lectura del QR con la placa en la bandeja: imagen completa (como antes) frente a zoom
    a la esquina de la placa (como ahora en F3), a varias resoluciones con el mismo campo
    de visión. Cada escena se deja quieta hasta 'intentos' × 200 ms (unos 3 s)."""
    rng = np.random.default_rng(semilla)
    print("\nQR en la bandeja · lecturas en escenas quietas (hasta ~3 s cada una) · ms hasta leer")
    for (w, h) in ((640, 480), (1280, 960), (1600, 1200), (1920, 1440)):
        for placa, estilo in (("morado", "blanco"), ("salmon", False), ("turquesa", True)):
            escenas = [escena(datos, placa, "bandeja", rng, w=w, h=h, giro=rng.uniform(-3, 3),
                              despl=(rng.uniform(-15, 15) * w / 640, rng.uniform(-10, 10) * w / 640),
                              qr_relieve=estilo)[0] for _ in range(n)]
            linea = []
            for modo in ("completa", "zoom"):
                ok, tiempos = 0, []
                for img in escenas:
                    lector = vi.LectorQR()
                    zoom = vi.ZoomQR({}, lector) if modo == "zoom" else None
                    t0 = time.perf_counter()
                    for _ in range(intentos):
                        texto = (zoom.leer(img, "ninguno", 200)[0] if zoom else lector.leer(img, presupuesto_ms=200)[0])
                        if texto:
                            ok += 1
                            tiempos.append(1000 * (time.perf_counter() - t0))
                            break
                linea.append(f"{modo}: {ok}/{n}" + (f" ({np.median(tiempos):4.0f} ms)" if tiempos else "         "))
            lado = (vi.esquinas_qr_mm(datos)[1][0] - vi.esquinas_qr_mm(datos)[0][0]) * 1.66 * w / 640
            nombre = placa + (" (blanco)" if estilo == "blanco" else " (relieve)" if estilo else "")
            print(f"  {w}x{h}  {nombre:<19} QR ≈ {lado:3.0f} px · " + " · ".join(linea))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", help="CSV de la gráfica (por defecto, el de datos_graficas que tenga QR)")
    p.add_argument("--guardar", help="carpeta donde guardar las escenas dibujadas")
    p.add_argument("--qr", action="store_true", help="añadir la prueba del QR")
    a = p.parse_args()
    import explorador_graficas_gui_qr as eg
    ruta = a.datos
    if not ruta:
        for r in sorted(glob.glob(os.path.join(AQUI, "datos_graficas", "*.csv"))):
            with open(r, encoding="utf-8-sig") as f:
                if "\ntexto,qr," in f.read():
                    ruta = r
                    break
        ruta = ruta or os.path.join(AQUI, "datos_graficas", "ejemplo_temperatura.csv")
    datos = eg.cargar_desde_enlace(ruta)
    print(f"Gráfica: {datos.titulo} ({os.path.basename(ruta)}), placa {datos.ancho_ref:g} × {datos.alto_ref:g} mm, "
          f"esquina recortada: {datos.esquina_recortada or 'ninguna'}\n")
    probar_autodeteccion(datos, a.guardar)
    if a.qr:
        probar_qr(datos)


if __name__ == "__main__":
    main()
