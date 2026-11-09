"""Genera el código QR de una placa en SVG, a su tamaño real en mm, para modelarlo e imprimirlo.

Por qué: desde la altura de la bandeja la cámara ve el QR actual (enlace completo, 25,8 mm) con
menos de 2 px por módulo, y así no se puede leer. Un QR con SOLO el código de la lámina
(p. ej. 54D7E01D5C) cabe en la versión 1 (21 × 21 módulos) con la máxima corrección de errores (H,
30 %): a 32 mm cada módulo mide 1,5 mm en vez de ~0,9-1,0 mm, y aguanta manchas de pintura. La
aplicación completa el código con "qr_url_base" (configuración) y descarga la gráfica igual que antes.

Uso:
    py generar_qr_placa.py 54D7E01D5C                 → qr_54D7E01D5C.svg (32 mm, corrección H)
    py generar_qr_placa.py 54D7E01D5C --mm 35
    py generar_qr_placa.py https://tactonauta.onrender.com/q/54D7E01D5C   (también vale el enlace:
                                                       se usa solo el código del final)
    py generar_qr_placa.py 54D7E01D5C --margen 2      (2 módulos de margen blanco alrededor)

Recomendaciones para imprimir: módulos en relieve de 0,8-1,2 mm y con mucho contraste con la placa
(oscuro sobre claro o blanco sobre oscuro); sin redondear las esquinas de los módulos; el QR
empieza a 10 mm de los bordes superior e izquierdo, y hasta 36 mm de lado cabe en la zona que la
aplicación amplía. Si cambias el tamaño, actualiza en el CSV la fila "qr" (x2_mm, y2_mm) y en la
configuración "qr_modulo_mm" (= lado / 21).
"""
import argparse
import re
import sys

import cv2
import numpy as np

NIVELES = {"L": "CORRECT_LEVEL_L", "M": "CORRECT_LEVEL_M", "Q": "CORRECT_LEVEL_Q", "H": "CORRECT_LEVEL_H"}


def matriz_qr(texto, nivel="H"):
    """Matriz de módulos (True = oscuro), sin margen."""
    p = cv2.QRCodeEncoder_Params()
    p.correction_level = getattr(cv2, "QRCodeEncoder_" + NIVELES[nivel])
    img = cv2.QRCodeEncoder.create(p).encode(texto)
    ys, xs = np.where(img < 128)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # El codificador dibuja cada módulo con varios px: se toma el centro de cada uno.
    n = None
    for v in (21, 25, 29, 33, 37, 41):
        if img.shape[0] % v == 0:
            n = v
            break
    if n is None:
        sys.exit("No pude interpretar el QR generado.")
    k = img.shape[0] // n
    return img[k // 2::k, k // 2::k] < 128


def svg(matriz, lado_mm, margen_mod=0):
    n = matriz.shape[0]
    mod = lado_mm / n
    total = lado_mm + 2 * margen_mod * mod
    rects = []
    for y in range(n):
        x = 0
        while x < n:                       # une los módulos seguidos de cada fila (menos polígonos)
            if matriz[y, x]:
                x0 = x
                while x < n and matriz[y, x]:
                    x += 1
                rects.append(f'<rect x="{(x0 + margen_mod) * mod:.3f}" y="{(y + margen_mod) * mod:.3f}" '
                             f'width="{(x - x0) * mod:.3f}" height="{mod:.3f}"/>')
            else:
                x += 1
    fondo = f'<rect width="{total:.3f}" height="{total:.3f}" fill="white"/>' if margen_mod else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{total:.3f}mm" height="{total:.3f}mm" '
            f'viewBox="0 0 {total:.3f} {total:.3f}">{fondo}<g fill="black">' + "".join(rects) + "</g></svg>\n")


def main():
    a = argparse.ArgumentParser(description="QR de una placa, en SVG a tamaño real.")
    a.add_argument("codigo", help="código de la lámina (p. ej. 54D7E01D5C) o su enlace")
    a.add_argument("--mm", type=float, default=32.0, help="lado del QR en mm (por defecto 32)")
    a.add_argument("--nivel", default="H", choices=list(NIVELES), help="corrección de errores (por defecto H)")
    a.add_argument("--margen", type=int, default=0, help="módulos de margen blanco (0: la placa hace de margen)")
    a.add_argument("--salida", help="archivo SVG (por defecto qr_<código>.svg)")
    args = a.parse_args()
    texto = args.codigo.strip()
    m = re.search(r"/q/([0-9A-Za-z]+)/?$", texto, re.I)
    if m:
        texto = m.group(1)
    texto = texto.upper()
    mat = matriz_qr(texto, args.nivel)
    # Comprobación: que el QR generado se lea y diga lo mismo.
    k = 10
    img = np.full(((mat.shape[0] + 8) * k,) * 2, 255, np.uint8)
    img[4 * k:-4 * k, 4 * k:-4 * k] = np.where(np.kron(mat, np.ones((k, k), bool)), 0, 255)
    leido = cv2.QRCodeDetector().detectAndDecode(img)[0]
    if leido != texto:
        sys.exit(f"El QR generado no se lee bien ({leido!r}); no lo guardo.")
    salida = args.salida or f"qr_{texto}.svg"
    with open(salida, "w", encoding="utf-8") as f:
        f.write(svg(mat, args.mm, args.margen))
    n = mat.shape[0]
    print(f"{salida}: «{texto}» · {n} × {n} módulos · corrección {args.nivel} · {args.mm:g} mm "
          f"({args.mm / n:.2f} mm por módulo). En la configuración: \"qr_modulo_mm\": {args.mm / n:.2f}")


if __name__ == "__main__":
    main()
