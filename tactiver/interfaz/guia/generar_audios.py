"""Graba el audio de cada guía de usuario (guias.json) en audio/<id>.m4a.

Solo funciona en una Mac: usa la voz del sistema (comando `say`) y la
convierte a AAC con `afconvert`, que vienen con macOS.

    python3 generar_audios.py              # graba las que falten
    python3 generar_audios.py --todas      # vuelve a grabar todas
    python3 generar_audios.py --voz Mónica # otra voz en español

Si cambias el texto de una guía, vuelve a grabar su audio (con --todas, o
borrando su .m4a). También se puede reemplazar un .m4a por una grabación
propia, con el mismo nombre: la app usa el archivo que encuentre.
"""
import argparse
import json
import os
import subprocess
import tempfile

CARPETA = os.path.dirname(os.path.abspath(__file__))
VOZ = "Paulina"           # español latinoamericano
VELOCIDAD = 170           # palabras por minuto (la de `say` es ~175)
KBPS = 48000              # voz en mono: chico y claro


def grabar(texto, destino, voz, velocidad):
    with tempfile.TemporaryDirectory() as tmp:
        aiff = os.path.join(tmp, "voz.aiff")
        subprocess.run(["say", "-v", voz, "-r", str(velocidad), "-o", aiff, texto], check=True)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", str(KBPS), "-c", "1",
                        aiff, destino], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--todas", action="store_true", help="volver a grabar también las que ya existen")
    p.add_argument("--voz", default=VOZ)
    p.add_argument("--velocidad", type=int, default=VELOCIDAD)
    args = p.parse_args()

    with open(os.path.join(CARPETA, "guias.json"), encoding="utf-8") as f:
        guias = json.load(f)["guias"]
    os.makedirs(os.path.join(CARPETA, "audio"), exist_ok=True)
    for g in guias:
        destino = os.path.join(CARPETA, "audio", f"{g['id']}.m4a")
        if os.path.exists(destino) and not args.todas:
            print(f"ya está: {g['id']}")
            continue
        grabar(f"{g['titulo']}. {g['texto']}", destino, args.voz, args.velocidad)
        print(f"grabada: {g['id']} ({os.path.getsize(destino) // 1024} KB)")


if __name__ == "__main__":
    main()
