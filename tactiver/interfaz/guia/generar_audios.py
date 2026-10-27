"""Graba el audio de cada guía de usuario (guias.json) en audio/<id>.mp3.

Usa la voz neuronal Camila (español de Perú) del servicio de lectura en voz
alta de Microsoft, con la librería edge-tts (hace falta internet):

    pip install edge-tts
    python3 generar_audios.py              # graba las que falten
    python3 generar_audios.py --todas      # vuelve a grabar todas
    python3 generar_audios.py --voz es-PE-AlexNeural   # otra voz

Voces en español: edge-tts --list-voices | grep ^es-

Si cambias el texto de una guía, vuelve a grabar su audio (con --todas, o
borrando su .mp3). También se puede reemplazar un .mp3 por una grabación
propia, con el mismo nombre: la app usa el archivo que encuentre.
"""
import argparse
import asyncio
import json
import os

import edge_tts

CARPETA = os.path.dirname(os.path.abspath(__file__))
VOZ = "es-PE-CamilaNeural"
VELOCIDAD = "-5%"         # un poco más pausado que lo normal


async def grabar(texto, destino, voz, velocidad):
    await edge_tts.Communicate(texto, voz, rate=velocidad).save(destino)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--todas", action="store_true", help="volver a grabar también las que ya existen")
    p.add_argument("--voz", default=VOZ)
    p.add_argument("--velocidad", default=VELOCIDAD, help='por ejemplo "-5%%" o "+10%%"')
    args = p.parse_args()

    with open(os.path.join(CARPETA, "guias.json"), encoding="utf-8") as f:
        guias = json.load(f)["guias"]
    os.makedirs(os.path.join(CARPETA, "audio"), exist_ok=True)
    for g in guias:
        destino = os.path.join(CARPETA, "audio", f"{g['id']}.mp3")
        if os.path.exists(destino) and not args.todas:
            print(f"ya está: {g['id']}")
            continue
        asyncio.run(grabar(f"{g['titulo']}. {g['texto']}", destino, args.voz, args.velocidad))
        print(f"grabada: {g['id']} ({os.path.getsize(destino) // 1024} KB)")


if __name__ == "__main__":
    main()
