#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCANEAR EL QR CON EL CELULAR  ·  página temporal en la red local
=================================================================

La cámara del módulo, desde su altura, no distingue los módulos del QR. La del celular sí:
enfoca de cerca y tiene mucha más resolución. Este módulo levanta, solo mientras se escanea
(tecla F5 en la app), una página web en la red Wi-Fi:

    http://<IP de la computadora>:8765/<código de 4 cifras>

El celular la abre en su navegador (escribiendo la dirección o apuntando su cámara al QR que
la app muestra en pantalla), toca «Tomar foto del código», fotografía la esquina de la placa
y la página envía la foto. La computadora busca el QR en la foto y,
si lo encuentra, se lo pasa a la app, que descarga la gráfica y apaga la página.

· No hay que instalar nada en el celular: usa el selector de fotos del navegador
  (<input type="file" capture>), que funciona sin HTTPS. La cámara en vivo del navegador
  (getUserMedia) no se puede usar: exige HTTPS, y en una red local no lo hay.
· La dirección lleva un código de 4 cifras al azar: otra persona en la red no lo adivina
  durante los pocos minutos que la página está abierta (se apaga sola a los 10 minutos).
· Solo funciona si el celular y la computadora están en la misma red, y esa red deja que
  los equipos se vean entre sí. Las dos causas típicas de que la página «no cargue»:
    1. El Firewall de Windows bloquea la conexión entrante (sobre todo si la red Wi-Fi está
       marcada como «pública», o si alguna vez se cerró sin aceptar el aviso de Python).
       permitir_firewall() agrega una regla para los puertos 8765-8770 (pide permiso de
       administrador una sola vez).
    2. La red aísla a los equipos entre sí («aislamiento de clientes»): es lo normal en redes
       de universidad, hoteles o invitados. Ahí no hay nada que hacer en la computadora: se
       usa la zona con cobertura (hotspot) de la computadora o la del celular, y los dos
       equipos se conectan a esa red.
  direcciones_locales() lista todas las direcciones de la computadora (Wi-Fi, cable, hotspot)
  para poder elegir la de la red en la que está el celular.
"""

import html
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

MAX_FOTO = 25_000_000          # bytes
PUERTOS = range(8765, 8771)    # el primero libre; la regla del firewall cubre todos
REGLA_FIREWALL = "Explorador de graficas - celular"
_VIRTUALES = ("vethernet", "virtualbox", "vmware", "hyper-v", "wsl", "docker", "loopback", "bluetooth",
              "tap-", "tap ", "vpn", "tailscale", "zerotier", "hamachi", "pseudo", "teredo", "isatap")
LADO_MAX = 3000                # px: una foto de 12 MP se reduce a esto antes de leer
DURACION_S = 600               # la página se apaga sola a los 10 minutos


def ip_local():
    """IP de esta computadora en la red local. No envía nada: solo pregunta al sistema qué
    interfaz usaría para salir a internet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def _salida_cmd(args, timeout=6):
    """Texto que imprime un comando de Windows (en la codificación de la consola)."""
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None, ""
    for cod in ("utf-8", "cp850", "mbcs", "latin-1"):
        try:
            return r.returncode, r.stdout.decode(cod)
        except (UnicodeDecodeError, LookupError):
            continue
    return r.returncode, r.stdout.decode("latin-1", "replace")


def _adaptadores_ipconfig(texto):
    """[(nombre del adaptador, ip)] a partir de la salida de ipconfig (Windows, en cualquier idioma)."""
    res, actual = [], None
    for linea in texto.splitlines():
        if linea.strip() and not linea[:1].isspace() and linea.rstrip().endswith(":"):
            actual = linea.strip().rstrip(":")
            continue
        m = re.search(r"IPv4[^:]*:\s*(\d{1,3}(?:\.\d{1,3}){3})", linea)
        if m and actual:
            res.append((actual, m.group(1)))
    return res


def direcciones_locales():
    """Direcciones IPv4 de esta computadora, de la más probable a la menos:
    [{"ip", "adaptador", "tipo"}]. tipo: "wifi", "cable", "hotspot" (la zona con cobertura de
    Windows, 192.168.137.1) u "otra". Se descartan las de máquinas virtuales y VPN."""
    principal = ip_local()
    vistas = []
    if sys.platform.startswith("win"):
        _cod, txt = _salida_cmd(["ipconfig"])
        vistas = _adaptadores_ipconfig(txt)
    if not vistas:
        try:
            ips = {ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
        except OSError:
            ips = set()
        vistas = [("", ip) for ip in sorted(ips | {principal})]
    res = []
    for nombre, ip in vistas:
        if ip.startswith(("127.", "169.254.", "0.")) or any(r["ip"] == ip for r in res):
            continue
        n = nombre.lower()
        if any(v in n for v in _VIRTUALES):
            continue
        if ip.startswith("192.168.137."):
            tipo, puntos = "hotspot", 1
        elif any(k in n for k in ("wi-fi", "wifi", "inalámbrica", "inalambrica", "wireless", "wlan")):
            tipo, puntos = "wifi", 5
        elif any(k in n for k in ("ethernet", "lan")):
            tipo, puntos = "cable", 3
        else:
            tipo, puntos = "otra", 2
        if ip == principal:
            puntos += 2
        res.append({"ip": ip, "adaptador": nombre, "tipo": tipo, "puntos": puntos})
    if not res:
        res = [{"ip": principal, "adaptador": "", "tipo": "otra", "puntos": 0}]
    res.sort(key=lambda r: -r["puntos"])
    return res


def estado_firewall():
    """Windows: {"regla": bool|None, "perfiles": ["Public", ...]}. Fuera de Windows: None."""
    if not sys.platform.startswith("win"):
        return None
    cod, txt = _salida_cmd(["netsh", "advfirewall", "firewall", "show", "rule", f"name={REGLA_FIREWALL}"])
    regla = None if cod is None else (cod == 0 and "8765" in txt)
    _c, perf = _salida_cmd(["powershell", "-NoProfile", "-Command",
                            "(Get-NetConnectionProfile).NetworkCategory"], timeout=10)
    perfiles = [l.strip() for l in perf.splitlines() if l.strip()]
    return {"regla": regla, "perfiles": perfiles}


def permitir_firewall():
    """Pide a Windows (con el aviso de administrador) una regla que deje entrar conexiones a los
    puertos 8765-8770 en cualquier tipo de red. Devuelve True si se lanzó la petición."""
    if not sys.platform.startswith("win"):
        return False
    import ctypes
    args = (f'advfirewall firewall add rule name="{REGLA_FIREWALL}" dir=in action=allow protocol=TCP '
            f'localport={PUERTOS[0]}-{PUERTOS[-1]} profile=any')
    try:
        r = ctypes.windll.shell32.ShellExecuteW(None, "runas", "netsh", args, None, 0)
    except Exception:
        return False
    return int(r) > 32


def direccion_hablada(url):
    """'http://192.168.1.20:8765/4821' → texto para decir en voz alta, cifra por cifra."""
    resto = url.split("://", 1)[-1]
    partes = []
    for c in resto:
        if c.isdigit():
            partes.append(c)
        elif c == ".":
            partes.append("punto")
        elif c == ":":
            partes.append("dos puntos")
        elif c == "/":
            partes.append("barra")
        else:
            partes.append(c)
    return " ".join(partes)


def leer_codigo(img, cfg, lector=None):
    """Busca el QR en una foto. Devuelve el texto o None."""
    from vision import LectorQR
    lector = lector or LectorQR()
    h, w = img.shape[:2]
    f = min(1.0, LADO_MAX / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    # La foto puede estar muy cerca o muy lejos: varias escalas, cada una con varios intentos.
    for escala in (0.5, 1.0, 0.25):                  # de cerca, a la mitad suele sobrar y es más rápido
        im = img if escala == 1.0 else cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        if max(im.shape[:2]) < 300:
            continue
        for _ in range(3):
            texto, _ = lector.leer(im, presupuesto_ms=1500)
            if texto:
                return texto
    return None


PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Escanear la gráfica</title>
<style>
 body{{margin:0;padding:24px 16px;background:#000;color:#ff0;font:22px/1.4 system-ui,sans-serif}}
 h1{{font-size:28px;margin:0 0 12px}}
 p{{margin:0 0 16px}}
 label.boton{{display:block;text-align:center;background:#ff0;color:#000;font-weight:700;font-size:28px;
   padding:28px 16px;border-radius:14px;border:4px solid #0ff;cursor:pointer}}
 label.boton:focus-within{{outline:6px solid #0ff}}
 input[type=file]{{position:absolute;opacity:0;width:1px;height:1px}}
 #estado{{margin-top:20px;font-size:24px;min-height:2em}}
 .ok{{color:#7cfc00}} .mal{{color:#ff7f7f}}
</style></head><body>
<h1>Escanear el código de la gráfica</h1>
<p>Acerca el celular a unos 15 centímetros de la esquina superior izquierda de la placa, donde está
el código, y toma la foto. La computadora la lee y te avisa.</p>
<label class="boton">Tomar foto del código
 <input id="foto" type="file" accept="image/*" capture="environment" aria-label="Tomar foto del código"></label>
<div id="estado" role="status" aria-live="assertive"></div>
<h2 style="font-size:24px;margin:32px 0 8px">¿Ya tienes el enlace?</h2>
<p>Si la cámara del celular ya leyó el código y te dio un enlace, cópialo, pégalo aquí y envíalo.</p>
<input id="enlace" type="url" inputmode="url" aria-label="Enlace de la gráfica"
 style="width:100%;box-sizing:border-box;font-size:22px;padding:14px;border:4px solid #0ff;background:#111;color:#fff">
<button id="enviar" style="margin-top:12px;width:100%;font-size:26px;font-weight:700;padding:20px;background:#ff0;
 color:#000;border:4px solid #0ff;border-radius:14px">Enviar enlace</button>
<script>
const estado = document.getElementById('estado');
function decir(t, clase){{ estado.className = clase || ''; estado.textContent = t; }}
async function reducir(archivo){{
  // Fotos de 12 MP pesan mucho por Wi-Fi: se reducen a 2400 px, que sobra para el código.
  try {{
    const bm = await createImageBitmap(archivo);
    const k = Math.min(1, 2400 / Math.max(bm.width, bm.height));
    if (k >= 1) return archivo;
    const c = document.createElement('canvas');
    c.width = Math.round(bm.width * k); c.height = Math.round(bm.height * k);
    c.getContext('2d').drawImage(bm, 0, 0, c.width, c.height);
    return await new Promise(r => c.toBlob(b => r(b || archivo), 'image/jpeg', 0.92));
  }} catch (e) {{ return archivo; }}
}}
document.getElementById('enviar').addEventListener('click', async () => {{
  const t = document.getElementById('enlace').value.trim();
  if (!t) {{ decir('Primero pega el enlace en el cuadro.', 'mal'); return; }}
  try {{
    const r = await fetch('{ruta}/enlace', {{method: 'POST', body: t, headers: {{'Content-Type': 'text/plain'}}}});
    const j = await r.json();
    decir(j.mensaje, j.ok ? 'ok' : 'mal');
  }} catch (e) {{ decir('No pude enviar el enlace. Revisa la red Wi-Fi.', 'mal'); }}
}});
document.getElementById('foto').addEventListener('change', async (ev) => {{
  const archivo = ev.target.files[0];
  if (!archivo) return;
  decir('Enviando la foto. Un momento.');
  try {{
    const cuerpo = await reducir(archivo);
    const r = await fetch('{ruta}/foto', {{method: 'POST', body: cuerpo,
                                          headers: {{'Content-Type': 'application/octet-stream'}}}});
    const j = await r.json();
    decir(j.mensaje, j.ok ? 'ok' : 'mal');
    if (navigator.vibrate) navigator.vibrate(j.ok ? [200] : [100, 80, 100]);
  }} catch (e) {{
    decir('No pude enviar la foto. Revisa que el celular siga en la misma red Wi-Fi y vuelve a intentarlo.', 'mal');
  }}
  ev.target.value = '';
}});
</script></body></html>
"""


class ServidorCelular:
    """Página temporal para escanear con el celular. al_leer(texto) se llama (desde otro hilo)
    cuando una foto trae un código legible."""

    def __init__(self, cfg, al_leer, puerto=None):
        self.cfg = cfg
        self.al_leer = al_leer
        self.puerto = int(puerto or cfg.get("celular_puerto", 8765))
        self.codigo = "%04d" % random.randint(0, 9999)
        self.ruta = "/" + self.codigo
        self.httpd = None
        self.url = None
        self.direcciones = []          # [{"ip", "adaptador", "tipo"}], la primera es la de self.url
        self.visitas = 0               # cuántas veces un celular abrió la página
        self.t_inicio = None
        self.leido = None
        self._candado = threading.Lock()          # una foto a la vez
        self._t_fin = None

    # ---- ciclo de vida ----
    def iniciar(self):
        """Arranca la página. Devuelve la URL. Si el puerto está ocupado, usa otro libre."""
        servidor = self
        candidatos = [self.puerto] + [p for p in PUERTOS if p != self.puerto] + [0]
        for puerto in candidatos:
            try:
                self.httpd = ThreadingHTTPServer(("0.0.0.0", puerto), self._manejador(servidor))
                break
            except OSError as e:
                print("Aviso: el puerto", puerto, "no está libre:", e)
        if self.httpd is None:
            raise OSError("No pude abrir la página para el celular.")
        self.httpd.daemon_threads = True
        self.puerto = self.httpd.server_address[1]
        try:
            self.direcciones = direcciones_locales()
        except Exception:
            traceback.print_exc()
            self.direcciones = [{"ip": ip_local(), "adaptador": "", "tipo": "otra"}]
        self.url = self.url_de(0)
        self.t_inicio = time.time()
        self._t_fin = time.time() + DURACION_S
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.3}, daemon=True).start()
        threading.Thread(target=self._vigilar, daemon=True).start()
        print("Página para el celular:", self.url)
        for d in self.direcciones[1:]:
            print("   también:", self.url_de(self.direcciones.index(d)), "·", d["adaptador"] or d["tipo"])
        return self.url

    def url_de(self, i):
        d = self.direcciones[i % len(self.direcciones)]
        return f"http://{d['ip']}:{self.puerto}{self.ruta}"

    def detener(self):
        h, self.httpd = self.httpd, None
        if h is not None:
            threading.Thread(target=lambda: (h.shutdown(), h.server_close()), daemon=True).start()
            print("Página para el celular apagada.")

    @property
    def activo(self):
        return self.httpd is not None

    def _vigilar(self):
        while self.httpd is not None:
            if time.time() > self._t_fin:
                print("La página para el celular se apagó tras", DURACION_S // 60, "minutos.")
                self.detener()
                break
            time.sleep(1)

    # ---- la foto ----
    def procesar_foto(self, datos):
        """Bytes de la foto → (ok, mensaje para la página)."""
        if not self._candado.acquire(blocking=False):
            return False, "Todavía estoy leyendo la foto anterior. Espera un momento."
        try:
            img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return False, ("No pude abrir la foto. Si el celular guarda en formato HEIC, "
                               "vuelve a intentarlo: la página la convierte al reducirla.")
            t0 = time.perf_counter()
            texto = leer_codigo(img, self.cfg)
            print("Foto del celular: %dx%d, %s en %.1f s" % (img.shape[1], img.shape[0],
                  "código leído" if texto else "sin código", time.perf_counter() - t0))
            if not texto:
                return False, ("No encontré el código en la foto. Acerca el celular a unos 15 centímetros "
                               "de la esquina superior izquierda de la placa y toma otra.")
            self.leido = texto
            try:
                self.al_leer(texto)
            except Exception:
                traceback.print_exc()
            return True, "Código leído. La computadora está cargando la gráfica; escucha su aviso."
        except Exception:
            traceback.print_exc()
            return False, "Ocurrió un error al leer la foto. Vuelve a intentarlo."
        finally:
            self._candado.release()

    def procesar_enlace(self, texto):
        """Un enlace pegado en la página (el celular ya leyó el QR con su propia cámara)."""
        t = texto.strip()
        solo_codigo = re.fullmatch(r"[0-9A-Za-z]{6,16}", t) is not None and re.search(r"\d", t)
        if not t or len(t) > 2000 or not ("://" in t or "." in t or solo_codigo):
            return False, "Eso no parece un enlace. Pega el enlace completo que te dio el código."
        self.leido = t
        try:
            self.al_leer(t)
        except Exception:
            traceback.print_exc()
        return True, "Enlace recibido. La computadora está cargando la gráfica; escucha su aviso."

    @staticmethod
    def _manejador(servidor):
        class Manejador(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass                                  # sin ruido en la consola

            def _enviar(self, codigo, cuerpo, tipo):
                datos = cuerpo.encode("utf-8") if isinstance(cuerpo, str) else cuerpo
                self.send_response(codigo)
                self.send_header("Content-Type", tipo)
                self.send_header("Content-Length", str(len(datos)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(datos)

            def do_GET(self):
                if self.path.rstrip("/") == servidor.ruta:
                    servidor.visitas += 1
                    print("El celular abrió la página (%s)." % self.client_address[0])
                    self._enviar(200, PAGINA.format(ruta=html.escape(servidor.ruta)), "text/html; charset=utf-8")
                else:
                    self._enviar(404, "No encontrado", "text/plain; charset=utf-8")

            def do_POST(self):
                if self.path not in (servidor.ruta + "/foto", servidor.ruta + "/enlace"):
                    self._enviar(404, "No encontrado", "text/plain; charset=utf-8")
                    return
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_FOTO:
                    ok, msg = False, "Llegó vacío o es demasiado grande."
                elif self.path.endswith("/enlace"):
                    ok, msg = servidor.procesar_enlace(self.rfile.read(min(n, 4096)).decode("utf-8", "replace"))
                else:
                    ok, msg = servidor.procesar_foto(self.rfile.read(n))
                self._enviar(200, json.dumps({"ok": ok, "mensaje": msg}, ensure_ascii=False),
                             "application/json; charset=utf-8")

        return Manejador


def imagen_qr_ppm(texto, lado_px=280):
    """QR de la dirección, como imagen PPM para tkinter (lo puede escanear la cámara del celular)."""
    cod = cv2.QRCodeEncoder.create().encode(texto)
    cod = cv2.copyMakeBorder(cod, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
    k = max(1, lado_px // cod.shape[0])
    img = cv2.resize(cod, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes()
