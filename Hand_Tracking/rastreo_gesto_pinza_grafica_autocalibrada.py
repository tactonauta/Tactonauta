import cv2
import mediapipe as mp
import math
import threading
import queue
import time
import pyttsx3

# ============================================================
# AUDIO EN SEGUNDO PLANO
# ============================================================
# pyttsx3.runAndWait() bloquea el bucle principal y falla al
# reutilizarse. Se usa un hilo con el "bucle externo" de pyttsx3
# (startLoop/iterate/endLoop) para poder cortar el audio en marcha.

cola_audio = queue.Queue()
hablando = threading.Event()            # True mientras se está reproduciendo audio
detener_solicitado = threading.Event()  # Se activa para pedir el corte del audio
utterance_terminada = threading.Event() # La marca el motor cuando acaba una frase

def _worker_audio():
    engine = pyttsx3.init()
    engine.setProperty("rate", 200)
    engine.setProperty("volume", 1.0)

    def _al_terminar(name, completed):
        utterance_terminada.set()

    engine.connect("finished-utterance", _al_terminar)

    while True:
        texto = cola_audio.get()
        if texto is None:      # señal de cierre
            break

        detener_solicitado.clear()
        utterance_terminada.clear()
        try:
            engine.say(texto)
            engine.startLoop(False)

            # Bucle externo (en vez de runAndWait) para poder
            # revisar en cada vuelta si hay que cortar el audio.
            inicio = time.time()
            while not utterance_terminada.is_set():
                if detener_solicitado.is_set():
                    engine.stop()
                engine.iterate()
                time.sleep(0.01)
                if time.time() - inicio > 20:  # salvaguarda anti-cuelgue
                    break

            engine.endLoop()
        except Exception as e:
            print("Aviso: error en el motor de audio:", e)  # no matar el hilo
        finally:
            hablando.clear()
            cola_audio.task_done()

hilo_audio = threading.Thread(target=_worker_audio, daemon=True)
hilo_audio.start()

def hablar(texto):
    """Encola el texto para reproducirlo en el hilo de audio, sin
    bloquear el bucle principal."""
    hablando.set()
    cola_audio.put(texto)

def detener_audio():
    """Corta el audio en seco (al separar los dedos) y vacía la cola
    de textos pendientes. Solo activa una bandera; quien detiene el
    motor de voz de verdad es el propio hilo de audio."""
    while True:
        try:
            cola_audio.get_nowait()
            cola_audio.task_done()
        except queue.Empty:
            break

    detener_solicitado.set()
    hablando.clear()

# ============================================================
# SELECCIÓN DE CÁMARA POR NOMBRE
# ============================================================
# OpenCV solo identifica cámaras por índice numérico, que puede
# cambiar entre reinicios. 'pygrabber' (pip install pygrabber) lee
# los nombres reales en Windows para elegir siempre la correcta;
# si no está instalada, se cae de vuelta al índice 0.

NOMBRE_CAMARA_DESEADA = "WN CAM-K211L"  # nombre tal cual aparece en la app Cámara

def obtener_indice_por_nombre(nombre_buscado):
    """Devuelve el índice de OpenCV de la cámara cuyo nombre contiene
    'nombre_buscado' (sin distinguir mayúsculas). None si no la
    encuentra o si 'pygrabber' no está instalado."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print(
            "Aviso: no está instalado 'pygrabber' (pip install pygrabber). "
            "No se puede buscar la cámara por nombre; se usará el índice 0."
        )
        return None

    dispositivos = FilterGraph().get_input_devices()  # mismo orden que los índices de cv2

    print("Cámaras detectadas:")
    for indice, nombre in enumerate(dispositivos):
        print(f"  [{indice}] {nombre}")

    for indice, nombre in enumerate(dispositivos):
        if nombre_buscado.lower() in nombre.lower():
            return indice

    return None

indice_camara = obtener_indice_por_nombre(NOMBRE_CAMARA_DESEADA)
if indice_camara is None:
    print(
        f"No se encontró ninguna cámara con el nombre '{NOMBRE_CAMARA_DESEADA}'. "
        f"Usando la cámara por defecto (índice 0)."
    )
    indice_camara = 0
else:
    print(f"Usando la cámara '{NOMBRE_CAMARA_DESEADA}' (índice {indice_camara}).")

# ============================================================
# MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(indice_camara, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"No se pudo abrir la cámara en el índice {indice_camara}.")

# ============================================================
# PUNTOS DE LA GRÁFICA
# ============================================================
puntos = {
    "Enero": {
        "x": 80,
        "y": 350,
        "valor": 11},
    "Febrero": {
        "x": 120,
        "y": 330,
        "valor": 13},
    "Marzo": {
        "x": 160,
        "y": 330,
        "valor": 13},
    "Abril": {
        "x": 200,
        "y": 280,
        "valor": 17},
    "Mayo": {
        "x": 240,
        "y": 250,
        "valor": 20},
    "Junio": {
        "x": 280,
        "y": 220,
        "valor": 23},
    "Julio": {
        "x": 320,
        "y": 180,
        "valor": 26},
    "Agosto": {
        "x": 360,
        "y": 170,
        "valor": 27},
    "Septiembre": {
        "x": 400,
        "y": 200,
        "valor": 25},
    "Octubre": {
        "x": 440,
        "y": 240,
        "valor": 21},
    "Noviembre": {
        "x": 480,
        "y": 290,
        "valor": 17},
    "Diciembre": {
        "x": 520,
        "y": 330,
        "valor": 14}
}

# ============================================================
# ESCALADO DE PUNTOS AL ESPACIO DE LA HOJA A5
# ============================================================
# 'puntos' está diseñado sobre un lienzo de referencia. Para que
# coincida con la hoja A5 física de la cámara, se reescala hacia el
# rectángulo (HOJA_X/Y/ANCHO/ALTO) donde aparece la hoja en el frame.

ANCHO_REFERENCIA = 600   # ancho del lienzo original de 'puntos'
ALTO_REFERENCIA = 400    # alto del lienzo original de 'puntos'

# Rectángulo (px, sobre el frame ya volteado) donde está la hoja A5.
# Ancho y alto se miden a ojo (calibración abajo), no por fórmula:
# el ángulo/distancia de la cámara distorsiona la proporción real.
HOJA_X = 104
HOJA_Y = 6
HOJA_ANCHO = 436
HOJA_ALTO = 306

# ============================================================
# MODO DE CALIBRACIÓN
# ============================================================
# MODO_CALIBRACION=True: clic esq. sup-izq y luego inf-der de la
# hoja -> imprime HOJA_X/Y/ANCHO/ALTO listos para copiar y dibuja
# el rectángulo en amarillo para verificarlo. 'r' reinicia clics,
# 'c' activa/desactiva el modo.
MODO_CALIBRACION = True

_calibracion_clics = []


def _clic_calibracion(evento, x, y, flags, param):
    """Callback de mouse para la calibración (ver arriba)."""
    if not MODO_CALIBRACION or evento != cv2.EVENT_LBUTTONDOWN:
        return

    _calibracion_clics.append((x, y))

    if len(_calibracion_clics) == 1:
        print(f"[Calibración] Esquina superior izquierda: x={x}, y={y}")
    elif len(_calibracion_clics) == 2:
        global HOJA_X, HOJA_Y, HOJA_ANCHO, HOJA_ALTO, puntos, segmentos

        (x1, y1), (x2, y2) = _calibracion_clics
        nuevo_x, nuevo_y = min(x1, x2), min(y1, y2)
        nuevo_ancho, nuevo_alto = abs(x2 - x1), abs(y2 - y1)
        print(
            "\n[Calibración] Copia esto en el script:\n"
            f"HOJA_X = {nuevo_x}\n"
            f"HOJA_Y = {nuevo_y}\n"
            f"HOJA_ANCHO = {nuevo_ancho}\n"
            f"HOJA_ALTO = {nuevo_alto}\n"
        )

        # Actualizar las variables de verdad para esta misma corrida:
        # así la hoja, los puntos y las líneas se recalculan al vuelo,
        # sin tener que copiar/pegar y reiniciar el script.
        HOJA_X, HOJA_Y, HOJA_ANCHO, HOJA_ALTO = (
            nuevo_x,
            nuevo_y,
            nuevo_ancho,
            nuevo_alto,
        )
        puntos = escalar_puntos_a_hoja_a5(
            PUNTOS_ORIGINALES,
            hoja_x=HOJA_X,
            hoja_y=HOJA_Y,
            hoja_ancho=HOJA_ANCHO,
            hoja_alto=HOJA_ALTO,
        )
        segmentos = construir_segmentos(puntos)
        print("[Calibración] Variables actualizadas en esta corrida.\n")


def _dibujar_calibracion(frame):
    """Dibuja los clics y el rectángulo resultante (amarillo)."""
    for punto in _calibracion_clics:
        cv2.circle(frame, punto, 5, (0, 255, 255), -1)

    if len(_calibracion_clics) == 2:
        (x1, y1), (x2, y2) = _calibracion_clics
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.putText(
        frame,
        "CALIBRACION: clic esq. sup-izq y inf-der | 'r' reiniciar | 'c' salir",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

def escalar_puntos_a_hoja_a5(
    puntos_originales,
    ancho_referencia=ANCHO_REFERENCIA,
    alto_referencia=ALTO_REFERENCIA,
    hoja_x=HOJA_X,
    hoja_y=HOJA_Y,
    hoja_ancho=HOJA_ANCHO,
    hoja_alto=HOJA_ALTO,
):
    """Reescala 'puntos_originales' del lienzo de referencia al
    rectángulo de la hoja A5, sin deformar (un solo factor de
    escala) y centrado dentro de ese rectángulo. Devuelve un
    diccionario nuevo; no modifica el original."""
    escala_x = hoja_ancho / ancho_referencia
    escala_y = hoja_alto / alto_referencia
    escala = min(escala_x, escala_y)

    ancho_escalado = ancho_referencia * escala
    alto_escalado = alto_referencia * escala
    margen_x = hoja_x + (hoja_ancho - ancho_escalado) / 2
    margen_y = hoja_y + (hoja_alto - alto_escalado) / 2

    puntos_escalados = {}
    for nombre, punto in puntos_originales.items():
        puntos_escalados[nombre] = {
            "x": int(margen_x + punto["x"] * escala),
            "y": int(margen_y + punto["y"] * escala),
            "valor": punto["valor"],
        }
    return puntos_escalados

# Se conserva sin escalar (sobre el lienzo de referencia) para poder
# volver a escalar en caliente cada vez que la calibración cambia.
PUNTOS_ORIGINALES = puntos
puntos = escalar_puntos_a_hoja_a5(puntos)

# ============================================================
# SEGMENTOS DE LA LÍNEA
# ============================================================
def construir_segmentos(puntos_escalados):
    """Construye la lista de segmentos (tramos entre meses
    consecutivos) a partir de 'puntos_escalados'. Se separó en
    función para poder reconstruirla cada vez que la calibración
    cambia los puntos en caliente."""
    segmentos = []
    nombres = list(puntos_escalados.keys())

    for i in range(len(nombres) - 1):
        nombre1 = nombres[i]
        nombre2 = nombres[i + 1]

        p1 = puntos_escalados[nombre1]
        p2 = puntos_escalados[nombre2]

        valor1 = p1["valor"]
        valor2 = p2["valor"]

        # Determinar tendencia
        if valor2 > valor1:
            tendencia = "aumento"
        elif valor2 < valor1:
            tendencia = "disminución"
        else:
            tendencia = "estable"

        segmentos.append({
            "inicio": nombre1,
            "fin": nombre2,

            "x1": p1["x"],
            "y1": p1["y"],

            "x2": p2["x"],
            "y2": p2["y"],

            "tendencia": tendencia,
            "valor_inicial": valor1,
            "valor_final": valor2
        })

    return segmentos

segmentos = construir_segmentos(puntos)

# ============================================================
# DISTANCIA ENTRE DOS PUNTOS
# ============================================================
def distancia(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1)**2 +
        (y2 - y1)**2)

# ============================================================
# DISTANCIA DE UN PUNTO A UN SEGMENTO
# ============================================================
def distancia_segmento(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # Si el segmento tiene longitud 0
    if dx == 0 and dy == 0:
        return distancia(px, py, x1, y1)

    # Proyección del punto sobre el segmento
    t = (
        (px - x1) * dx +
        (py - y1) * dy
    ) / (dx * dx + dy * dy)

    # Limitar al segmento
    t = max(0, min(1, t))

    # Punto más cercano del segmento
    cercano_x = x1 + t * dx
    cercano_y = y1 + t * dy

    return distancia(
        px,
        py,
        cercano_x,
        cercano_y)

elemento_anterior = None     # ELEMENTO ANTERIOR
audio_activado = False         # Solo se reproduce audio con la pinza activa (medio + índice)
pinza_activa_antes = False     # Estado de la pinza en el frame anterior (para detectar cuándo se abre)
DISTANCIA_PINZA = 35           # Umbral en píxeles para detectar medio + índice juntos

# ============================================================
# MEDIAPIPE
# ============================================================
cv2.namedWindow("Grafica interactiva")
if MODO_CALIBRACION:
    cv2.setMouseCallback("Grafica interactiva", _clic_calibracion)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        if indice_camara == 0:
            frame = cv2.flip(frame, 1)     # Espejo en Eje Y
        else:
            frame = cv2.flip(frame, -1)     # Espejo en Eje Y y X
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # BGR → RGB
        results = hands.process(frame_rgb)     # Procesar mano
        elemento_actual = None

        # ====================================================
        # DETECTAR MANO
        # ====================================================
        if results.multi_hand_landmarks is None:
            # Si la mano sale de cámara, es como soltar la pinza:
            # no debe quedar "activado" el audio de forma indefinida.
            audio_activado = False
        else:

            for hand_landmarks in results.multi_hand_landmarks:
                # Landmark 12 = punta del dedo medio (gesto de "pinza"
                # se realiza ahora entre dedo medio e índice)
                # Landmark 8 = punta del índice
                medio = hand_landmarks.landmark[12]
                dedo = hand_landmarks.landmark[8]

                medio_x = int(medio.x * width)
                medio_y = int(medio.y * height)
                x = int(dedo.x * width)
                y = int(dedo.y * height)

                # Detectar "pinza": punta del dedo medio junto a punta del índice.
                distancia_pinza = distancia(
                    medio_x, medio_y,
                    x, y
                )
                audio_activado = distancia_pinza <= DISTANCIA_PINZA

                # Dibujar dedo medio e índice y la línea que los une.
                cv2.circle(
                    frame,
                    (medio_x, medio_y),
                    8,
                    (255, 0, 0),
                    -1
                )
                cv2.circle(
                    frame,
                    (x, y),
                    8,
                    (255, 0, 0),
                    -1
                )
                cv2.line(
                    frame,
                    (medio_x, medio_y),
                    (x, y),
                    (255, 0, 0),
                    2
                )

                # ====================================================
                # 1. BUSCAR SI ESTÁ SOBRE UN PUNTO
                # ====================================================

                for nombre, punto in puntos.items():
                    d = distancia(
                        x,
                        y,
                        punto["x"],
                        punto["y"])

                    if d <= 20:
                        elemento_actual = (
                            "punto",
                            nombre)
                        break

                # ====================================================
                # 2. BUSCAR SI ESTÁ SOBRE UNA LÍNEA
                # ====================================================

                if elemento_actual is None:
                    for segmento in segmentos:
                        d = distancia_segmento(
                            x,
                            y,
                            segmento["x1"],
                            segmento["y1"],

                            segmento["x2"],
                            segmento["y2"])

                        if d <= 15:
                            elemento_actual = (
                                "tendencia",
                                segmento)
                            break

        # ====================================================
        # CAMBIO DE ELEMENTO
        # ====================================================
        # Convertimos el segmento en un identificador sencillo
        # para poder compararlo con el elemento anterior.

        if elemento_actual is not None:
            tipo = elemento_actual[0]
            if tipo == "punto":
                identificador = (
                    "punto",
                    elemento_actual[1])
            else:
                segmento = elemento_actual[1]
                identificador = (
                    "tendencia",
                    segmento["inicio"],
                    segmento["fin"])
        else:
            identificador = None

        # ====================================================
        # REPRODUCIR SONIDO
        # ====================================================
        # El audio solo empieza con la pinza activa, y no se encola
        # uno nuevo si ya hay uno sonando (evita solapes), sin
        # frenar la cámara ni la detección de landmarks.
        if audio_activado and identificador != elemento_anterior and not hablando.is_set():
            if elemento_actual is not None:
                tipo = elemento_actual[0]

                # --------------------------------------------
                # PUNTO
                # --------------------------------------------
                if tipo == "punto":
                    nombre = elemento_actual[1]
                    valor = puntos[nombre]["valor"]
                    texto = f"{nombre}. Temperatura de {valor} grados Celsius."
                    print(
                        nombre,
                        "=",
                        valor,
                        "°C")
                    hablar(texto)

                # --------------------------------------------
                # TENDENCIA
                # --------------------------------------------
                elif tipo == "tendencia":
                    segmento = elemento_actual[1]

                    inicio = segmento["inicio"]
                    fin = segmento["fin"]
                    valor_inicio = segmento["valor_inicial"]
                    valor_fin = segmento["valor_final"]

                    tendencia = segmento["tendencia"]
                    texto = (f"De {inicio} a {fin}, "
                             f"la temperatura presenta una tendencia de {tendencia}. "
                             f"Pasa de {valor_inicio} a {valor_fin} grados Celsius.")

                    print(
                        segmento["inicio"],
                        "→",
                        segmento["fin"],
                        ":",
                        tendencia)

                    hablar(texto)

                # Solo se marca como "narrado" si el audio se encoló
                # de verdad; si estaba ocupado, se reintentará al
                # terminar el audio en curso.
                elemento_anterior = identificador

        if not audio_activado:
            # Pinza soltada (o mano fuera de cámara): corta el audio
            # en seco y olvida lo ya narrado.
            if pinza_activa_antes:
                detener_audio()
            elemento_anterior = None

        pinza_activa_antes = audio_activado

        # ====================================================
        # DIBUJAR PUNTOS
        # ====================================================
        for nombre, punto in puntos.items():
            cv2.circle(
                frame,
                (punto["x"], punto["y"]),
                6,
                (0, 0, 255),
                -1)

        # ====================================================
        # DIBUJAR LÍNEAS
        # ====================================================
        for segmento in segmentos:
            cv2.line(
                frame,
                (segmento["x1"],
                 segmento["y1"]),

                (segmento["x2"],
                 segmento["y2"]),

                (0, 255, 0),
                2
            )

        # Indicador visual del estado de la pinza.
        estado_pinza = "AUDIO ACTIVO" if audio_activado else "Junta medio + indice"
        cv2.putText(
            frame,
            estado_pinza,
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if audio_activado else (255, 255, 255),
            2
        )

        # ====================================================
        # MOSTRAR INFORMACIÓN
        # ====================================================
        if elemento_actual is not None:
            tipo = elemento_actual[0]

            if tipo == "punto":
                nombre = elemento_actual[1]
                texto = (
                    nombre +
                    ": " +
                    str(puntos[nombre]["valor"]) +
                    " °C")
            else:
                segmento = elemento_actual[1]
                texto = (
                    segmento["inicio"] +
                    " → " +
                    segmento["fin"] +
                    ": " +
                    segmento["tendencia"])

            cv2.putText(
                frame,
                texto,
                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,

                (0, 0, 255),
                2
            )

        # ====================================================
        # MOSTRAR
        # ====================================================
        if MODO_CALIBRACION:
            _dibujar_calibracion(frame)

        cv2.imshow("Grafica interactiva", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        if tecla == ord("r"):  # reiniciar clics de calibración
            _calibracion_clics.clear()
            print("[Calibración] Clics reiniciados.")
        if tecla == ord("c"):  # activar/desactivar modo calibración
            MODO_CALIBRACION = not MODO_CALIBRACION
            _calibracion_clics.clear()
            print(f"[Calibración] Modo {'activado' if MODO_CALIBRACION else 'desactivado'}.")

cap.release()
cv2.destroyAllWindows()