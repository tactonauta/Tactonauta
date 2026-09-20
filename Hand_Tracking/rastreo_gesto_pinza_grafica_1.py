import cv2
import mediapipe as mp
import math
import threading
import queue
import time
###import winsound

import pyttsx3

# ============================================================
# AUDIO EN SEGUNDO PLANO
# ============================================================
# pyttsx3.runAndWait() es una llamada BLOQUEANTE: si se ejecuta
# dentro del bucle principal, la cámara y la detección de la mano
# se "congelan" hasta que termina de hablar.
#
# Además, reutilizar el mismo motor llamando a runAndWait() una y
# otra vez es conocido por dejar de funcionar a partir de la
# segunda frase (sobre todo con el driver SAPI5 de Windows): habla
# la primera vez y luego se queda "mudo" sin dar ningún error.
#
# Solución: un hilo dedicado que consume una cola de textos, y que
# en vez de runAndWait() usa el "bucle externo" que la propia
# documentación de pyttsx3 recomienda para tener control fino
# (startLoop(False) + iterate() + endLoop()). Así podemos revisar
# en cada vuelta si se pidió detener el audio, y cortarlo desde
# DENTRO de este mismo hilo (nunca desde el principal).

cola_audio = queue.Queue()
hablando = threading.Event()            # True mientras se está reproduciendo audio
detener_solicitado = threading.Event()  # Se activa desde el hilo principal para pedir el corte
utterance_terminada = threading.Event() # La marca el propio motor cuando acaba una frase
motor_audio = None                      # referencia al engine, se asigna dentro del hilo


def _worker_audio():
    global motor_audio
    engine = pyttsx3.init()
    engine.setProperty("rate", 200)
    engine.setProperty("volume", 1.0)

    def _al_terminar(name, completed):
        utterance_terminada.set()

    engine.connect("finished-utterance", _al_terminar)
    motor_audio = engine

    while True:
        texto = cola_audio.get()
        if texto is None:      # señal de cierre
            break

        detener_solicitado.clear()
        utterance_terminada.clear()
        try:
            engine.say(texto)
            engine.startLoop(False)

            # Bucle externo: vamos "empujando" al motor nosotros
            # mismos, en vez de bloquear con runAndWait(). Esto nos
            # deja revisar en cada vuelta si hay que cortar el audio.
            inicio = time.time()
            while not utterance_terminada.is_set():
                if detener_solicitado.is_set():
                    engine.stop()
                engine.iterate()
                time.sleep(0.01)
                # Salvaguarda: si algo se queda colgado más de 20s,
                # no dejamos el hilo bloqueado para siempre.
                if time.time() - inicio > 20:
                    break

            engine.endLoop()
        except Exception as e:
            # Si algo falla, NO dejamos morir el hilo: si no,
            # nunca más se reproduciría audio en toda la ejecución.
            print("Aviso: error en el motor de audio:", e)
        finally:
            hablando.clear()
            cola_audio.task_done()


hilo_audio = threading.Thread(target=_worker_audio, daemon=True)
hilo_audio.start()


def hablar(texto):
    """Encola el texto para reproducirlo en el hilo de audio.

    No bloquea el bucle principal: la marca 'hablando' se activa
    aquí mismo (de forma síncrona) para que, aunque el hilo todavía
    no haya recogido el texto de la cola, no se puedan encolar dos
    audios al mismo tiempo.
    """
    hablando.set()
    cola_audio.put(texto)


def detener_audio():
    """Pide cortar el audio en seco (al separar pulgar e índice) y
    vacía cualquier texto pendiente en la cola, para que no se
    reproduzca nada "atrasado" después de soltar la pinza.

    Esta función solo VACÍA la cola y ACTIVA una bandera: quien
    realmente detiene el motor de voz es el propio hilo de audio,
    revisando esa bandera dentro de su bucle (nunca este hilo)."""
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
# OpenCV, por sí solo, no identifica cámaras por nombre: solo usa
# índices numéricos (0, 1, 2...) y ese número puede variar entre
# reinicios o al conectar/desconectar dispositivos USB. Para elegir
# SIEMPRE la cámara correcta (por ejemplo, tu webcam "WN CAM-K211L"
# y no la webcam integrada de la laptop), usamos la librería
# 'pygrabber', que en Windows lee los nombres reales de los
# dispositivos DirectShow — los mismos que muestra la app Cámara.
#
#   pip install pygrabber
#
# Si no está instalada, el programa avisa y cae de vuelta al
# índice 0 (comportamiento anterior), para no dejar de funcionar.

NOMBRE_CAMARA_DESEADA = "WN CAM-K211L"  # nombre tal cual aparece en la app Cámara


def obtener_indice_por_nombre(nombre_buscado):
    """Busca, entre las cámaras conectadas, el índice de OpenCV que
    corresponde al dispositivo cuyo nombre contiene 'nombre_buscado'
    (sin distinguir mayúsculas/minúsculas). Devuelve None si no la
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
# SEGMENTOS DE LA LÍNEA
# ============================================================

segmentos = []
nombres = list(puntos.keys())

for i in range(len(nombres) - 1):
    nombre1 = nombres[i]
    nombre2 = nombres[i + 1]

    p1 = puntos[nombre1]
    p2 = puntos[nombre2]

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


# ============================================================
# SONIDOS
# ============================================================

sonidos_puntos = {
    "Enero": 500,
    "Febrero": 550,
    "Marzo": 600,
    "Abril": 650,
    "Mayo": 700,
    "Junio": 750,
    "Julio": 800,
    "Agosto": 850,
    "Septiembre": 900,
    "Octubre": 950,
    "Noviembre": 1000,
    "Diciembre": 1050
}

sonidos_tendencia = {
    "aumento": 1200,
    "disminución": 400,
    "estable": 800}


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
audio_activado = False         # Solo se reproduce audio mientras pulgar e índice están juntos
pinza_activa_antes = False     # Estado de la pinza en el frame anterior (para detectar cuándo se abre)
DISTANCIA_PINZA = 35           # Umbral en píxeles para detectar pulgar + índice

# ============================================================
# MEDIAPIPE
# ============================================================
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
                # Landmark 4 = punta del pulgar
                # Landmark 8 = punta del índice
                pulgar = hand_landmarks.landmark[4]
                dedo = hand_landmarks.landmark[8]

                pulgar_x = int(pulgar.x * width)
                pulgar_y = int(pulgar.y * height)
                x = int(dedo.x * width)
                y = int(dedo.y * height)

                # Detectar "pinza": punta del pulgar junto a punta del índice.
                distancia_pinza = distancia(
                    pulgar_x, pulgar_y,
                    x, y
                )
                audio_activado = distancia_pinza <= DISTANCIA_PINZA

                # Dibujar pulgar e índice y la línea que los une.
                cv2.circle(
                    frame,
                    (pulgar_x, pulgar_y),
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
                    (pulgar_x, pulgar_y),
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
        # El audio SOLO comienza cuando pulgar e índice están juntos (pinza).
        # Mover el índice sobre un punto o línea sin hacer la pinza no reproduce nada.
        # Además, mientras ya se está reproduciendo un audio (hablando.is_set())
        # no se encola uno nuevo: así se evita audio superpuesto/cortado, pero la
        # cámara y la detección de landmarks NO se detienen en ningún momento,
        # porque hablar() ya no bloquea el bucle.
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

                    # frecuencia = sonidos_puntos[nombre]
                    # winsound.Beep(
                    #     frecuencia,
                    #     1000)  ##milisegundos
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
                    
                    # frecuencia = sonidos_tendencia[tendencia]
                    # winsound.Beep(
                    #     frecuencia,
                    #     1500)  ##milisegundos
                    
                    print(
                        segmento["inicio"],
                        "→",
                        segmento["fin"],
                        ":",
                        tendencia)

                    hablar(texto)

                # Solo marcamos este elemento como "ya narrado" cuando el
                # audio realmente se encoló. Si estaba ocupado (hablando
                # el punto/tendencia anterior), NO actualizamos aquí: así,
                # en cuanto termine el audio en curso, este mismo elemento
                # se narrará (mientras la pinza se mantenga sobre él).
                elemento_anterior = identificador

        if not audio_activado:
            # La pinza se acaba de soltar (o la mano salió de cámara):
            # cortamos el audio en seco, sin esperar a que termine la
            # frase, y reiniciamos la detección de regiones como al
            # principio (nada queda "recordado" como ya narrado).
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
        estado_pinza = "AUDIO ACTIVO" if audio_activado else "Junta pulgar + indice"
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
        cv2.imshow("Grafica interactiva", frame)

        # ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()