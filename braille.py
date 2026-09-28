# ============================================================
# TACTONAUTA
# Módulo para manejo de texto Braille y etiquetas largas
# ============================================================

import unicodedata


# ------------------------------------------------------------
# CONSTANTES
# Deben coincidir con generador_stl.py
# ------------------------------------------------------------

ESPACIADO_BRAILLE = 2.4
DIAM_PUNTO_BRAILLE = 1.4
CELDA_PITCH = 6.2


# ------------------------------------------------------------
# ALFABETO BRAILLE
# Mismo alfabeto utilizado actualmente por generador_stl.py
# ------------------------------------------------------------

LETRAS = {
    "a": [1],
    "b": [1, 2],
    "c": [1, 4],
    "d": [1, 4, 5],
    "e": [1, 5],
    "f": [1, 2, 4],
    "g": [1, 2, 4, 5],
    "h": [1, 2, 5],
    "i": [2, 4],
    "j": [2, 4, 5],
    "k": [1, 3],
    "l": [1, 2, 3],
    "m": [1, 3, 4],
    "n": [1, 3, 4, 5],
    "o": [1, 3, 5],
    "p": [1, 2, 3, 4],
    "q": [1, 2, 3, 4, 5],
    "r": [1, 2, 3, 5],
    "s": [2, 3, 4],
    "t": [2, 3, 4, 5],
    "u": [1, 3, 6],
    "v": [1, 2, 3, 6],
    "w": [2, 4, 5, 6],
    "x": [1, 3, 4, 6],
    "y": [1, 3, 4, 5, 6],
    "z": [1, 3, 5, 6],
}


# ------------------------------------------------------------
# NÚMEROS
# Misma lógica Nemeth que usa actualmente Tactonauta
# ------------------------------------------------------------

DESPLAZAMIENTO_NEMETH = {
    1: 2,
    2: 3,
    4: 5,
    5: 6
}

ORDEN_DIGITOS = list("abcdefghij")


BRAILLE = dict(LETRAS)

BRAILLE["numeral"] = [3, 4, 5, 6]
BRAILLE["menos"] = [3, 6]
BRAILLE["punto"] = [4, 6]


for numero, letra in enumerate(ORDEN_DIGITOS, start=1):

    digito = str(numero % 10)

    BRAILLE[digito] = sorted(
        DESPLAZAMIENTO_NEMETH[punto]
        for punto in LETRAS[letra]
    )


# ============================================================
# 1. LIMPIAR TEXTO
# ============================================================

def sanear_texto(texto):

    """
    Convierte el texto a minúsculas y elimina tildes.
    Esto sigue la lógica actual de generador_stl.py.
    """

    texto = str(texto)

    sin_tildes = unicodedata.normalize(
        "NFKD",
        texto
    )

    sin_tildes = "".join(
        caracter
        for caracter in sin_tildes
        if not unicodedata.combining(caracter)
    )

    return sin_tildes.lower()


# ============================================================
# 2. CONVERTIR TEXTO A CELDAS BRAILLE
# ============================================================

def texto_a_celdas(texto):

    """
    Convierte un texto a las celdas Braille necesarias.

    Ejemplo:

    "cat"

    devuelve:

    ["c", "a", "t"]

    Luego generador_stl.py sabe que:

    c = [1,4]
    a = [1]
    t = [2,3,4,5]
    """

    texto = sanear_texto(texto)

    celdas = []

    anterior_digito = False


    for caracter in texto:

        # -------------------------
        # NÚMEROS
        # -------------------------

        if caracter.isdigit():

            # Antes del primer número se coloca
            # el indicador numérico.
            if not anterior_digito:
                celdas.append("numeral")

            celdas.append(caracter)

            anterior_digito = True


        # -------------------------
        # LETRAS
        # -------------------------

        elif caracter in LETRAS:

            celdas.append(caracter)

            anterior_digito = False


        # -------------------------
        # ESPACIOS
        # -------------------------

        elif caracter == " ":

            # Evitamos espacios repetidos.
            if celdas and celdas[-1] != " ":
                celdas.append(" ")

            anterior_digito = False


        # Otros símbolos se ignoran por ahora.
        else:

            anterior_digito = False


    # Evitar que termine con un espacio.
    while celdas and celdas[-1] == " ":
        celdas.pop()


    return celdas


# ============================================================
# 3. OBTENER LOS PUNTOS BRAILLE
# ============================================================

def celdas_a_puntos(celdas):

    """
    Convierte las celdas en los puntos que físicamente
    debe levantar generador_stl.py.
    """

    resultado = []


    for celda in celdas:

        if celda == " ":

            resultado.append([])

        else:

            resultado.append(
                BRAILLE.get(celda, [])
            )


    return resultado


# ============================================================
# 4. CALCULAR ANCHO FÍSICO DEL BRAILLE
# ============================================================

def ancho_braille(celdas):

    """
    Calcula el ancho REAL aproximado que ocupará
    el texto Braille dentro del STL.

    Utiliza las mismas medidas que generador_stl.py.
    """

    numero_celdas = len(celdas)


    if numero_celdas == 0:
        return 0.0


    ancho = (
        (numero_celdas - 1) * CELDA_PITCH
        + ESPACIADO_BRAILLE
        + DIAM_PUNTO_BRAILLE
    )


    return ancho


# ============================================================
# 5. COMPROBAR SI UN TEXTO CABE
# ============================================================

def texto_cabe(texto, ancho_disponible):

    """
    Devuelve True si el texto entra en el espacio
    disponible y False si no entra.
    """

    celdas = texto_a_celdas(texto)

    ancho = ancho_braille(celdas)

    return ancho <= ancho_disponible


# ============================================================
# 6. CREAR IDENTIFICADORES PARA TEXTOS LARGOS
# ============================================================

def crear_identificador(indice):

    """
    Genera identificadores cortos:

    0 -> a
    1 -> b
    2 -> c
    ...

    Estos identificadores pueden imprimirse
    dentro de la gráfica.

    El texto completo se guarda en la leyenda.
    """

    alfabeto = "abcdefghijklmnopqrstuvwxyz"

    if indice < len(alfabeto):
        return alfabeto[indice]

    # Si algún día existen más de 26 etiquetas:
    return "x" + str(indice + 1)


# ============================================================
# 7. PROCESAR UNA LISTA DE ETIQUETAS
# ============================================================

def procesar_etiquetas(etiquetas, ancho_disponible):

    """
    Decide automáticamente qué textos pueden escribirse
    completos y cuáles necesitan identificador + leyenda.

    Devuelve:

    - etiquetas_procesadas
    - leyenda
    """

    etiquetas_procesadas = []

    leyenda = []

    contador_leyenda = 0


    for texto in etiquetas:

        celdas = texto_a_celdas(texto)

        ancho = ancho_braille(celdas)


        # ====================================================
        # EL TEXTO CABE
        # ====================================================

        if ancho <= ancho_disponible:

            etiquetas_procesadas.append({

                "texto_original": texto,

                "texto_stl": texto,

                "celdas": celdas,

                "puntos": celdas_a_puntos(celdas),

                "ancho_mm": ancho,

                "usa_leyenda": False,

                "identificador": None

            })


        # ====================================================
        # EL TEXTO NO CABE
        # ====================================================

        else:

            identificador = crear_identificador(
                contador_leyenda
            )

            contador_leyenda += 1


            celdas_id = texto_a_celdas(
                identificador
            )


            etiquetas_procesadas.append({

                "texto_original": texto,

                "texto_stl": identificador,

                "celdas": celdas_id,

                "puntos": celdas_a_puntos(celdas_id),

                "ancho_mm": ancho_braille(celdas_id),

                "usa_leyenda": True,

                "identificador": identificador

            })


            # Guardamos la equivalencia para que
            # generador_stl.py pueda construir la leyenda.

            leyenda.append({

                "identificador": identificador,

                "texto": texto,

                "celdas_identificador": celdas_id,

                "celdas_texto": celdas

            })


    return etiquetas_procesadas, leyenda


# ============================================================
# PRUEBA DEL MÓDULO
# ============================================================

if __name__ == "__main__":

    print("\nTACTONAUTA - PRUEBA DE ETIQUETAS BRAILLE\n")


    etiquetas = [

        "Time",

        "Temperature",

        "Average temperature during experiment",

        "Pressure",

        "Experimental group with treatment"

    ]


    # Espacio disponible de prueba.
    ancho_disponible = 80.0


    procesadas, leyenda = procesar_etiquetas(
        etiquetas,
        ancho_disponible
    )


    print(
        "Ancho disponible:",
        ancho_disponible,
        "mm"
    )


    print("\nETIQUETAS EN LA GRÁFICA\n")


    for etiqueta in procesadas:

        print(
            etiqueta["texto_original"],
            "->",
            etiqueta["texto_stl"],
            "| ancho:",
            round(etiqueta["ancho_mm"], 1),
            "mm",
            "| leyenda:",
            etiqueta["usa_leyenda"]
        )


    print("\nLEYENDA\n")


    if not leyenda:

        print(
            "No fue necesario crear una leyenda."
        )


    else:

        for entrada in leyenda:

            print(
                entrada["identificador"],
                "=",
                entrada["texto"]
            )