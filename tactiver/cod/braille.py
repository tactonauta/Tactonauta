# ============================================================
# TACTONAUTA
# Módulo para manejo de texto Braille español
# y detección de etiquetas demasiado largas
#
# Basado en "braille2" (rama Selección_Gráficas). Es la única tabla Braille
# del proyecto: generador_stl.py la importa de acá.
# ============================================================

import unicodedata


# ============================================================
# 1. MEDIDAS (mm)
# generador_stl.py las importa de acá, así no pueden desincronizarse.
# ============================================================

ESPACIADO_BRAILLE = 2.4     # entre centros de dos puntos de la misma celda
DIAM_PUNTO_BRAILLE = 1.5
CELDA_PITCH = 6.2           # entre centros de dos celdas seguidas


# ============================================================
# 2. LETRAS BRAILLE
# ============================================================

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

# Letras propias del español: tienen su propio signo (no se les quita la
# tilde: "año" no es "ano").
LETRAS_ESPANOL = {
    "á": [1, 2, 3, 5, 6],
    "é": [2, 3, 4, 6],
    "í": [3, 4],
    "ó": [3, 4, 6],
    "ú": [2, 3, 4, 5, 6],
    "ü": [1, 2, 5, 6],
    "ñ": [1, 2, 4, 5, 6],
}


# ============================================================
# 3. TABLA BRAILLE
# ============================================================

BRAILLE = dict(LETRAS)
BRAILLE.update(LETRAS_ESPANOL)

# Signos especiales
BRAILLE["numeral"] = [3, 4, 5, 6]

# Signo de mayúscula en Braille español
BRAILLE["mayuscula"] = [4, 6]

# Signos usados por el proyecto
BRAILLE["menos"] = [3, 6]
# Coma decimal (punto 2): "12,5" = numeral 1 2 , 5. La clave se llama
# "punto" por compatibilidad con el generador.
BRAILLE["punto"] = [2]


# ============================================================
# 4. NÚMEROS EN BRAILLE ESPAÑOL
#
# En Braille español los números utilizan las formas
# de las letras a-j después del signo numeral.
#
# 1=a, 2=b, ..., 9=i, 0=j
# ============================================================

DIGITOS = {
    "1": LETRAS["a"],
    "2": LETRAS["b"],
    "3": LETRAS["c"],
    "4": LETRAS["d"],
    "5": LETRAS["e"],
    "6": LETRAS["f"],
    "7": LETRAS["g"],
    "8": LETRAS["h"],
    "9": LETRAS["i"],
    "0": LETRAS["j"],
}

BRAILLE.update(DIGITOS)


# ============================================================
# 5. LIMPIAR UNA LETRA
#
# Quita tildes de las letras que no tienen signo propio en la tabla
# (à, ç, ô...), para poder encontrar su letra base.
# Las del español (á, é, í, ó, ú, ü, ñ) se conservan.
# ============================================================

def quitar_tilde(caracter):

    if caracter.lower() in LETRAS_ESPANOL:
        return caracter

    normalizado = unicodedata.normalize("NFKD", caracter)

    sin_tilde = "".join(
        c
        for c in normalizado
        if not unicodedata.combining(c)
    )

    return sin_tilde


# ============================================================
# 6. CONVERTIR TEXTO A CELDAS BRAILLE
#
# Ejemplo:
#
# "hola"   -> ["h", "o", "l", "a"]
# "A"      -> ["mayuscula", "a"]
# "A2"     -> ["mayuscula", "a", "numeral", "2"]
# "PIB"    -> ["mayuscula", "mayuscula", "p", "i", "b"]
#             (palabra entera en mayúsculas: doble signo, una sola vez)
# "12,5"   -> ["numeral", "1", "2", "punto", "5"]
#
# Lo que no está en la tabla (puntuación, símbolos) se omite en vez de
# dejar una celda vacía.
# ============================================================

def _palabra_en_mayusculas(texto, inicio):
    """True si la palabra que empieza en `inicio` tiene 2+ letras y todas
    son mayúsculas."""
    letras = []
    for c in texto[inicio:]:
        if c == " ":
            break
        if c.isalpha():
            letras.append(c)
    return len(letras) >= 2 and all(c.isupper() for c in letras)


def texto_a_celdas(texto):

    texto = " ".join(str(texto).split())

    celdas = []

    anterior_digito = False
    palabra_mayus = False      # dentro de una palabra toda en mayúsculas

    for i, caracter_original in enumerate(texto):

        # ----------------------------------------
        # ESPACIOS
        # ----------------------------------------

        if caracter_original == " ":

            if celdas and celdas[-1] != " ":
                celdas.append(" ")

            anterior_digito = False
            palabra_mayus = False
            continue


        # ----------------------------------------
        # NÚMEROS
        # ----------------------------------------

        if caracter_original.isdigit():

            # Solo agregamos un signo numeral
            # al inicio de cada grupo de números.

            if not anterior_digito:
                celdas.append("numeral")

            celdas.append(caracter_original)

            anterior_digito = True
            continue

        # Coma o punto ENTRE dígitos: separador decimal (o de miles), sigue
        # siendo el mismo número.
        if (caracter_original in ",." and anterior_digito
                and i + 1 < len(texto) and texto[i + 1].isdigit()):
            celdas.append("punto")
            continue

        anterior_digito = False


        # ----------------------------------------
        # LETRAS
        # ----------------------------------------

        caracter = quitar_tilde(caracter_original)

        if not caracter:
            continue

        # ¿La letra original era mayúscula?
        es_mayuscula = caracter_original.isupper()

        caracter = caracter.lower()


        if caracter in BRAILLE and caracter.isalpha():

            # Si era mayúscula agregamos primero el signo de mayúscula; si
            # es toda la palabra, doble signo al principio y nada más.

            if es_mayuscula and not palabra_mayus:
                if (i == 0 or texto[i - 1] == " ") and _palabra_en_mayusculas(texto, i):
                    celdas += ["mayuscula", "mayuscula"]
                    palabra_mayus = True
                else:
                    celdas.append("mayuscula")

            celdas.append(caracter)


    # Evitamos espacios sobrantes al final

    while celdas and celdas[-1] == " ":
        celdas.pop()


    return celdas


# ============================================================
# 7. CONVERTIR CELDAS A PUNTOS
#
# ["mayuscula", "a"] -> [[4, 6], [1]]
# ============================================================

def celdas_a_puntos(celdas):

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
# 8. CALCULAR EL ANCHO FÍSICO DEL BRAILLE
#
# De borde a borde de los puntos, con las mismas medidas del generador.
# ============================================================

def ancho_braille(celdas):

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
# 9. COMPROBAR SI UN TEXTO CABE
# ============================================================

def texto_cabe(texto, ancho_disponible):

    celdas = texto_a_celdas(texto)

    ancho = ancho_braille(celdas)

    return ancho <= ancho_disponible


# ============================================================
# 10. CREAR IDENTIFICADORES PARA LA LEYENDA
#
# A, B, C, ... Z, y si hubiera más: A1, A2, ...
# ============================================================

def crear_identificador(indice):

    alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if indice < len(alfabeto):
        return alfabeto[indice]

    # Por seguridad, si hubiera más de 26 etiquetas
    # usamos A1, A2, etc.

    return "A" + str(indice - 25)


# ============================================================
# 11. ABREVIAR ETIQUETAS QUE NO CABEN
#
# Una etiqueta que no entra en su lugar se reemplaza por un identificador
# (A, B, C...) y el texto completo va a la leyenda. El Abreviador lleva la
# cuenta entre llamadas, así todas las etiquetas de una lámina (títulos,
# categorías del eje...) comparten la misma serie de letras, y un mismo
# texto repetido usa siempre la misma letra.
# ============================================================

class Abreviador:

    def __init__(self):
        self.leyenda = []
        self._por_texto = {}

    def procesar(self, texto, ancho_disponible, donde=None, clave=None):
        """Dict con lo que va físicamente en la gráfica (ver
        procesar_etiquetas). `donde` (opcional) dice qué es la etiqueta
        ("título del eje X"...), para la leyenda y la narración; `clave`
        (opcional) identifica el elemento de la lámina de donde salió."""

        celdas = texto_a_celdas(texto)

        ancho = ancho_braille(celdas)


        # ========================================
        # CASO 1: EL TEXTO SÍ CABE
        # ========================================

        if ancho <= ancho_disponible:

            return {

                "texto_original": texto,

                "texto_stl": texto,

                "celdas": celdas,

                "puntos": celdas_a_puntos(celdas),

                "ancho_mm": ancho,

                "usa_leyenda": False,

                "identificador": None

            }


        # ========================================
        # CASO 2: EL TEXTO NO CABE
        # ========================================

        clave = " ".join(str(texto).split())

        if clave in self._por_texto:

            entrada = self._por_texto[clave]

        else:

            identificador = crear_identificador(len(self.leyenda))

            # Convertimos también el identificador a Braille:
            # "A" -> ["mayuscula", "a"]
            celdas_id = texto_a_celdas(identificador)

            # Guardamos el texto completo para la leyenda.
            entrada = {

                "identificador": identificador,

                "texto": texto,

                "donde": donde,

                "clave": clave,

                "celdas_identificador": celdas_id,

                "puntos_identificador": celdas_a_puntos(celdas_id),

                "celdas_texto": celdas,

                "puntos_texto": celdas_a_puntos(celdas)

            }

            self.leyenda.append(entrada)
            self._por_texto[clave] = entrada


        # Lo que irá físicamente en la gráfica
        return {

            "texto_original": texto,

            "texto_stl": entrada["identificador"],

            "celdas": entrada["celdas_identificador"],

            "puntos": entrada["puntos_identificador"],

            "ancho_mm": ancho_braille(entrada["celdas_identificador"]),

            "usa_leyenda": True,

            "identificador": entrada["identificador"]

        }


# ============================================================
# 12. PROCESAR TODAS LAS ETIQUETAS
#
# Función principal del módulo original: todas las etiquetas con el mismo
# ancho disponible. Devuelve (etiquetas_procesadas, leyenda).
# ============================================================

def procesar_etiquetas(etiquetas, ancho_disponible):

    abreviador = Abreviador()

    etiquetas_procesadas = [
        abreviador.procesar(texto, ancho_disponible)
        for texto in etiquetas
    ]

    return etiquetas_procesadas, abreviador.leyenda


# ============================================================
# 13. PARTIR UN TEXTO LARGO EN RENGLONES
#
# Para la leyenda: el texto completo nunca se recorta, se sigue en el
# renglón de abajo (cortando en un espacio; una palabra más larga que un
# renglón entero se parte).
# ============================================================

def partir_en_renglones(celdas, max_celdas):

    max_celdas = max(1, int(max_celdas))

    palabras, actual = [], []

    for c in celdas:

        if c == " ":
            if actual:
                palabras.append(actual)
            actual = []

        else:
            actual.append(c)

    if actual:
        palabras.append(actual)


    renglones, renglon = [], []

    for palabra in palabras:

        while len(palabra) > max_celdas:
            # palabra más larga que un renglón: se parte
            if renglon:
                renglones.append(renglon)
                renglon = []
            renglones.append(palabra[:max_celdas])
            palabra = palabra[max_celdas:]

        if not palabra:
            continue

        if renglon and len(renglon) + 1 + len(palabra) <= max_celdas:
            renglon = renglon + [" "] + palabra

        elif renglon:
            renglones.append(renglon)
            renglon = list(palabra)

        else:
            renglon = list(palabra)

    if renglon:
        renglones.append(renglon)

    return renglones


# ============================================================
# 14. PRUEBA
#
# Esta parte solo se ejecuta cuando corres directamente braille.py.
# ============================================================

if __name__ == "__main__":

    print("\nTACTONAUTA - PRUEBA BRAILLE ESPAÑOL\n")

    etiquetas = [
        "Tiempo",
        "Temperatura",
        "Presión",
        "Temperatura promedio durante el experimento",
        "Grupo experimental con tratamiento",
    ]

    ancho_disponible = 80.0

    procesadas, leyenda = procesar_etiquetas(etiquetas, ancho_disponible)

    print("Ancho disponible:", ancho_disponible, "mm")

    print("\nETIQUETAS EN LA GRÁFICA\n")

    for etiqueta in procesadas:
        print(
            etiqueta["texto_original"], "->", etiqueta["texto_stl"],
            "| ancho:", round(etiqueta["ancho_mm"], 1), "mm",
            "| usa leyenda:", etiqueta["usa_leyenda"],
        )

    print("\nLEYENDA\n")

    if not leyenda:
        print("No fue necesario crear una leyenda.")

    else:
        for entrada in leyenda:
            print(entrada["identificador"], "=", entrada["texto"])
            print("  Braille identificador:", entrada["celdas_identificador"])
            print("  Puntos:", entrada["puntos_identificador"])
