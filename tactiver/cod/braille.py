# ============================================================
# TACTONAUTA
# Módulo para manejo de texto Braille español
# y detección de etiquetas demasiado largas
#
# Basado en "braille2" (rama Selección_Gráficas). Es la única tabla Braille
# del proyecto: generador_stl.py la importa de acá.
# ============================================================

import re
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


def _texto_literal_a_celdas(texto):

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
# 6b. ECUACIONES: CÓDIGO MATEMÁTICO UNIFICADO
#
# Las expresiones matemáticas ("y = 2x + 1", "R² = 0,98", "σ ≈ 1,5") se
# escriben con el Código Matemático Unificado para Iberoamérica (Consejo
# Iberoamericano del Braille, 2023):
# https://www.once.es/servicios-sociales/braille/consejo-iberoamericano/codigo-matematico-unificado-para-iberoamerica
# Antes, el texto literario omitía todo lo que no fuera letra o número:
# "y = 2x + 1" quedaba "y 2x 1".
#
# Cada signo, con la página del documento donde está:
#   + 235 · - 36 · × 236 · / 256 · ÷ y : 5-2 (pág. 55)
#   = 2356 · < 246 · > 135 · ≤ 246-2356 · ≥ 135-2356 (pág. 64-65)
#   ≠ 45-2356 (pág. 65) · ≈ 5-2356-3 (pág. 64)
#   ± 235-25-36 · % 456-356 (pág. 60) · ° 356 (pág. 95)
#   ( 126 · ) 345 · [ 12356 · ] 23456 · paréntesis auxiliares 26 ... 35 (pág. 23)
#   exponente (superíndice) 16 · subíndice 34 (pág. 46, 57)
#   raíz cuadrada 1246-156 + radicando (pág. 58)
#   mayúscula latina 46 + letra, una por letra (pág. 13)
#   minúscula a-j después de un número: punto 5 delante (pág. 12, 14)
#   griega minúscula 4 + letra · griega mayúscula 45 + letra (pág. 14-15)
#   coma decimal 2 (pág. 19)
# El punto de multiplicar (6 + medio cajetín en blanco) se escribe con el
# aspa (236): el significado es el mismo y no depende del cajetín vecino.
# ============================================================

def _celda(puntos):
    """Nombre de una celda por sus puntos ("p235") en la tabla BRAILLE."""
    nombre = "p" + puntos
    BRAILLE.setdefault(nombre, [int(c) for c in puntos])
    return nombre


def _signo(*celdas):
    return [_celda(c) for c in celdas]


SIGNOS_MATEMATICOS = {
    "+": _signo("235"), "-": _signo("36"), "−": _signo("36"), "–": _signo("36"),
    "×": _signo("236"), "·": _signo("236"), "*": _signo("236"), "⋅": _signo("236"),
    "/": _signo("256"), "÷": _signo("5", "2"), ":": _signo("5", "2"),
    "=": _signo("2356"), "<": _signo("246"), ">": _signo("135"),
    "≤": _signo("246", "2356"), "≥": _signo("135", "2356"),
    "≠": _signo("45", "2356"), "≈": _signo("5", "2356", "3"),
    "±": _signo("235", "25", "36"), "∓": _signo("36", "25", "235"),
    "%": _signo("456", "356"), "°": _signo("356"),
    "(": _signo("126"), ")": _signo("345"), "[": _signo("12356"), "]": _signo("23456"),
}
_EXPONENTE = _signo("16")
_SUBINDICE = _signo("34")
_AUX_ABRE, _AUX_CIERRA = _signo("26"), _signo("35")
_RAIZ = _signo("1246", "156")
_PREFIJO_MAYUSCULA = _celda("46")
_PREFIJO_MINUSCULA = _celda("5")
_PREFIJO_GRIEGA = _celda("4")
_PREFIJO_GRIEGA_MAYUSCULA = _celda("45")

# letra base de cada griega (pág. 14-15): casi todas, la latina parecida
_GRIEGAS = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": _celda("156"),
    "θ": _celda("1456"), "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "u", "φ": "f",
    "χ": _celda("12346"), "ψ": "y", "ω": "w",
}

_SUPERINDICES = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+-=()ni"))
_SUBINDICES = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓ", "0123456789+-=()aeox"))

# Lo que hace que un texto sea una ecuación (y no, por ejemplo, un guion
# entre palabras como "PEGMA-PEGDA" o un rango "1-3").
_SIGNOS_FUERTES = (set("=≠≈≤≥±∓√^×÷%°_") | set(_SUPERINDICES) | set(_SUBINDICES)
                   | set(_GRIEGAS) | {c.upper() for c in _GRIEGAS})
# un número, una letra suelta o una función de una letra o abreviada:
# "2", "-1,5", "2x", "y", "f(x)", "sen(x)", "(a"
_RE_OPERANDO = re.compile(
    r"^[(\[]*(?:[-−]?\d+(?:[.,]\d+)*[a-zA-Zα-ωΑ-Ω]?|[a-zA-Zα-ωΑ-Ω]|[a-zA-Z]{1,3}\([^()\s]{0,8}\))"
    r"[)\]]*[.,;]?$")
# un número seguido de una letra de la a a la j ("3a"): sin el código
# matemático se leería como otra cifra ("31")
_RE_NUMERO_LETRA = re.compile(r"\d[a-j](?![a-zA-Z])")
_OPERADORES_SUELTOS = {"+", "-", "−", "–", "×", "·", "*", "/", "÷", "<", ">"}


def _es_fuerte(token):
    return (any(c in _SIGNOS_FUERTES for c in token) or bool(_RE_NUMERO_LETRA.search(token))
            or bool(re.fullmatch(r"[a-zA-Z]{1,3}\([^()\s]{0,8}\)[.,;]?", token)))


def es_ecuacion(texto):
    """True si el texto tiene una expresión matemática ("y = 2x + 1", "R²")."""
    return any(_es_fuerte(t) for t in str(texto).split()) or bool(
        re.search(r"\b\d+[a-zA-Z]\s*[+\-−]\s*\d", str(texto)))


def _tramos(texto):
    """Parte el texto en tramos (es_matematico, texto): la expresión va con
    el código matemático, el resto como texto. Una palabra es parte de la
    expresión si tiene un signo matemático, o si es un número, una letra
    suelta o un operador al lado de una que ya lo es."""
    palabras = str(texto).split()
    mat = [_es_fuerte(p) for p in palabras]
    if not any(mat) and re.search(r"\b\d+[a-zA-Z]\s*[+\-−]\s*\d", str(texto)):
        mat = [p in _OPERADORES_SUELTOS for p in palabras]
    cambio = True
    while cambio:
        cambio = False
        for i, p in enumerate(palabras):
            if mat[i]:
                continue
            vecino = (i > 0 and mat[i - 1]) or (i + 1 < len(palabras) and mat[i + 1])
            if vecino and (p in _OPERADORES_SUELTOS or _RE_OPERANDO.match(p)):
                mat[i] = cambio = True
    tramos = []
    for p, m in zip(palabras, mat):
        if tramos and tramos[-1][0] == m:
            tramos[-1][1].append(p)
        else:
            tramos.append((m, [p]))
    return [(m, " ".join(ps)) for m, ps in tramos]


def _normalizar_indices(expr):
    """"x²" -> "x^(2)", "x₁" -> "x_(1)": los superíndices y subíndices de
    Unicode, como exponente o subíndice explícito."""
    salida, i = [], 0
    while i < len(expr):
        for tabla, marca in ((_SUPERINDICES, "^"), (_SUBINDICES, "_")):
            if expr[i] in tabla:
                j = i
                while j < len(expr) and expr[j] in tabla:
                    j += 1
                salida.append(marca + "(" + "".join(tabla[c] for c in expr[i:j]) + ")")
                i = j
                break
        else:
            salida.append(expr[i])
            i += 1
    return "".join(salida)


def _grupo(expr, i):
    """Lo que sigue a un ^, un _ o una raíz: un grupo entre paréntesis o
    llaves, o un término simple (número con signo, o una letra). Devuelve
    (texto del grupo, compuesto?, posición siguiente)."""
    if i < len(expr) and expr[i] in "({":
        cierre = ")" if expr[i] == "(" else "}"
        nivel, j = 0, i
        while j < len(expr):
            if expr[j] in "({":
                nivel += 1
            elif expr[j] in ")}":
                nivel -= 1
                if nivel == 0:
                    break
            j += 1
        interior = expr[i + 1:j]
        simple = re.fullmatch(r"[-−]?(\d+(?:[.,]\d+)?|[^\W\d_])", interior) is not None
        return interior, not simple, j + 1
    m = re.match(r"[-−]?(\d+(?:[.,]\d+)?|[^\W\d_])", expr[i:])
    if m:
        return m.group(0), False, i + m.end()
    return "", False, i


def expresion_a_celdas(expr):
    """Celdas Braille de una expresión matemática, con el Código Matemático
    Unificado: "y=2x+1" -> y = #b x + #a (sin espacios, como en el código)."""
    expr = _normalizar_indices(str(expr).replace(" ", ""))
    celdas, numero, i = [], False, 0
    while i < len(expr):
        c = expr[i]
        if c.isdigit():
            if not numero:
                celdas.append("numeral")
            celdas.append(c)
            numero = True
            i += 1
            continue
        if c in ",." and numero and i + 1 < len(expr) and expr[i + 1].isdigit():
            celdas.append("punto")      # coma decimal: punto 2
            i += 1
            continue
        if c in "^_" or c == "√":
            celdas += _EXPONENTE if c == "^" else _SUBINDICE if c == "_" else _RAIZ
            grupo, compuesto, i = _grupo(expr, i + 1)
            interior = expresion_a_celdas(grupo)
            celdas += (_AUX_ABRE + interior + _AUX_CIERRA) if compuesto else interior
            numero = bool(interior) and interior[-1] in DIGITOS
            continue
        numero_antes, numero = numero, False
        if c in _GRIEGAS or c.lower() in _GRIEGAS:
            base = _GRIEGAS[c.lower()]
            celdas += [_PREFIJO_GRIEGA_MAYUSCULA if c.isupper() else _PREFIJO_GRIEGA, base]
        elif c.isalpha():
            letra = quitar_tilde(c).lower()
            if letra in BRAILLE:
                if c.isupper():
                    celdas.append(_PREFIJO_MAYUSCULA)
                elif numero_antes and letra in "abcdefghij":
                    celdas.append(_PREFIJO_MINUSCULA)   # si no, se leería como cifra
                celdas.append(letra)
        elif c in SIGNOS_MATEMATICOS:
            celdas += SIGNOS_MATEMATICOS[c]
        i += 1
    return celdas


def texto_a_celdas(texto):
    """Celdas Braille de un texto. Las expresiones matemáticas que tenga
    van con el Código Matemático Unificado (ver arriba); el resto, como
    texto (ver _texto_literal_a_celdas)."""
    if not es_ecuacion(texto):
        return _texto_literal_a_celdas(texto)
    celdas = []
    for matematico, tramo in _tramos(texto):
        if celdas:
            celdas.append(" ")
        celdas += expresion_a_celdas(tramo) if matematico else _texto_literal_a_celdas(tramo)
    return celdas


# ============================================================
# 7. CALCULAR EL ANCHO FÍSICO DEL BRAILLE
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
# 8. CREAR IDENTIFICADORES PARA LA LEYENDA
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
# 9. ABREVIAR ETIQUETAS QUE NO CABEN
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
        # dos series de letras: MAYÚSCULAS para textos (A, B...) y
        # minúsculas para valores de los ejes (a, b...), ver procesar()
        self._cuenta = {True: 0, False: 0}

    def procesar(self, texto, ancho_disponible, donde=None, clave=None, celdas=None,
                 minuscula=False):
        """Dict con lo que va físicamente en la gráfica: el texto completo
        si entra en `ancho_disponible` (mm), si no su letra. `donde` (opcional) dice qué es la etiqueta
        ("título del eje X"...), para la leyenda y la narración; `clave`
        (opcional) identifica el elemento de la lámina de donde salió.

        `celdas`: las celdas Braille del texto completo, si ya vienen
        armadas (un número con su signo menos y su coma decimal); si no, se
        arman con texto_a_celdas. Con `ancho_disponible` negativo la etiqueta
        va siempre con letra.

        `minuscula=True`: la letra va en minúscula (a, b...: 1 celda, sin el
        signo de mayúscula). Se usa para los valores de los ejes, donde el
        espacio entre marcas es poco; sin signo numeral delante una letra
        sola no se confunde con un número. Las minúsculas llevan su propia
        cuenta: "A" (un texto) y "a" (un valor) son cosas distintas."""

        celdas = texto_a_celdas(texto) if celdas is None else list(celdas)

        ancho = ancho_braille(celdas)


        # ========================================
        # CASO 1: EL TEXTO SÍ CABE
        # ========================================

        if ancho <= ancho_disponible:

            return {

                "texto_original": texto,

                "texto_stl": texto,

                "celdas": celdas,

                "ancho_mm": ancho,

                "usa_leyenda": False,

                "identificador": None

            }


        # ========================================
        # CASO 2: EL TEXTO NO CABE
        # ========================================

        # el mismo texto en el mismo lugar ("2020" en el eje X) usa la misma letra
        repetido = (donde, " ".join(str(texto).split()))

        if repetido in self._por_texto:

            entrada = self._por_texto[repetido]

        else:

            identificador = crear_identificador(self._cuenta[minuscula])
            self._cuenta[minuscula] += 1
            if minuscula:
                identificador = identificador.lower()

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

                "celdas_texto": celdas

            }

            self.leyenda.append(entrada)
            self._por_texto[repetido] = entrada


        # Lo que irá físicamente en la gráfica
        return {

            "texto_original": texto,

            "texto_stl": entrada["identificador"],

            "celdas": entrada["celdas_identificador"],

            "ancho_mm": ancho_braille(entrada["celdas_identificador"]),

            "usa_leyenda": True,

            "identificador": entrada["identificador"]

        }


# ============================================================
# 10. PARTIR UN TEXTO LARGO EN RENGLONES
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
