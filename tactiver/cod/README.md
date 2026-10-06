# Tactonauta — servidor

Convierte las gráficas de líneas de un PDF en láminas táctiles para
imprimir en 3D (STL), con texto en Braille español y una narración para
Hand_Tracking. Sirve la interfaz web (`tactiver/interfaz/tactiverso (10).html`)
y la API que usa.

## Instalación

```bash
pip install -r requirements.txt
python api.py          # http://localhost:5000
```

Tesseract OCR es opcional: solo hace falta para gráficas escaneadas o
pegadas como imagen (en un PDF con texto se usa el texto real). En
Windows se busca en `C:\Program Files\Tesseract-OCR\tesseract.exe`;
para verificarlo: `python -c "import segmentador; segmentador.diagnostico()"`.
En Render lo instala el `Dockerfile`.

## Flujo

1. **Estudiante**: sube el PDF; se detectan las gráficas de líneas y elige
   cuáles enviar a su supervisor.
2. **Supervisor**: "Revisar" lee cada gráfica. La compara con la original
   y con la vista previa de la lámina, y la corrige en la pizarra si hace
   falta (la lectura a la izquierda; a la derecha, lo que se imprime:
   quitar, mover o cambiar textos). Da el visto bueno y "Autoriza": recién
   ahí se genera el STL, una sola vez.
3. **Imprenta**: recibe las láminas y las pasa por su tablero.

## La lámina

- Placa fija de **22 × 22 cm**, 1,5 mm de grosor, esquina superior derecha
  recortada (orientación al tacto).
- La gráfica va **arriba**, con la proporción del original; lo que sobra
  abajo es la **leyenda**.
- Hasta 3 curvas, cada una con su textura: en celdas, rayada, punteada.
- Lo que no entra en su lugar va con una letra explicada en la leyenda:
  **A, B…** para textos largos (títulos, categorías) y **a, b…** para los
  números de un eje cuando no entran todos.
- Junto al STL se generan la narración (JSON) y un CSV para Hand_Tracking
  con la posición en la placa de todo lo que está en relieve.

## Módulos

```
api.py              servidor Flask: sesiones, solicitudes, revisión, imprenta
db.py               base de datos (SQLite local o Turso)
pipeline_rapido.py  extrae las figuras del PDF (y su texto real)
classify_charts.py  clasifica cada figura (líneas, barras, torta, dispersión...)
segmentador.py      lee la gráfica: ejes, escala, curvas, títulos, leyenda
generador_stl.py    arma la lámina: distribución, Braille, texturas, STL
braille.py          Braille español (tabla, mayúsculas, letras A/a para la leyenda)
narracion.py        descripción narrada + exportación para Hand_Tracking
```

Los endpoints están listados al principio de `api.py`.
