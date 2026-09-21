# Digitalizador automático de gráficas — API

Backend que recibe la imagen de una gráfica y devuelve, sin ninguna
interacción manual (sin SAM2, sin clics), un CSV con:

- Posición (píxeles) de los ejes X e Y.
- Título y etiquetas numéricas de cada eje (leídos por OCR).
- Calibración píxel → valor real de cada eje.
- Todos los puntos de la curva de datos, ya convertidos a valores reales.

No trae interfaz propia — está pensado para que **tu página web ya
existente** (sin importar el stack: React, Vue, PHP, HTML plano, etc.)
lo llame por HTTP.

## Instalación

```bash
pip install -r requirements.txt
```

También necesitas el binario de Tesseract OCR instalado (no solo la
librería de Python):

- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
- **Linux**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`

Si Tesseract no queda en el PATH del sistema, agrega al inicio de
`api.py`:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Levantar la API

```bash
python api.py
```

Queda escuchando en `http://localhost:5000`.

## Cómo conectarla desde tu interfaz

### 1) Opción JSON (recomendada) — subes la imagen, te da un resumen + links

```javascript
const formData = new FormData();
formData.append("imagen", archivoSeleccionado); // <input type="file">

const res = await fetch("http://localhost:5000/api/procesar", {
  method: "POST",
  body: formData,
});
const data = await res.json();

// data.resumen            -> qué se detectó (ejes, curva, nº de etiquetas, título...)
// data.csv_url             -> para mostrar/previsualizar
// data.csv_download_url    -> para forzar la descarga del CSV
// data.overlay_url         -> imagen PNG de verificación (ejes en rojo/azul, curva en verde)
```

Puedes usar `overlay_url` directo en un `<img src="...">` de tu web para
que el usuario confirme visualmente que la detección fue correcta antes
de usar el CSV.

### 2) Opción archivo directo — te devuelve el CSV sin JSON intermedio

```javascript
const res = await fetch("http://localhost:5000/api/procesar?formato=csv", {
  method: "POST",
  body: formData,
});
const blob = await res.blob(); // el CSV, listo para descargar o leer
```

### 3) Desde curl / backend en otro lenguaje

```bash
curl -F "imagen=@grafica.png" http://localhost:5000/api/procesar
curl -F "imagen=@grafica.png" "http://localhost:5000/api/procesar?formato=csv" -o resultado.csv
```

Cualquier lenguaje que pueda hacer un POST `multipart/form-data` puede
consumir esta API (no hace falta que sea JavaScript).

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/api/salud` | Chequeo de que la API está viva |
| POST | `/api/procesar` | Sube una imagen (campo `imagen`), procesa y devuelve JSON con resumen + URLs |
| POST | `/api/procesar?formato=csv` | Igual, pero devuelve el CSV directo como descarga |
| GET | `/api/resultados/<archivo>` | Sirve un CSV u overlay ya generado (para `<img>` o previsualización) |
| GET | `/api/resultados/<archivo>/descargar` | Igual, forzando descarga |
| POST | `/api/generar-stl` | Genera y devuelve una placa táctil STL de la Fase 1 |

### Generar una placa táctil STL (Fase 1)

Instala también `numpy-stl` (ya incluido en `requirements.txt`) y envía dos
puntos expresados en milímetros. El resultado es una placa de 210 x 148 mm
por defecto, con ejes, marcas y números Braille en relieve.

```bash
curl -X POST http://localhost:5000/api/generar-stl \
  -H "Content-Type: application/json" \
  -d '{"p1":[0,0],"p2":[150,90],"ancho":210,"alto":148,"intervalo_ticks":25}'
```

La respuesta contiene `stl_url` para visualizar/obtener el archivo y
`stl_download_url` para descargarlo e imprimirlo en 3D.

### PDF a STL automático

Para procesar un PDF completo, detectar sus gráficas lineales, tomar los dos
extremos de cada recta y crear un STL por cada una, usa `POST /api/pdf-a-stl` con el campo
multipart `pdf`:

```bash
curl -F "pdf=@paper.pdf" http://localhost:5000/api/pdf-a-stl
```

La respuesta incluye una entrada en `stls` por cada gráfica lineal detectada.
Cada entrada contiene las URLs del STL o un campo `error` si esa gráfica no
pudo segmentarse. El generador usa los valores calibrados por OCR cuando están
disponibles; si no lo están, usa las coordenadas de píxeles de los extremos.

CORS está abierto (`Access-Control-Allow-Origin: *`) para que tu web,
aunque esté en otro dominio o puerto, pueda llamar a la API sin
problemas. Si quieres restringirlo a tu dominio, edita la función
`habilitar_cors` en `api.py`.

## Estructura

```
api.py              API HTTP (Flask): endpoints de arriba
segmentador.py       toda la lógica de detección (ejes, curva, texto, CSV)
requirements.txt
```

## Notas sobre precisión

- **Ejes**: se asume que son líneas rectas y contrastadas (como en una
  gráfica típica de matplotlib/Excel). Si tu gráfica tiene ejes muy
  delgados o de bajo contraste, ajusta `largo_minimo` en
  `detectar_ejes()` dentro de `segmentador.py`.
- **Curva**: se detecta por el color más saturado de la imagen (distinto
  del fondo y del texto). Si la gráfica tiene varias curvas de colores,
  hoy el pipeline se queda con la de mayor color dominante; se puede
  extender fácilmente para detectar varias (avísame si lo necesitas).
- **Calibración**: se necesitan al menos 2 números legibles por eje. Con
  imágenes de baja resolución el OCR puede perder alguna etiqueta — usa
  capturas o fotos nítidas y de buen tamaño para mejores resultados.
- Cada resultado se guarda con un ID único en `resultados/`, así que
  puedes procesar muchas imágenes desde tu interfaz sin que se
  sobrescriban entre sí.
