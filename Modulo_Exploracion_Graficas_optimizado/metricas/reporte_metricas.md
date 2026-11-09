# Reporte de métricas del explorador de gráficas

Generado el 2026-11-08 21:56 con medir_metricas.py 1.2 · 2026-11-05.

| Grupo | Métrica | Objetivo | morado_g2 | salmon_g2 | turquesa_g3 |
| --- | --- | --- | --- | --- | --- |
| Selección | Acierto de selección | ≥ 90 % | 20 % (n=15; IC95 7–45) ✗ | 27 % (n=15; IC95 11–52) ✗ | 25 % (n=8; IC95 7–59) ✗ |
| Selección | · en zonas aisladas | ≥ 90 % | 14 % (n=7; IC95 3–51) ✗ | 29 % (n=7; IC95 8–64) ✗ | 25 % (n=4; IC95 5–70) ✗ |
| Selección | · junto a cruces | ≥ 80 % | 0 % (n=5; IC95 0–43) ✗ | 0 % (n=5; IC95 0–43) ✗ | — |
| Selección | · ejes | ≥ 90 % | 100 % (n=1; IC95 21–100) ✓ | 100 % (n=1; IC95 21–100) ✓ | 50 % (n=2; IC95 9–91) ✗ |
| Selección | · textos y leyendas | ≥ 90 % | 50 % (n=2; IC95 9–91) ✗ | 50 % (n=2; IC95 9–91) ✗ | 0 % (n=2; IC95 0–66) ✗ |
| Selección | Acierto de serie | ≥ 95 % | 53 % ✗ | 60 % ✗ | 38 % ✗ |
| Selección | Error de la yema (mediana) | ≤ 4 mm | 5,8 mm ✗ | 5,1 mm ✗ | 9,0 mm ✗ |
| Selección | Sesgo de la yema |  | 4,6 mm | 1,5 mm | 7,7 mm |
| Selección | Temblor | ≤ 1,5 mm | 2,6 mm ✗ | 2,2 mm ✗ | 1,7 mm ✗ |
| Selección | Cambios de elemento por objetivo | ≤ 1 | 1,0 ✓ | 1,0 ✓ | 1,0 ✓ |
| Selección | · sin suavizado ni estabilidad |  | 8,0 | 9,0 | 0,5 |
| Selección | Acierto sin suavizado ni estabilidad |  | 33 % | 27 % | 25 % |
| Registro | Error de registro (mediana) | ≤ 3 mm | 2,0 mm ✓ | 10,4 mm ✗ | 10,4 mm ✗ |
| Registro | Error máximo en el gráfico | ≤ 6 mm | 2,4 mm ✓ | 10,7 mm ✗ | 11,7 mm ✗ |
| Registro | Ídem con autodetección (A) | ≤ 6 mm | 2,3 mm ✓ | 2,5 mm ✓ | 2,9 mm ✓ |
| Registro | Autodetección (A), mediana | ≤ 3 mm | 0,9 mm ✓ | 2,0 mm ✓ | 2,0 mm ✓ |
| Pinza | Sensibilidad (pinza mantenida) | ≥ 95 % | 100 % ✓ | 100 % ✓ | 100 % ✓ |
| Pinza | Pinza falsa (% frames) | ≤ 2 % | 0 % ✓ | 0 % ✓ | 0 % ✓ |
| Pinza | Activaciones falsas | ≤ 1 /min | 0,0 /min ✓ | 0,0 /min ✓ | 0,0 /min ✓ |
| Pinza | Cortes con pinza cerrada | ≤ 2 /min | 0,0 /min ✓ | 0,0 /min ✓ | 0,0 /min ✓ |
| Pinza | · con el umbral antiguo (px, sin gracia) |  | 32,0 /min | 52,0 /min | 28,0 /min |
| Pinza | Sensibilidad con el umbral antiguo |  | 92 % | 90 % | 96 % |
| Pinza | Índice–medio / palma, dedos juntos (p90) |  | 0,24 | 0,29 | 0,24 |
| Pinza | Índice–medio / palma, dedos separados (p10) |  | 0,81 | 0,63 | 0,66 |
| Mano | Detección de la mano | ≥ 95 % | 100 % (n=300; IC95 99–100) ✓ | 100 % (n=301; IC95 99–100) ✓ | 100 % (n=300; IC95 98–100) ✓ |
| Mano | Contraste mano–placa |  | ΔE 30 | ΔE 5 | ΔE 26 |
| Mano | fps procesados | ≥ 20 | 15 ✗ | 15 ✗ | 15 ✗ |
| Mano | fps que entrega la cámara | ≥ 25 | 15 ✗ | 15 ✗ | 15 ✗ |
| Mano | Latencia de visión (p95) | ≤ 100 ms | 94 ms ✓ | 82 ms ✓ | 83 ms ✓ |
| Mano | Umbral de pinza / palma |  | 0,28 | 0,28 | 0,28 |
| Placa | Autodetección | ≥ 95 % | 100 % (n=45; IC95 92–100) ✓ | 100 % (n=45; IC95 92–100) ✓ | 100 % (n=45; IC95 92–100) ✓ |
| Placa | · con confianza alta | ≥ 90 % | 93 % ✓ | 93 % ✓ | 38 % ✗ |
| Placa | Esquina recortada correcta | ≥ 95 % | 93 % ✗ | 98 % ✓ | 33 % ✗ |
| Placa | Rotación de la placa | ≤ 2° | 1,2° ✓ | 3,5° ✗ | 1,0° ✓ |
| Placa | Placa cortada por el encuadre | 0 % | 0 % ✓ | 0 % ✓ | 33 % ✗ |
| Placa | Contraste de gris placa–fondo | ≥ 40 | 64 ✓ | 88 ✓ | 78 ✓ |
| Placa | Contraste de color placa–fondo | ≥ 20 | ΔE 28 ✓ | ΔE 31 ✓ | ΔE 35 ✓ |
| Placa | Desfase vs. calibración guardada | ≤ 3 mm | 0,7 mm ✓ | 9,7 mm ✗ | 10,7 mm ✗ |
| Placa | Dispersión al recolocar | ≤ 2 mm | 0,4 mm ✓ | 0,3 mm ✓ | 2,2 mm ✗ |
| Placa | Anisotropía de la calibración | 0,95–1,05 | 1,03 ✓ | 1,03 ✓ | 1,03 ✓ |
| QR | Lectura del QR | ≥ 80 % | 0 % (n=144; IC95 0–3) ✗ | 24 % (n=83; IC95 16–34) ✗ | 0 % (n=152; IC95 0–2) ✗ |
| QR | Tiempo hasta la 1.ª lectura | ≤ 3 s | — | 0,9 s ✓ | — |
| Geometría | Área ambigua (2+ series) | ≤ 10 % | 23 % ✗ | 23 % ✗ | 0 % ✓ |
| Geometría | Conflicto de prioridad | ≤ 5 % | 0 % ✓ | 0 % ✓ | 0 % ✓ |
| Geometría | Acierto esperado, yema ±4 mm, puntos aislados | ≥ 90 % | 53 % ✗ | 53 % ✗ | 68 % ✗ |
| Geometría | Acierto esperado, yema ±4 mm, líneas aisladas | ≥ 90 % | 53 % ✗ | 53 % ✗ | 95 % ✓ |
| Geometría | Acierto esperado, yema ±4 mm, cruces | ≥ 80 % | 25 % ✗ | 25 % ✗ | — |
| Geometría | Acierto esperado, yema ±4 mm, ejes | ≥ 90 % | 78 % ✗ | 78 % ✗ | 89 % ✗ |
| Geometría | Trazo simulado: serie seguida en cruces | ≥ 90 % | 100 % ✓ | 100 % ✓ | — |
| Geometría | · sin estabilidad |  | 58 % | 58 % | — |
| Geometría | Trazo simulado: cambios de elemento |  | 2,6 /s | 2,6 /s | 1,2 /s |
| Geometría | · sin estabilidad |  | 6,4 /s | 6,4 /s | 1,8 /s |
| Geometría | Separación mínima entre series |  | 1,5 mm | 1,5 mm | — |
| Geometría | Cruces entre series |  | 3 | 3 | 0 |

## Condiciones y sesiones

- **morado_g2**: grafica = 2; gráfica: Ventas mensuales por producto; sesiones: falsos ×1, geometria ×1, hoja ×1, mano ×1, objetivos ×1, qr ×1, registro ×1
  - Confusiones más frecuentes: s2_p7 → s2_p6-p7 (×1); s1_p10-p11 → s1_p10 (×1); s1_p7 → s1_p6-p7 (×1); s1_p8-p9 → s2_p9 (×1); s2_p1 → eje_y (×1)
- **salmon_g2**: color = claro, grafica = 2; gráfica: Ventas mensuales por producto; sesiones: falsos ×1, hoja ×1, mano ×1, objetivos ×1, qr ×1, registro ×1; geometría tomada de morado_g2 (misma gráfica: solo depende del CSV)
  - Confusiones más frecuentes: s2_p7 → s3_p6-p7 (×1); s1_p10-p11 → s1_p10 (×1); texto_2 → eje_x (×1); s1_p8-p9 → None (×1); s2_p1 → s2_p1-p2 (×1)
- **turquesa_g3**: grafica = 3; gráfica: Presión vs altitud (con discontinuidad y datos faltantes); sesiones: falsos ×1, geometria ×1, hoja ×1, mano ×1, objetivos ×1, qr ×1, registro ×1
  - Confusiones más frecuentes: s1_p3 → s1_p2-p3 (×1); s1_p1 → None (×1); texto_3 → eje_x (×1); texto_6 → s1_p5 (×1); s1_p5 → None (×1)

✓ cumple el objetivo propuesto · ✗ no lo cumple · — sin medir. Las proporciones llevan su intervalo de confianza del 95 % (Wilson).
