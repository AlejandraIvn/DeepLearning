# 09 · Experimentos del notebook

## Diseño

CNN de dos bloques a 96 × 96 px, BCE, Adam `lr = 1e-3`, lote 32, selección por pérdida de validación y restauración de la mejor época. Con 20 épocas el modelo base sobreajusta (pérdida de entrenamiento 0,125 frente a validación fluctuante de hasta 0,654).

## Barridos (un factor a la vez, hasta 40 épocas)

| Factor | Valores | Hallazgo (pérdida de validación; base 0,3487) |
|---|---|---|
| L2 | 0 a 1e-2 | No mejoró; el mejor fue λ = 0 |
| Dropout | 0 a 0,7 | No mejoró; el mejor con dropout fue p = 0,3 (0,3676) |
| **Aumento de datos** | Sin / Leve / Moderada / Fuerte | **Única mejora:** Leve 0,2907; Fuerte 0,4119 (peor) |
| Paciencia | 3 · 5 · 10 | Con 3 hay subajuste; 5 y 10 igualan al base y ahorran cómputo |
| Tasa de aprendizaje | 1e-4 a 3e-3 | 5e-4 ≈ 1e-3; 3e-3 inestable |
| Optimizador | SGD, RMSprop, Adam | Diferencias despreciables; se mantiene Adam |

Niveles de aumento: **Leve** = solo volteo horizontal; **Moderada** = volteo, rotación ±10° y brillo/contraste; **Fuerte** = además rotación ±20°, saturación y traslación. El volteo conserva el disco óptico; un aumento fuerte lo distorsiona.

**Diseño factorial** (48 configuraciones: 3 optimizadores × L2, dropout, aumento, parada): la mejor fue **Adam + aumento** (0,2907) y las 10 mejores incluyen aumento. La parada con paciencia 5 empeoró el resultado. La configuración «Combinado» (los mejores valores de cada barrido) quedó 10.ª de 48: juntar lo mejor de cada factor no garantiza la mejor combinación.

## Resultados en test (84 imágenes, umbral 0,5)

| Modelo | Exactitud | Sensibilidad | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline | 0,798 | 0,529 | 0,514 | 0,777 |
| Mejor `lr` (5e-4) | 0,774 | 0,529 | 0,487 | 0,841 |
| **Combinado (final)** | **0,857** | **0,588** | **0,625** | 0,817 |

El modelo final detecta 10 de 17 glaucomas con 5 falsos positivos. El aumento Leve tuvo la mejor pérdida de validación pero las métricas de test más bajas: **la pérdida de validación no garantiza el mejor resultado en todas las métricas**.

## Cómo ejecutarlo

Abre el notebook en Colab o Jupyter (mejor con GPU) y ejecuta las celdas en orden; la primera fija `CUBLAS_WORKSPACE_CONFIG` y debe correr antes de inicializar CUDA. Necesita internet (clona los datos). Los barridos entrenan decenas de modelos.

## Nota sobre el README principal

El `README.md` describe el aumento «Leve» como «volteo, rotación ±10° y jitter»; en el notebook eso es el nivel **Moderada** (Leve es solo el volteo).

---

