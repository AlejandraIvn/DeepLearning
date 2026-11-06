# Detección de glaucoma en imágenes de fondo de ojo

Sistema de clasificación de imágenes de fondo de ojo para estimar si hay glaucoma, con probabilidad asociada a cada predicción. Incluye el análisis experimental de hiperparámetros, el modelo final y una interfaz web para subir imágenes y ver el resultado.

> **Aviso:** proyecto académico/educativo. No es un instrumento de diagnóstico clínico.

## Contenido
- [Dataset](#dataset)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Modelo final](#modelo-final)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y uso](#instalación-y-uso)
- [Limitaciones](#limitaciones)

## Dataset
Se usa el dataset PAPILA (repositorio fuente: https://github.com/valentinaache/glaucomaPGUSab).

- 488 imágenes en total; se excluye la clase *sospechoso*, quedan **420 imágenes**.
- 210 pacientes, con ambos ojos (OD y OS).
- Distribución: 333 sanos (79 %) y 87 glaucoma (21 %). **Desbalanceado.**
- Resolución original 2576 × 1934 px, redimensionada a **96 × 96 px**.
- Partición **estratificada por paciente** (semilla 42), para que los dos ojos de un mismo paciente no estén en conjuntos distintos:

| Split | Imágenes | % glaucoma |
|-------|----------|------------|
| Train | 272 | 21.0 % |
| Val   | 64  | 20.3 % |
| Test  | 84  | 20.2 % |

## Metodología
1. **Modelo base:** CNN con 2 bloques Conv2D (32 y 64 filtros, 3×3) + ReLU + MaxPool, capa densa de 128 unidades y salida sigmoide. Pérdida BCE y optimizador Adam (lr = 1e-3).
2. **Diagnóstico:** el modelo base presenta sobreajuste (la pérdida de entrenamiento baja mientras la de validación se mantiene fluctuante).
3. **Barridos de hiperparámetros**, uno a la vez:
   - Regularización L2 (weight decay): {0, 1e-5, 1e-4, 1e-3, 1e-2}
   - Dropout: {0.0, 0.2, 0.3, 0.5, 0.7}
   - Data augmentation: Sin / Leve / Moderada / Fuerte
   - Early stopping (paciencia): {3, 5, 10}
   - Learning rate y optimizador (SGD+momentum, RMSprop, Adam), con un barrido de lr propio para cada uno
4. **Diseño factorial completo:** 4 factores binarios × 3 optimizadores = 48 configuraciones.
5. **Selección por Val Loss.** El conjunto de test se usa solo para el reporte final.

## Resultados
Métricas sobre el conjunto de **test (84 imágenes)**, clase positiva = glaucoma, umbral 0.5:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|---------:|----------:|-------:|---:|--------:|
| Baseline (sin regularización) | 0.798 | 0.500 | 0.529 | 0.514 | 0.777 |
| Mejor learning rate (5e-4)    | 0.774 | 0.450 | 0.529 | 0.487 | **0.841** |
| **Combinado (modelo final)**  | **0.857** | **0.667** | **0.588** | **0.625** | 0.817 |

Matriz de confusión del modelo final: 10 de 17 glaucomas detectados (7 falsos negativos) y 5 falsos positivos.

Se prioriza el **recall**, porque un falso negativo (glaucoma no detectado) es más costoso que un falso positivo.

## Modelo final
- Arquitectura: CNN de la sección *Metodología*, entrada 96 × 96 × 3.
- Optimizador: Adam, lr = 1e-3.
- Regularización: augmentation Leve (flip horizontal, rotación ±10°, jitter de brillo y contraste), early stopping con paciencia 5. Sin L2 ni Dropout.
- Salida: probabilidad de glaucoma en [0, 1] y clase según umbral.

<!-- Completar cuando esté el modelo YOLO: arquitectura, lr, épocas, métricas y motivo de elección -->

## Estructura del repositorio
```
├── notebooks/        # experimentos (barridos, factorial, métricas)
├── src/              # datos, modelo, entrenamiento y evaluación
├── models/           # pesos exportados del modelo final
├── backend/          # API de inferencia
├── frontend/         # interfaz web
└── README.md
```

## Instalación y uso
<!-- Completar cuando estén listos backend y frontend -->

## Limitaciones
- El conjunto de test tiene solo 84 imágenes (17 con glaucoma). Una diferencia de un caso cambia el recall aproximadamente 6 puntos, así que las comparaciones entre modelos son orientativas.
- Un único split. No se usó validación cruzada.
- La resolución de 96 × 96 px puede perder detalles finos del disco óptico.
- Resultados de investigación; no reemplazan la evaluación de un especialista.

## Licencia y créditos
<!-- Completar: licencia y cita del dataset PAPILA -->
