# Detección de glaucoma en imágenes de fondo de ojo

Sistema de clasificación de imágenes de fondo de ojo para estimar si hay glaucoma, con probabilidad asociada a cada predicción. Incluye el análisis experimental de hiperparámetros, el modelo final y una interfaz web para subir imágenes y ver el resultado.

> **Aviso:** proyecto académico/educativo. No es un instrumento de diagnóstico clínico.

## Contenido
- [Dataset](#dataset)
- [Preparación reproducible de PAPILA](#preparación-reproducible-de-papila)
- [Carga de PAPILA y CUDA](#carga-de-papila-y-cuda)
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

## Preparación reproducible de PAPILA

Esta etapa usa la [copia de PAPILA utilizada por el notebook](https://github.com/valentinaache/glaucomaPGUSab/tree/41f0a0718252dbed301adf90976b67688d918551), fijada al commit `41f0a0718252dbed301adf90976b67688d918551`. Consúltese también el [artículo de PAPILA](https://doi.org/10.1038/s41597-022-01388-1) y la [publicación oficial en Figshare](https://figshare.com/articles/dataset/PAPILA/14798004), que lista la versión 2 y la licencia GPL 3.0+. La corrida de verificación de esta etapa se hizo sobre la copia fijada de GitHub.

Desde la raíz del proyecto, con Python 3.11 o posterior, coloca los datos en `data/papila/`. Esa ruta debe contener `FundusImages/` y `ClinicalData/`, con `patient_data_od.xlsx` y `patient_data_os.xlsx` dentro de esta última carpeta.

```bash
python -m pip install -r requirements-data.txt
python -m src.prepare_papila --dataset-dir data/papila --output data/papila_manifest.csv --seed 42
python -m unittest discover -s tests -v
```

El comando indexa y decodifica los JPEG, y los une con las etiquetas por paciente y ojo. En la copia fijada, la planilla `os` contiene ambos ojos con lateralidad explícita. La planilla `od` contiene únicamente ojos derechos, con lateralidad implícita. Se concilian las coincidencias entre planillas y se rechazan los duplicados dentro de una planilla, las contradicciones y los registros sin pareja. El manifiesto conserva las clases 0 y 1, con una fila por ojo elegible y las columnas `image_id`, `patient_id`, `eye`, `label`, `relative_path` y `split`. Las rutas son relativas a `--dataset-dir`.

La partición usa la semilla **42** y agrupa a cada paciente por la combinación de diagnósticos de sus ojos elegibles. Así, los pacientes con un ojo sano y otro con glaucoma forman un estrato propio y cada ojo conserva su etiqueta. Dentro de cada estrato se ordenan los identificadores, se barajan con la semilla y se asignan pacientes a `train`, `val` y `test` en proporción aproximada de **65/15/20**, usando redondeo por restos mayores. Los empates se resuelven en el orden `train`, `val`, `test`. Los estratos pequeños pueden quedar ausentes de algún conjunto.

Se verificó la decodificación de **488 imágenes**, se conciliaron **244** etiquetas derechas redundantes y se excluyeron **68** ojos sospechosos. El manifiesto contiene **420 imágenes** de **210 pacientes**, con cero pacientes compartidos entre conjuntos.

| Split | Pacientes | Imágenes | Sanos (0) | Glaucoma (1) |
|-------|----------:|---------:|----------:|-------------:|
| Train | 137 | 274 | 217 | 57 |
| Val   | 31  | 62  | 49  | 13 |
| Test  | 42  | 84  | 67  | 17 |

Los conteos pasan de 272/64/84 en el experimento histórico a 274/62/84 con el nuevo método. Coincidir en el tamaño de prueba no implica que los pacientes sean los mismos. Dos ejecuciones sobre la misma fuente con semilla 42 produjeron CSV idénticos, con SHA-256 `c82776614b94d458e42e56b8f66270e1fbe0be9b4a80a741e2cb6923449ef5cd`. Los resultados de las secciones siguientes corresponden a la partición histórica.

## Carga de PAPILA y CUDA

El entorno verificado usa Windows, Python `3.14.6`, PyTorch `2.14.1+cu130` y torchvision `0.29.1+cu130`. CUDA `13.0` se verificó en una NVIDIA GeForce RTX 4050 Laptop GPU de 6 GB con el controlador `621.01`. En la `.venv` existente, ejecuta desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Para reproducir ese entorno en una copia nueva del repositorio, usa Python 3.14, un controlador NVIDIA compatible y el índice oficial `cu130`. Estos comandos son para una instalación nueva. [Instalación de PyTorch](https://pytorch.org/get-started/locally/)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 torchvision==0.29.1+cu130
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

Usa el CSV de la etapa anterior. `make_loaders` conserva sus asignaciones y valida que ningún paciente cruce entre conjuntos. Los valores iniciales son `batch_size=16`, `num_workers=0` e imágenes RGB de `96 × 96`. La media y la desviación por canal se calculan en una pasada por `train` sin *augmentation*. La decodificación se hace bajo demanda en CPU y el volteo horizontal aleatorio se aplica solo en `train`. El argumento `seed` controla el orden de carga y `torch.manual_seed(42)` inicializa también el generador usado por el *augmentation* con `num_workers=0`.

```python
from pathlib import Path

import torch

from src.data import make_loaders
from src.device import select_device

torch.manual_seed(42)
device = select_device()
loaders = make_loaders(Path("data/papila"), Path("data/papila_manifest.csv"), device=device)
images, labels = device.move_batch(next(iter(loaders.train)))
```

El cargador real de prueba se habilita con `include_test=True` para la evaluación final. Con CUDA, los lotes usan memoria fijada y se transfieren con `non_blocking=True`. `device.autocast()` y `device.make_grad_scaler()` preparan AMP en FP16 y quedan deshabilitados en CPU. Al implementar la CNN entrenable con AMP, usa logits y `BCEWithLogitsLoss`, y aplica la sigmoide al obtener puntajes de predicción. [Documentación de DataLoader](https://docs.pytorch.org/docs/2.14/data.html) · [Documentación de AMP](https://docs.pytorch.org/docs/2.14/amp.html)

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
