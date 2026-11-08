# Detección de glaucoma en imágenes de fondo de ojo

Sistema de clasificación de imágenes de fondo de ojo para estimar si hay glaucoma, con probabilidad asociada a cada predicción. Incluye el análisis experimental de hiperparámetros, el modelo final y una interfaz web para subir imágenes y ver el resultado.

> **Aviso:** proyecto académico/educativo. No es un instrumento de diagnóstico clínico.

## Contenido
- [Dataset](#dataset)
- [Preparación reproducible de PAPILA](#preparación-reproducible-de-papila)
- [Carga de PAPILA y CUDA](#carga-de-papila-y-cuda)
- [Entrenamiento de la CNN e inferencia local](#entrenamiento-de-la-cnn-e-inferencia-local)
- [Transfer Learning con ResNet18](#transfer-learning-con-resnet18)
- [Evaluación interna congelada de PAPILA](#evaluación-interna-congelada-de-papila)
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

## Entrenamiento de la CNN e inferencia local

La referencia usa la CNN histórica con salida de logits, Adam a `0.001`, BCE con logits, semilla `42`, lote `16`, imágenes RGB de `96 × 96`, `num_workers=0`, volteo horizontal en train, dropout y weight decay en `0`. Se detiene tras cinco épocas consecutivas sin mejora estricta de la pérdida de validación, hasta un máximo de 40. La época seleccionada es la de menor pérdida de validación. El umbral exploratorio de `0.5` se usa para las métricas y la clase predicha. PR-AUC es el área trapezoidal de la curva precisión–recall. ROC-AUC y PR-AUC quedan vacías cuando falta una clase en validación.

Usa el manifiesto existente y una carpeta nueva por corrida. El bucle carga exclusivamente imágenes de train y val. En CUDA, el entrenamiento usa AMP FP16 y registra pasos omitidos por desbordamiento de gradientes. La validación y la inferencia usan FP32. El comando fija las semillas y activa los algoritmos deterministas de PyTorch. Esta corrida usa la partición nueva y AMP, por lo que sus métricas de validación corresponden a un experimento distinto del histórico. [Reproducibilidad de PyTorch](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --dataset-dir data/papila --manifest data/papila_manifest.csv --run-dir models/runs/papila-smoke --epochs 1
.\.venv\Scripts\python.exe -m src.train --dataset-dir data/papila --manifest data/papila_manifest.csv --run-dir models/runs/papila-reference
```

Cada corrida escribe `history.csv`, `loss.png`, `summary.json` y `best.pt` en su carpeta. El resumen registra el máximo de memoria asignada y reservada por PyTorch en CUDA. El checkpoint contiene pesos, época seleccionada, configuración, hash del manifiesto y normalización de train. Git excluye `models/runs/`. Consulta `python -m src.train --help` para ajustar los parámetros, y usa `--cpu` o `--no-amp` cuando corresponda.

```powershell
.\.venv\Scripts\python.exe -m src.predict --checkpoint models/runs/papila-reference/best.pt --image path/to/fundus.jpg
```

La salida de inferencia es un puntaje académico de glaucoma sin calibrar. La aplicación de la sigmoide convierte el logit al rango `[0, 1]`.

## Transfer Learning con ResNet18

La ResNet18 usa `torchvision 0.29.1+cu130` y los pesos oficiales `ResNet18_Weights.IMAGENET1K_V1` (`resnet18-f37072fd.pth`, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`). El backbone y sus estadísticas de BatchNorm quedan congelados. Solo se ajusta la cabeza lineal de 512 entradas y un logit, con 513 parámetros entrenables. [Pesos y normalización de Torchvision](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.resnet18.html)

La entrada RGB se ajusta dentro de `224 × 224` con interpolación bilineal, conservación de aspecto y padding negro centrado. Una imagen PAPILA de `2576 × 1934` ocupa `224 × 168`, con 28 píxeles de padding arriba y abajo. Se escala a `[0, 1]` y se aplica la media `(0.485, 0.456, 0.406)` y desviación `(0.229, 0.224, 0.225)` de ImageNet. Se sustituye el recorte central de las transformaciones oficiales por padding para conservar el campo visual. El único aumento es el volteo horizontal con probabilidad `0.5` en train.

El entrenamiento usa el mismo manifiesto, `batch_size=16`, `num_workers=0`, Adam a `0.001`, semilla `42`, BCE con logits y AMP FP16 en CUDA. Se restaura la época de menor pérdida de validación con paciencia de cinco y máximo de 30 épocas. La validación usa FP32. Desde la raíz, con la `.venv` de CUDA y las dependencias de entrenamiento instaladas:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --architecture resnet18 --dataset-dir data/papila --manifest data/papila_manifest.csv --run-dir models/runs/papila-resnet18-smoke --epochs 1
.\.venv\Scripts\python.exe -m src.train --architecture resnet18 --dataset-dir data/papila --manifest data/papila_manifest.csv
.\.venv\Scripts\python.exe -m src.compare --dataset-dir data/papila --manifest data/papila_manifest.csv
.\.venv\Scripts\python.exe -m src.predict --checkpoint models/runs/papila-resnet18/best.pt --image path/to/fundus.jpg
```

La comparación recarga ambos checkpoints en las mismas imágenes de validación y usa `0.50` como referencia. Guarda `comparison.json`, `curves.png`, `predictions.csv` y `thresholds.csv` en `models/runs/papila-resnet18/comparison/`. Average Precision (AP) pondera la precisión por cada incremento de recall. El área trapezoidal PR-AUC se registra como una métrica separada. La pérdida BCE es comparable sobre esas etiquetas, con la reserva de que cada modelo usa su propio preprocesamiento. Los tiempos totales incluyen preparación y guardado, y dependen del número de épocas ejecutadas. [Definición de Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

El análisis exploratorio usa una rejilla de `0.00` a `1.00` en pasos de `0.05`. Entre los umbrales con especificidad de al menos 80 %, maximiza sensibilidad y resuelve empates por mayor especificidad y luego mayor umbral. El punto elegido queda en el informe, separado del umbral `0.50` del checkpoint. Con 13 positivos, una sola imagen mueve la sensibilidad 7,7 puntos porcentuales. Las métricas de estos umbrales usan el mismo conjunto que los seleccionó. Una futura evaluación del test de PAPILA sería interna y exploratoria debido a su consulta histórica.

Un experimento posterior de ajuste parcial podría descongelar `layer4` con un learning rate inferior al de la cabeza y conservar el protocolo de validación agrupado por paciente. La corrida actual ajusta exclusivamente la cabeza y reporta puntajes académicos sin calibrar.

## Evaluación interna congelada de PAPILA

`configs/papila-internal-test.json` fija checkpoints, preprocesamiento, dependencias, umbrales e identidades SHA-256 del manifiesto y las imágenes de cada split. ResNet18, época 18, se evalúa a `0.35`. La CNN de referencia, época 8, se evalúa a `0.50`. Su umbral exploratorio de validación `0.55` se conserva como antecedente. Los resultados de validación en los umbrales aplicados se copiaron del informe de comparación previo, identificado por su hash.

Desde la raíz, en PowerShell, indica la copia local de PAPILA y el manifiesto congelado:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.internal_test --dataset-dir data/papila --manifest data/papila_manifest.csv --preflight
.\.venv\Scripts\python.exe -m src.internal_test --dataset-dir data/papila --manifest data/papila_manifest.csv --confirm-internal-test
```

El preflight verifica bytes y particiones y prueba ambos modelos con tensores sintéticos. La confirmación explícita ejecuta una pasada por checkpoint en test, en FP32 y modo inferencia. Comprueba que los parámetros y buffers conservan su hash. La carpeta local `models/runs/papila-internal-test/` se reserva antes de abrir el primer JPEG de test y su existencia bloquea otra ejecución. Guarda la configuración, métricas, predicciones por ojo agrupadas también por paciente, matrices, curvas e información de reproducibilidad. Los artefactos quedan fuera de Git.

Los intervalos exploratorios del 95 % usan 5.000 remuestreos percentiles por paciente con semilla 42. Cada extracción conserva ambos ojos y el mismo remuestreo para los dos modelos. Las réplicas con una métrica indefinida se omiten solo para esa métrica y se registra su cantidad. ROC-AUC y AP requieren ambas clases, sensibilidad y F requieren positivos reales, especificidad requiere negativos reales y precisión requiere predicciones positivas. [Bootstrap con datos oculares correlacionados](https://pmc.ncbi.nlm.nih.gov/articles/PMC7500131/)

PAPILA test se consultó en experimentos históricos. Esta evaluación es interna y exploratoria. Las decisiones de modelo, umbral y preprocesamiento se fijaron en validación antes de ejecutarla. Los resultados describen puntajes académicos sin calibrar.

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
