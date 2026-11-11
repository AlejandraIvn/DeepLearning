# 03 · Modelos y entrenamiento

## Modelos

| Modelo | Papel | Entrada | Qué se entrena |
|---|---|---|---|
| **ResNet34** | Principal de la demo | 224 px, *letterbox* | Cabeza y `layer4`: 13 114 881 de 21 285 185 parámetros |
| ResNet18 | Respaldo | 224 px | Solo la cabeza: 513 parámetros |
| EfficientNet-B0 | Solo campañas | 224 px | Según receta |
| CNN (`papila_cnn`) | Referencia | 96 px | Toda la red (~4,74 M) |

**CNN:** dos bloques `Conv(3→32)`/`Conv(32→64)` + ReLU + MaxPool, capa densa de 128 y salida de 1 logit.

**Transfer learning (`PapilaTransferModel`):** parte de pesos `IMAGENET1K_V1` verificados por SHA-256, sustituye la capa final por `Linear(…, 1)` y permite ajustar 0, 1 o 2 bloques finales (`tune_blocks`), con BatchNorm congelada o activa. La entrada se ajusta con *letterbox* para no perder campo visual.

**ResNet34 servida** (`resnet34-lr-low-s43`): Adam 0,0001, BCE con peso positivo 3,807 (= 217/57), semilla 43, lote 16, FP32; 9 épocas ejecutadas y **época 4** elegida (menor BCE de validación). Umbral **0,48704302310943604**, fijado solo con validación.

## Checkpoints

Archivo versionado con pesos, arquitectura, época, métricas de validación, configuración, normalización, preprocesamiento y hash del manifiesto. Se carga con `weights_only=True`. Hay dos en `backend/predict/`, declarados en `configs/glaucoai-models.json`:

| Archivo | Modelo | SHA-256 (abreviado) | Umbral |
|---|---|---|---:|
| `resnet34.pt` | ResNet34 | `7ad8935d…13f49` | 0,487 |
| `best.pt` | ResNet18 | `721699061b…c154b761` | 0,35 |

## Entrenamiento (`src/train.py`)

Por defecto: BCE con logits, Adam `lr = 0,001`, lote 16, semilla 42, volteo horizontal, parada con paciencia 5, hasta 40 épocas (30 en preentrenados), AMP FP16 en CUDA. Se restauran los pesos de la época con menor pérdida de validación. **El test no se abre.**

```bash
pip install -r requirements-training.txt
python -m src.train --dataset-dir data/papila --manifest data/papila_manifest.csv                        # CNN
python -m src.train --architecture resnet18 --dataset-dir data/papila --manifest data/papila_manifest.csv   # ResNet18
python -m src.train --architecture resnet34 --tune-last-block --no-freeze-batch-norm --learning-rate 0.0001 \
    --positive-weight 3.807017543859649 --epochs 15 --seed 43 --no-amp --cache-preprocessed \
    --dataset-dir data/papila --manifest data/papila_manifest.csv --run-dir models/runs/resnet34-demo
```

Cada corrida necesita una carpeta nueva (`--run-dir`). `--help` lista todas las opciones: pérdida (`--positive-weight`, `--focal-loss`, `--label-smoothing`), optimización (`--optimizer`, `--scheduler`, `--accumulation-steps`), datos (`--augmentation`, `--cache-preprocessed`) y ejecución (`--cpu`, `--no-amp`, `--amp-dtype`, `--init-checkpoint`).

Salidas: `history.csv`, `loss.png`, `validation.png`, `summary.json` y `best.pt`. El script se detiene si la pérdida o los gradientes dejan de ser finitos o si cambia el manifiesto.

## Inferencia local

```bash
python -m src.predict --checkpoint backend/predict/resnet34.pt --image foto.jpg --threshold 0.48704302310943604
```

Sin `--threshold` usa el umbral guardado en el checkpoint (no el operativo fijado después en validación).

---
