# 02 · Datos

## PAPILA

Imágenes de fondo de ojo de ambos ojos de 244 pacientes, con datos clínicos. Se usa la copia de [`valentinaache/glaucomaPGUSab`](https://github.com/valentinaache/glaucomaPGUSab/tree/41f0a0718252dbed301adf90976b67688d918551) (commit `41f0a07…`).

Diagnóstico por ojo: 0 = sano, 1 = glaucoma, 2 = sospechoso (**excluido**). De 488 imágenes se excluyen 68 y quedan **420 imágenes de 210 pacientes**: 333 sanas (79 %) y 87 con glaucoma (21 %). Al estar desbalanceado, la exactitud sola engaña (responder siempre «sano» daría ~79 %).

Los datos **no se versionan**; se colocan en `data/papila/` (`FundusImages/` y `ClinicalData/` con `patient_data_od.xlsx` y `patient_data_os.xlsx`).

## Manifiesto y partición

```bash
python -m src.prepare_papila --dataset-dir data/papila --output data/papila_manifest.csv --seed 42
```

El script valida nombres, JPEG, una etiqueta por imagen y coherencia entre planillas antes de escribir. El CSV tiene `image_id`, `patient_id`, `eye`, `label`, `relative_path` y `split`.

La partición es **por paciente**: se agrupa por perfil de diagnósticos de sus ojos, se baraja con semilla 42 y se reparte ≈ 65/15/20. Resultado (`manifest-c.csv`, SHA-256 `c82776614b94d458e42e56b8f66270e1fbe0be9b4a80a741e2cb6923449ef5cd`):

| Split | Pacientes | Imágenes | Sanos | Glaucoma |
|---|---:|---:|---:|---:|
| Train | 137 | 274 | 217 | 57 |
| Val | 31 | 62 | 49 | 13 |
| Test | 42 | 84 | 67 | 17 |

> El notebook usó otra partición (272/64/84): sus métricas **no son comparables** con las de `src/`.

## Carga (`src/data.py`)

`make_loaders(...)` devuelve cargadores de train y val (test solo con `include_test=True`). Rechaza manifiestos con columnas faltantes, rutas inválidas, duplicados o pacientes en más de un split.

Preprocesamiento: RGB → redimensionar (o *letterbox* con relleno negro) → tensor [0,1] → volteo horizontal aleatorio solo en train → normalizar. La CNN normaliza con estadísticas de train; los modelos preentrenados, con las de ImageNet (media `0.485, 0.456, 0.406`, desviación `0.229, 0.224, 0.225`). Valores por defecto: lote 16, `image_size` 96, semilla 42.

La caché opcional (`--cache-preprocessed`) guarda en memoria la salida determinista previa al aumento y requiere `--num-workers 0`.

---
