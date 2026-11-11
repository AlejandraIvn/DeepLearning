# 08 · Pruebas automáticas

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m unittest tests.test_data -v          # un solo archivo
```

Cinco pruebas requieren CUDA y se omiten (*skipped*) sin GPU.

| Archivo | N.º | Verifica |
|---|---:|---|
| `test_prepare_papila.py` | 13 | Etiquetas, exclusión de sospechosos, partición por paciente |
| `test_explain.py` | 11 | Alineación del Grad-CAM, recorte del *letterbox*, mapas nulos |
| `test_backend.py` | 10 | Endpoints, límites de carga, selección de modelo, hash y arquitectura |
| `test_data.py` | 9 | Lotes, normalización, orden reproducible, caché |
| `test_training.py` | 6 | Forma de la CNN, entrenamiento CPU/CUDA, parada temprana |
| `test_transfer_learning.py` | 6 | *Letterbox*, ImageNet, ajuste del último bloque |
| `test_internal_evaluation.py` | 6 | Métricas indefinidas, *bootstrap* por paciente, protocolo |
| `test_demo.py` | 5 | Arranque con frontend, checkpoint ausente, respaldo ResNet18 |
| `test_final_campaign.py` | 4 | Telemetría, acumulación de gradientes, *dropout*, *bootstrap* |
| `test_research.py` | 3 | Selección de umbral y recarga de checkpoints ResNet34/EfficientNet |
| `test_device.py` | 2 | Respaldo a CPU, memoria fijada |
| `test_deployment.py` | 2 | Render exige checkpoint; CORS de `/analyze` |

Recuento al 10-nov-2026 (commit `8f2d42d`; los commits posteriores solo cambian el frontend).

## Qué no cubren

- Los resultados numéricos del experimento ni su validez clínica.
- El frontend: no tiene pruebas automáticas; se verifica con `npm run lint` y `npm run build`.
- `internal_test.py` solo se prueba con datos sintéticos.

## Añadir una prueba

Crea `tests/test_<modulo>.py` con una clase `unittest.TestCase` y métodos `test_...`; si necesita GPU, usa `@unittest.skipUnless(torch.cuda.is_available(), ...)`.

---
