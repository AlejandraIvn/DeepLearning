# 06 · API y ejecución de la demo

```
Navegador (React) ──POST imagen──► FastAPI (/api/analyze) ──► modelo + Grad-CAM
          ◄──── JSON: modelo, clase, puntaje, umbral, PNG en Base64 ────
```

## Arranque local (Windows)

Requisitos: Python 3.14 y Node.js 24.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe -m backend.demo
```

Abre <http://127.0.0.1:8000/>. Opciones: `--cpu`, `--port`. Para la GPU usa el índice `cu130` de PyTorch. Para el respaldo: `$env:GLAUCOAI_MODEL = 'resnet18'`.

## Endpoints

También existen sin prefijo `/api` (alias). Los `POST` reciben **los bytes de la imagen en el cuerpo** (`Content-Type: image/jpeg` o `image/png`), no multipart.

| Ruta | Función | Respuesta |
|---|---|---|
| `GET /api/health` | Estado | `status`, `model_ready`, `model`, `device`, `threshold` (503 si no hay modelo) |
| `POST /api/predict` | Clasificación | `model`, `score_glaucoma`, `label`, `classification`, `threshold` |
| `POST /api/explain` | Grad-CAM | `heatmap_png`, `overlay_png`, `target_layer`, `has_signal` |
| `POST /api/analyze` | Ambos en una pasada | Todos los campos |

`label = 1` (glaucoma) si `score_glaucoma >= threshold`. El umbral es el del modelo cargado (0,487 ResNet34; 0,35 ResNet18).

```python
import requests
with open("retinografia.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/api/analyze", data=f,
                      headers={"Content-Type": "image/jpeg"}, timeout=60)
print(r.json()["classification"], r.json()["score_glaucoma"])
```

## Límites y errores

| Condición | Código |
|---|---|
| Tipo distinto de JPEG/PNG | 415 |
| > 10 MiB, > 6 megapíxeles o > 4096 px por lado | 413 |
| Imagen vacía, ilegible, animada o proporciones extremas | 422 |
| Cuerpo tarda más de 15 s / carga interrumpida | 408 / 400 |
| Modelo no disponible / fallo interno | 503 / 500 |

## Comportamiento interno

- **Arranque seguro:** lee `configs/glaucoai-models.json`, elige el modelo (`resnet34` por defecto o `GLAUCOAI_MODEL`) y compara el SHA-256 del checkpoint; si no coincide, no sirve predicciones.
- **Concurrencia:** hasta dos solicitudes en proceso o espera; un único hilo ejecuta el modelo (aísla los *hooks* de Grad-CAM).
- **Privacidad:** imágenes en memoria, sin registro de acceso y con `Cache-Control: no-store`.
- **Variables:** `GLAUCOAI_CPU=1`, `GLAUCOAI_MODEL`, `RENDER=true` (solo API con CORS abierto a GET y POST), `PORT`.

## Rendimiento

Medido con la ResNet18 (`scripts/benchmark_demo.py`, 12 solicitudes por fase): `/api/analyze` tarda ~282 ms en CPU y ~243 ms en GPU.

---
