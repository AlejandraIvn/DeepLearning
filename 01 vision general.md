# 01 · Visión general

## Qué hace?

GlaucoAI recibe una imagen de fondo de ojo y devuelve la clase (**sano** o **glaucoma**), un puntaje y un mapa Grad-CAM. Tiene tres partes:

1. **Pipeline de aprendizaje profundo** (`src/`): prepara datos, entrena y evalúa.
2. **Demo web** (`backend/` FastAPI y `frontend/` React).
3. **Configuraciones congeladas** (`configs/`): modelos servidos, hashes, umbrales y campañas.

## Modelos

| | CNN de referencia | ResNet18 (respaldo) | **ResNet34 (demo)** |
|---|---|---|---|
| Origen | Desde cero | ImageNet (torchvision) | ImageNet (torchvision) |
| Entrada | 96 × 96 px | 224 × 224 px, *letterbox* | 224 × 224 px, *letterbox* |
| Se entrena | Toda la red | Solo la capa final (513 parámetros) | Cabeza y `layer4` (13,1 M de 21,3 M) |
| Umbral | 0,50 | 0,35 | 0,487 |

EfficientNet-B0 se evaluó en las campañas pero no se sirve. Ver [03](03-modelos-y-entrenamiento.md) y [11](11-campanas-experimentales.md).

## Flujo

```
PAPILA → prepare_papila.py → manifiesto CSV → data.py → train.py → checkpoint
      → compare.py / internal_test.py (evaluación) · explain.py (Grad-CAM) · backend → frontend
```

En la demo: el navegador envía los bytes de la imagen a `POST /api/analyze`; FastAPI valida y decodifica, aplica el preprocesamiento del checkpoint, calcula puntaje y Grad-CAM y devuelve JSON con los PNG en Base64 ([06](06-api-y-frontend.md)).

## Decisiones clave

- **Partición por paciente:** los dos ojos de un paciente van al mismo conjunto, para evitar fuga de información.
- **Reproducibilidad:** semilla 42, algoritmos deterministas y huellas SHA-256 de manifiesto, imágenes y checkpoints.
- **Logits y `BCEWithLogitsLoss`:** más estable numéricamente; la sigmoide solo se aplica al puntaje.
- **Puntajes sin calibrar:** no son probabilidades clínicas.

## Limitaciones

- El test tiene **84 imágenes (17 con glaucoma)**: un caso mueve la sensibilidad unos 6 puntos.
- Un solo split, sin validación cruzada.
- El test se consultó varias veces y **la ResNet34 se eligió viendo sus resultados**; la evaluación es interna y exploratoria.
- La ResNet34 detecta 13 de 17 glaucomas (76,5 %), bajo la meta de 85 % del equipo.
- Sin validación clínica ni externa.

## Tecnologías

Python 3.14, PyTorch 2.14.1, torchvision 0.29.1, scikit-learn, FastAPI, React 19, Vite 8, Docker, Render y Vercel.

---
