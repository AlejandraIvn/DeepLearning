# 05 · Explicabilidad con Grad-CAM

Grad-CAM (Selvaraju et al., 2017) produce un mapa de calor con las regiones que más contribuyeron al puntaje de glaucoma.

## Cómo se calcula

1. Se aplica el preprocesamiento del checkpoint (*letterbox* 224 × 224).
2. Se capturan las activaciones de la capa objetivo: `backbone.layer4[-1]` (512 × 7 × 7) en ResNet18/34 y `backbone.features[-1]` en EfficientNet-B0.
3. Se deriva el **logit de glaucoma** respecto a esas activaciones y se promedian los gradientes por canal.
4. Se suman los canales ponderados y se aplica ReLU (solo evidencia positiva).
5. El mapa de 7 × 7 se interpola, se recorta con la geometría exacta del *letterbox* y se normaliza a [0, 1].

Cada mapa tiene su propia escala; si es uniforme, la demo lo avisa (`has_signal = false`). Explica siempre el logit de glaucoma, incluso si la predicción es «sano». Debe ejecutarse fuera de `torch.inference_mode()`, con el modelo en evaluación y FP32.

```python
model, checkpoint = load_checkpoint("backend/predict/resnet34.pt")
result = explain_image(model, imagen_pillow, checkpoint, threshold=0.48704302310943604)
```

## Visualización y validación

`explain_visualization.py` genera la superposición (mapa `magma`), el mapa solo y figuras de tres paneles. `explain_validation.py` genera ejemplos TP/TN/FP/FN de validación; **no se actualizó para la ResNet34** y sigue usando la ResNet18 (umbral 0,35).

## Limitaciones

- La cuadrícula de 7 × 7 solo permite patrones espaciales gruesos.
- La campaña final halló mapas con máximos periféricos en algunos ejemplos y un mapa uniforme en un falso negativo ([11](11-campanas-experimentales.md)).
- Para evaluar disco y copa harían falta los contornos de expertos que publica PAPILA.

---

