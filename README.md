# Documentación de GlaucoAI

GlaucoAI es un proyecto académico de Deep Learning que clasifica imágenes de fondo de ojo en **sano** o **glaucoma** y explica la decisión con mapas Grad-CAM. Esta documentación describe el repositorio en `main` (commit `1f7e970`, 10-nov-2026).

> **Advertencia:** no es un instrumento de diagnóstico clínico.

## Por dónde empezar

| Si quieres… | Lee |
|---|---|
| Un resumen académico completo | [Informe técnico](informe-tecnico.md) |
| Entender el proyecto en cinco minutos | [01 · Visión general](01-vision-general.md) |
| Conocer los datos y la partición | [02 · Datos](02-datos.md) |
| Ver modelos y cómo se entrenan | [03 · Modelos y entrenamiento](03-modelos-y-entrenamiento.md) |
| Ver cómo se evalúa | [04 · Evaluación](04-evaluacion.md) |
| Entender Grad-CAM | [05 · Explicabilidad](05-explicabilidad-gradcam.md) |
| Usar la API | [06 · API](06-api-y-frontend.md) |
| Desplegar la demo | [07 · Despliegue](07-despliegue.md) |
| Ejecutar las pruebas | [08 · Pruebas](08-pruebas.md) |
| Ver los experimentos del notebook | [09 · Notebook](09-experimentos-notebook.md) |
| Trabajar en equipo con Git | [10 · Guía del equipo](10-guia-del-equipo.md) |
| Saber por qué se eligió la ResNet34 | [11 · Campañas experimentales](11-campanas-experimentales.md) |
| Entender la interfaz web | [12 · Frontend](12-frontend-y-recorrido-educativo.md) |

El equipo también escribió tres informes de experimentos en esta carpeta: [`optimizacion-papila.md`](optimizacion-papila.md), [`investigacion-papila.md`](investigacion-papila.md) y [`campana-final-papila.md`](campana-final-papila.md).

## Glosario mínimo

| Término | Significado |
|---|---|
| **PAPILA** | Dataset público de fondo de ojo con diagnóstico clínico |
| **Manifiesto** | CSV que asigna cada imagen a `train`, `val` o `test` |
| **Logit / puntaje** | Salida de la red / su sigmoide (0 a 1, sin calibrar) |
| **Umbral** | Puntaje desde el cual se predice glaucoma (0,487 en la ResNet34) |
| **Sensibilidad** | De los ojos con glaucoma, cuántos se detectan |
| **Especificidad** | De los ojos sanos, cuántos se clasifican bien |
| **ROC-AUC** | Qué tan bien separa el puntaje a las dos clases, sin depender del umbral |
| **Grad-CAM** | Mapa de calor de las zonas que pesaron en el puntaje |
| **Checkpoint** | Archivo `.pt` con los pesos entrenados |
| **SHA-256** | Huella digital de un archivo, para comprobar que no cambió |

---

*Nota: este documento fue elaborado con el apoyo de Claude (IA de Anthropic) para redactar y organizar su contenido.*
