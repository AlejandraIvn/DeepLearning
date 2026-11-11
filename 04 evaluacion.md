# 04 · Evaluación

Diseño: **comparar y fijar decisiones en validación; evaluar después en test**, verificando con SHA-256 que datos y modelos no cambiaron.

## Paso 1: comparación en validación

Recarga los checkpoints, los evalúa sobre las mismas imágenes de validación y guarda `comparison.json`, `curves.png`, `predictions.csv` y `thresholds.csv`.

**Umbral exploratorio:** rejilla de 0,00 a 1,00 en pasos de 0,05; entre los de especificidad ≥ 80 % se elige el de mayor sensibilidad. (Las campañas posteriores usan los puntos medios entre puntajes y maximizan el mínimo entre sensibilidad y especificidad.)

## Paso 2: evaluación congelada

`configs/papila-internal-test.json` fija hashes del manifiesto, imágenes y checkpoints, umbrales (ResNet18 0,35; CNN 0,50), versiones de librerías y equipo. Dos modos: `--preflight` (sin decodificar imágenes de test) y `--confirm-internal-test` (una pasada; la carpeta de salida se reserva antes y no se puede repetir). **Solo corre en el equipo de referencia** (RTX 4050, versiones exactas).

**Intervalos de confianza:** *bootstrap* de 5 000 remuestreos, semilla 42, **por paciente** (ambos ojos juntos), percentiles del 95 %.

## Resultados

| Partición | Modelo | Umbral | Sensibilidad | Especificidad | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Validación (62) | CNN | 0,50 | 38,5 % | 89,8 % | 0,739 |
| Validación (62) | ResNet18 | 0,35 | 69,2 % | 93,9 % | 0,859 |
| Validación (62) | ResNet34 | 0,487 | 84,6 % | 93,9 % | 0,898 |
| Test (84) | CNN | 0,50 | 47,1 % (8/17) | 88,1 % | 0,716 |
| Test (84) | ResNet18 | 0,35 | 35,3 % (6/17) | 94,0 % | 0,795 |
| Test (84) | **ResNet34** | 0,487 | **76,5 % (13/17)** | **92,5 %** | **0,873** |

- Los modelos con ResNet separan mejor las clases que la CNN.
- La ResNet34 queda bajo la meta de 85 % de sensibilidad; su intervalo es 50–100 %.
- Con 17 positivos, una imagen cambia la sensibilidad ~6 puntos.
- Los resultados de la ResNet34 provienen de los informes del equipo; `internal_test.py` solo cubre CNN y ResNet18.

*Nota:* cuando una métrica no está definida, `train.py` devuelve `0,0` e `internal_test.py` devuelve `None`.

---
