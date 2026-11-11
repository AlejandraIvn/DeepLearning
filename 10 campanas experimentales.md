# 10 · Campañas experimentales y elección de la ResNet34

## Rondas (test: 84 ojos, 17 con glaucoma)

| Ronda | Qué se probó | Resultado en test |
|---|---|---|
| Base | ResNet18 solo cabeza (0,35) | Sens. 35,3 % · espec. 94,0 % |
| Optimización | `layer4` ajustable, BCE ponderada, focal | `layer4`: 70,6 % / 74,6 %; ponderada: 58,8 % / 88,1 % |
| Investigación | ResNet18/34 y EfficientNet-B0, 3 semillas | EfficientNet-B0: 70,6 % / 85,1 % |
| Final | Hasta 64 corridas, 3 semillas por finalista | EfficientNet-B0: 76,5 % / 67,2 %; **ResNet34: 76,5 % / 92,5 %** |

**Campaña final:** meta ≥ 85 % de sensibilidad y especificidad; cribado por etapas (arquitectura y bloques → pérdida → optimizador y tasa → aumento); umbral por puntos medios maximizando el mínimo entre ambas tasas; confirmación con semillas 42/43/44; archivo `test-lock.json` que fija checkpoints y umbrales antes de puntuar el test; intervalos por paciente (4 000 remuestras). Se completaron 62 de 64 corridas.

En validación, tres recetas quedaron casi empatadas (sensibilidad media 84,6–87,2 %) y cada una cumplió 85/85 solo en una semilla. El criterio eligió EfficientNet-B0 por 0,37 puntos.

## La decisión

GlaucoAI sirve el finalista 3, **ResNet34** (`resnet34-lr-low-s43`): detecta 13 de 17 glaucomas con 5 falsos positivos (ROC-AUC 0,873). Hay que señalar ante el profesor que:

1. **La elección se hizo después de ver el test** (lo dice el propio informe), contra la recomendación de la campaña de conservar la ResNet18 y confirmar la ResNet34 en pacientes independientes.
2. El **umbral sí se fijó solo con validación** (0,487), pero el modelo se escogió con información del test: sus cifras son **optimistas**.
3. La sensibilidad (76,5 %) queda bajo la meta de 85 % y su intervalo al 95 % va de **50 % a 100 %**; la especificidad, de 86,4 % a 98,4 %.
4. Hace falta una **cohorte independiente** para confirmarla.

## Errores (candidato de la campaña)

Los 4 falsos negativos de test (`RET021OS`, `RET026OD`, `RET026OS`, `RET088OS`, tres pacientes) fueron fallados también por los controles. Los 26 errores afectan a 20 de 42 pacientes, por lo que el remuestreo debe hacerse por paciente.

---