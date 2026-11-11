# 12 · Frontend y recorrido educativo

## Tecnología

React 19 y Vite 8, JavaScript/JSX, CSS propio, fuente Nunito (SIL OFL 1.1) y ESLint. El estado vive en `useState`; no hay biblioteca de componentes.

```text
frontend/src/  main.jsx · App.jsx · NetworkJourney.jsx · App.css · NetworkJourney.css · index.css
```

## Pantalla principal (`App.jsx`)

Presentación, tarjeta **Carga la imagen**, tarjeta **Resultado**, recorrido educativo, «¿Cómo funciona?» y aviso de proyecto académico.

| Paso | Qué ocurre |
|---|---|
| Selección | Arrastrar o elegir JPG/PNG; se rechazan archivos vacíos o > 10 MiB (el backend vuelve a validar) |
| Análisis | `POST` de los bytes a `${VITE_API_URL}/api/analyze` con el `Content-Type` del archivo; cancelable y con límite de 60 s |
| Resultado | Medidor del puntaje con el umbral, clase, modelo utilizado, puntaje y umbral |
| Vistas | Original, Superposición Grad-CAM (por defecto) y Mapa de calor |
| Avisos | Mapa uniforme (`has_signal` falso) y recordatorio de que no es un diagnóstico |

Accesibilidad: etiquetas, `role="alert"` en errores, `aria-busy` al cargar y textos alternativos.

## Comandos y variables

```bash
npm --prefix frontend ci
npm --prefix frontend run dev -- --host 127.0.0.1 --strictPort   # Vite reenvía /api/ al puerto 8000
npm --prefix frontend run lint
npm --prefix frontend run build                                  # genera frontend/dist
```

`VITE_API_URL` (origen de la API, sin secretos) se incorpora al **compilar**; vacía, usa el mismo origen. Despliegue en [07](07-despliegue.md).

## Recorrido «Dentro de la red neuronal» (`NetworkJourney.jsx`)

Sección didáctica de ocho pasos. Lo importante es distinguir qué es real y qué es ilustrativo:

| Pasos | Contenido | Naturaleza |
|---|---|---|
| 1–4 | *Letterbox* 224 × 224, canales RGB, convolución 3 × 3 (Sobel, Laplaciano…), ReLU y *max pooling* | **Calculado en el navegador** sobre la imagen subida |
| 5–6 | Capas profundas (`layer1`…`layer4`) y vector de 512 valores | **Ilustración**, no los canales reales del modelo |
| 7 | Logit, sigmoide y umbral | **Real**: el logit se obtiene como `ln(p / (1 − p))` del puntaje de la API |
| 8 | Grad-CAM | **Real**, de la API |

Los textos de los pasos 5–6 hablan de ResNet34 aunque se sirva la ResNet18, y lo que «ve» cada capa (vasos, disco, copa) es una explicación pedagógica no verificada sobre los pesos.

## Limitaciones

El frontend no infiere (depende de la API); la validación del navegador no sustituye a la del servidor; Grad-CAM no demuestra relevancia clínica; no hay pruebas automáticas del frontend.

---
