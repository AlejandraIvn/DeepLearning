# 07 · Despliegue

| Modo | Qué corre |
|---|---|
| Local | API y frontend en un proceso ([06](06-api-y-frontend.md)) |
| Docker | Lo mismo, solo CPU |
| Nube | API en Render y frontend en Vercel |

Todos sirven por defecto la **ResNet34** con la ResNet18 de respaldo.

## Docker

```powershell
docker build --platform linux/amd64 -t glaucoai:local .
docker run --rm -p 127.0.0.1:8000:8000 glaucoai:local
```

El `Dockerfile` usa dos etapas: compila el frontend (Node 24) y arma la imagen final (Python 3.14, PyTorch CPU) con `backend/`, `src/`, `configs/` y los dos checkpoints (`resnet34.pt` y `best.pt`). Corre como usuario sin privilegios (`10001`), con `HEALTHCHECK` sobre `/api/health`.

## Render (API)

`render.yaml` define el servicio `glaucoai-api`: Docker, plan `free` (0,1 CPU, 512 MB de RAM), salud en `/api/health`, sin despliegue automático. Pasos: crear un Blueprint desde `render.yaml`, desplegar manualmente y comprobar `/api/health` (`model_ready: true`, `model: "resnet34"`). Con `RENDER=true` corre solo como API con CORS abierto.

## Vercel (frontend)

Importa el repositorio con **`frontend`** como *Root Directory* y Node 24. Define `VITE_API_URL` con la URL HTTPS de Render (se aplica al compilar; un cambio exige redesplegar). Verifica que el dominio de producción sea público.

---