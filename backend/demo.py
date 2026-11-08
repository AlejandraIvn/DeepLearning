"""Start the compiled GlaucoAI demo on Windows or another local host."""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Inicia GlaucoAI en localhost.")
    parser.add_argument("--cpu", action="store_true", help="Usar CPU aunque CUDA esté disponible.")
    parser.add_argument("--port", type=int, default=8000, help="Puerto local (8000 por defecto).")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 1 y 65535.")
    if args.cpu:
        os.environ["GLAUCOAI_CPU"] = "1"

    try:
        import uvicorn
        from .main import FRONTEND_DIST, create_app
    except (ImportError, OSError, RuntimeError) as error:
        print("No se pudieron importar las dependencias de GlaucoAI. Revisa la instalación de PyTorch, "
              f"torchvision y requirements-backend.txt. Detalle: {error}", file=sys.stderr)
        return 1

    try:
        app = create_app(FRONTEND_DIST)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    # The server reports the URL only after lifespan has verified and loaded the model.
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=args.port, lifespan="on", access_log=False, proxy_headers=False,
    ))
    server.run()
    return 0 if server.started else 1


if __name__ == "__main__":
    sys.exit(main())
