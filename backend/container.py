"""Container entry point for the local demo and Render's API-only service."""

import os

import uvicorn

from .main import FRONTEND_DIST, create_app, is_remote


if __name__ == "__main__":
    uvicorn.run(
        create_app(None if is_remote() else FRONTEND_DIST),
        host="0.0.0.0", port=int(os.getenv("PORT", "8000")),
        lifespan="on", access_log=False, proxy_headers=False,
    )
