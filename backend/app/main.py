from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .settings import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()
    app = FastAPI(title="SpecsGrader", version="0.0.1")

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    index_path = frontend_dir / "index.html"
    styles_path = frontend_dir / "styles.css"
    src_dir = frontend_dir / "src"

    @app.get("/api/health", response_class=JSONResponse)
    async def health() -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> Any:
        return index_path.read_text(encoding="utf-8")

    @app.get("/styles.css")
    async def serve_styles() -> FileResponse:
        return FileResponse(styles_path, media_type="text/css")

    app.mount("/src", StaticFiles(directory=src_dir), name="src")

    app.state.settings = settings
    return app


__all__ = ["create_app"]
