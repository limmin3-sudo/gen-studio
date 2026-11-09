"""FastAPI app: web editor, REST API and the MCP endpoint at /mcp."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings

from . import __version__, router
from .comfy import ComfyClient
from .config import ROOT, Settings, access_token, load_settings
from .jobs import EditRequest, Engine, GenerateRequest
from .library import Library
from .registry import Registry
from .security import TokenGate
from .workflows import REQUIRED_NODES

WEB_DIR = ROOT / "web"


@dataclass
class State:
    settings: Settings
    registry: Registry
    library: Library
    comfy: ComfyClient
    engine: Engine
    base_url: str = ""
    extra: dict = field(default_factory=dict)

    def public_base(self) -> str:
        return self.base_url or f"http://127.0.0.1:{self.settings.port}"


def create_app(settings: Settings | None = None, *, comfy_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or load_settings()
    registry = Registry.load()
    library = Library(settings.output_dir)
    comfy = ComfyClient(settings.comfy_url, transport=comfy_transport)
    engine = Engine(settings, registry, library, comfy)
    state = State(settings, registry, library, comfy, engine)

    from .mcp_tools import build_mcp
    mcp = build_mcp(state)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        async with mcp.session_manager.run():
            yield
        await engine.stop()
        await comfy.close()

    app = FastAPI(title="Gen Studio", version=__version__, lifespan=lifespan)
    app.state.gs = state
    app.add_middleware(TokenGate, token_getter=access_token, remote_enabled_getter=lambda: settings.remote_access)

    # ---------------------------------------------------------------- status
    @app.get("/api/status")
    async def status():
        stats = await comfy.ping()
        return {
            "version": __version__,
            "comfyui": {"url": settings.comfy_url, "connected": stats is not None,
                        "devices": [d.get("name") for d in (stats or {}).get("devices", [])]},
            "models_dir": str(settings.models_dir) if settings.models_dir else None,
            "remote_access": settings.remote_access,
            "queue": {"running": engine._running, "waiting": sum(1 for j in engine.jobs.values() if j.status == "queued")},
        }

    @app.get("/api/models")
    async def models():
        return registry.status(settings.models_dir)

    @app.get("/api/doctor")
    async def doctor():
        """Checks that ComfyUI has every node and file our workflows use."""
        report = {"comfyui": False, "missing_nodes": [], "models": registry.status(settings.models_dir)}
        stats = await comfy.ping()
        if stats is None:
            return report
        report["comfyui"] = True
        info = await comfy.object_info()
        report["missing_nodes"] = [n for n in REQUIRED_NODES if n not in info]
        return report

    # ------------------------------------------------------------ generation
    @app.post("/api/route")
    async def route_preview(req: GenerateRequest):
        return engine.route(req).as_dict()

    @app.post("/api/generate")
    async def generate(req: GenerateRequest):
        try:
            return engine.submit_generate(req).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    @app.post("/api/edit")
    async def edit(req: EditRequest):
        try:
            return engine.submit_edit(req).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/jobs")
    async def jobs():
        items = sorted(engine.jobs.values(), key=lambda j: j.created_at, reverse=True)[:50]
        return [j.public() for j in items]

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str):
        j = engine.jobs.get(job_id)
        if not j:
            raise HTTPException(404, "작업을 찾을 수 없어요")
        d = j.public()
        d["queue_position"] = engine.queue_position(job_id)
        return d

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel(job_id: str):
        return {"canceled": await engine.cancel(job_id)}

    # --------------------------------------------------------------- library
    @app.get("/api/images")
    async def images(limit: int = 60, offset: int = 0, roots_only: bool = False):
        return library.list(limit=min(limit, 500), offset=offset, roots_only=roots_only)

    @app.get("/api/images/{image_id}")
    async def image(image_id: str):
        rec = library.get(image_id)
        if not rec:
            raise HTTPException(404, "이미지를 찾을 수 없어요")
        return {**rec, "versions": library.versions(image_id)}

    @app.get("/api/images/{image_id}/file")
    async def image_file(image_id: str, download: bool = False):
        p = library.path(image_id)
        if not p:
            raise HTTPException(404, "이미지를 찾을 수 없어요")
        return FileResponse(p, media_type="image/png", filename=p.name if download else None)

    @app.get("/api/images/{image_id}/thumb")
    async def image_thumb(image_id: str):
        p = library.thumb_path(image_id)
        if not p:
            raise HTTPException(404, "이미지를 찾을 수 없어요")
        return FileResponse(p, headers={"Cache-Control": "max-age=86400"})

    # ------------------------------------------------------------ characters
    @app.get("/api/characters")
    async def characters():
        return [{**c, "thumbs": [f"/api/images/{r}/thumb" for r in c["ref_ids"]]} for c in engine.characters.list()]

    @app.post("/api/characters")
    async def save_character(body: dict):
        try:
            return engine.characters.save(body.get("name", ""), body.get("ref_ids") or [], body.get("description", ""),
                                          body.get("id"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/characters/{cid}")
    async def delete_character(cid: str):
        return {"deleted": engine.characters.delete(cid)}

    @app.post("/api/images/{image_id}/export")
    async def export_image(image_id: str, body: dict):
        try:
            return library.export(image_id, Path(body["path"]), body.get("format"), body.get("max_side"),
                                  body.get("width"), body.get("height"), int(body.get("quality", 90)),
                                  bool(body.get("overwrite", False)))
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    @app.post("/api/images/{image_id}/regenerate")
    async def regenerate(image_id: str):
        try:
            return engine.regenerate(image_id).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    @app.delete("/api/images/{image_id}")
    async def delete_image(image_id: str):
        return {"deleted": library.delete(image_id)}

    @app.post("/api/import")
    async def import_image(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > 40 * 1024 * 1024:
            raise HTTPException(413, "40MB보다 큰 이미지는 불러올 수 없어요")
        try:
            return library.import_image(data, file.filename or "image.png")
        except Exception as e:
            raise HTTPException(400, f"이미지를 읽을 수 없어요: {e}")

    # ------------------------------------------------------------------- MCP
    # use the route itself (not a mount) so POST /mcp works without a redirect to /mcp/
    app.router.routes.extend(mcp_app.routes)

    # ------------------------------------------------------------------- web
    @app.exception_handler(ValueError)
    async def value_error(_req, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
