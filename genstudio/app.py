"""FastAPI app: web editor, REST API and the MCP endpoint at /mcp."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings

from . import __version__, router
from .app_routes import set_remote  # noqa: F401  (used by the desktop tray)
from .comfy import ComfyClient, ComfyError
from .config import ROOT, Settings, access_token, load_settings
from .training import TrainRequest
from .jobs import CharacterBuildRequest, CutoutRequest, LtxRequest, ReposeRequest, UpscaleRequest, VideoRequest, EditRequest, Engine, GenerateRequest, SegmentRequest
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
    from .diagnostics import install as _keep_logs
    _keep_logs()
    registry = Registry.load()
    from . import user_models
    user_models.merge(registry)                 # models added from the settings screen
    user_models.fix_locations(registry, settings.models_dir)
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

    @app.get("/api/system")
    async def system():
        """GPU/VRAM/RAM/CPU and the job queue, for the status bar (polled every couple of seconds)."""
        from . import sysinfo
        stats = await comfy.ping()
        snap = await sysinfo.snapshot(stats)
        run = engine.jobs.get(engine._running) if engine._running else None
        snap["comfyui"] = stats is not None
        snap["queue"] = {
            "waiting": sum(1 for j in engine.jobs.values() if j.status == "queued"),
            "running": {"id": run.id, "kind": run.kind, "action": run.decision.get("action") or "생성",
                        "message": run.message, "progress": run.progress} if run else None}
        return snap

    @app.get("/api/diagnostics")
    async def diagnostics():
        """Plain-text report for pasting into a chat when something goes wrong (secrets masked)."""
        from fastapi.responses import PlainTextResponse
        from . import config as cfg
        from .diagnostics import build
        secrets = [str(v) for v in cfg.read_secrets().values() if isinstance(v, str)]
        import os
        secrets += [os.environ[k] for k in ("GENSTUDIO_TOKEN", "CIVITAI_API_KEY") if os.environ.get(k)]
        return PlainTextResponse(await build(app.state.gs, secrets))

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
        # newer features: need a recent ComfyUI (update\\update_comfyui.bat)
        from .training import TRAIN_NODES
        report["missing_optional"] = {k: [n for n in v if n not in info] for k, v in {
            "배경 제거": ["LoadBackgroundRemovalModel", "RemoveBackground"], "LoRA 학습": TRAIN_NODES}.items()
            if any(n not in info for n in v)}
        return report

    # ------------------------------------------------------------ generation
    @app.post("/api/recommend")
    async def recommend(body: dict):
        return engine.recommend(body.get("task", "generate"), body.get("prompt", ""), body.get("image_id"),
                                body.get("style", "auto"), body.get("aspect", "auto"))

    @app.post("/api/route")
    async def route_preview(req: GenerateRequest):
        try:
            return engine.plan(req)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    @app.get("/api/chips")
    async def get_chips():
        from . import chips
        return {"chips": chips.load()}

    @app.put("/api/chips")
    async def put_chips(body: dict):
        """Replace the list; `bulk` ('이름 | 문장' lines) is appended to it."""
        from . import chips
        items = list(body.get("chips") or [])
        if body.get("bulk"):
            items += chips.parse_bulk(str(body["bulk"]))
        try:
            return {"chips": chips.save(items)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/enhance-edit")
    async def enhance_edit(body: dict):
        text = str(body.get("prompt") or "").strip()
        if not text:
            raise HTTPException(400, "먼저 대충이라도 적어 주세요")
        try:
            return await engine.enhance_edit(text, body.get("image_id"), list(body.get("segments") or []))
        except ComfyError as e:
            raise HTTPException(502, str(e))

    @app.post("/api/enhance")
    async def enhance(req: GenerateRequest):
        try:
            return await engine.enhance_prompt(req)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ComfyError as e:
            raise HTTPException(502, str(e))

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
        media = {".mp4": "video/mp4", ".webm": "video/webm", ".webp": "image/webp"}.get(p.suffix.lower(), "image/png")
        return FileResponse(p, media_type=media, filename=p.name if download else None)

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

    @app.post("/api/characters/build")
    async def build_character(body: dict):
        """One photo -> face, full-body and side reference shots, saved together as a character."""
        try:
            req = CharacterBuildRequest(name=body.get("name") or "", image_id=body.get("image_id") or "",
                                        **({"views": body["views"]} if body.get("views") else {}),
                                        aliases=body.get("aliases") or [])
            return engine.submit_character_build(req).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
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

    @app.post("/api/ltx")
    async def ltx(body: dict):
        try:
            return engine.submit_ltx(LtxRequest(**body)).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    @app.post("/api/images/{image_id}/video")
    async def make_video(image_id: str, body: dict | None = None):
        body = body or {}
        try:
            return engine.submit_video(VideoRequest(image_id=image_id, **{k: v for k, v in body.items()
                                                                         if k in ("prompt", "seconds", "quality", "seed", "smooth")})).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/images/{image_id}/upscale")
    async def upscale(image_id: str, body: dict | None = None):
        body = body or {}
        try:
            return engine.submit_upscale(UpscaleRequest(image_id=image_id, **{k: v for k, v in body.items()
                                                                             if k in ("scale", "refine", "denoise", "prompt")})).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/images/{image_id}/detail")
    async def detail(image_id: str, body: dict | None = None):
        from .detail import DetailRequest
        body = body or {}
        try:
            return engine.submit_detail(DetailRequest(image_id=image_id, **{k: v for k, v in body.items()
                                                                           if k in ("targets", "strength", "prompt", "model")})).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/images/{image_id}/repose")
    async def repose(image_id: str, body: dict | None = None):
        body = body or {}
        try:
            return engine.submit_repose(ReposeRequest(image_id=image_id, pose=body.get("pose"),
                                                      prompt=body.get("prompt") or "")).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/images/{image_id}/remove-background")
    async def remove_background(image_id: str, body: dict | None = None):
        body = body or {}
        try:
            return engine.submit_cutout(CutoutRequest(image_id=image_id,
                                                      background=body.get("background") or "transparent")).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    # -------------------------------------------------------------- segments
    @app.post("/api/images/{image_id}/segment")
    async def segment(image_id: str, body: dict | None = None):
        body = body or {}
        try:
            return engine.submit_segment(SegmentRequest(image_id=image_id, labels=body.get("labels"),
                                                        points=body.get("points"),
                                                        negative_points=body.get("negative_points"))).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/images/{image_id}/segments")
    async def segments(image_id: str):
        try:
            return engine.segments.load(image_id)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    def _seg_png(fn, *args):
        try:
            return Response(fn(*args), media_type="image/png", headers={"Cache-Control": "max-age=3600"})
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    @app.get("/api/images/{image_id}/segments/{sid}/mask")
    async def segment_mask(image_id: str, sid: str):
        return _seg_png(engine.segments.alpha_mask, image_id, sid)

    @app.get("/api/images/{image_id}/segments/{sid}/cutout")
    async def segment_cutout(image_id: str, sid: str, size: int | None = None):
        return _seg_png(engine.segments.cutout, image_id, sid, size)

    @app.get("/api/images/{image_id}/segments.zip")
    async def segments_zip(image_id: str):
        import io as _io
        import zipfile
        segs = engine.segments.load(image_id)
        if not segs:
            raise HTTPException(404, "세그먼트가 없어요. 먼저 분석하세요.")
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, s in enumerate(segs, 1):
                name = f"{i:02d}-{s['label'].replace(' ', '_')}{'-' + str(s['instance']) if s.get('instance') else ''}.png"
                z.writestr(name, engine.segments.cutout(image_id, s["id"]))
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{image_id}-segments.zip"'})

    # ----------------------------------------------------------------- LoRAs
    @app.get("/api/loras")
    async def loras_list():
        return {"loras": engine.loras.list(), "downloads": list(engine.loras.downloads.values()),
                "folder": str(engine.loras.dir) if engine.loras.dir else None}

    @app.post("/api/loras/civitai")
    async def loras_civitai(body: dict):
        try:
            return {"download_id": await engine.loras.add_from_civitai(body.get("url", ""))}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/loras/upload")
    async def loras_upload(file: UploadFile = File(...)):
        try:
            return await engine.loras.add_file(file.filename or "lora.safetensors", await file.read())
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/loras/train")
    async def loras_train(req: TrainRequest):
        try:
            return engine.submit_train(req).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/loras/scan")
    async def loras_scan():
        return {"added": await engine.loras.scan()}

    @app.patch("/api/loras/{lid}")
    async def loras_update(lid: str, body: dict):
        try:
            return engine.loras.update(lid, **body)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    @app.delete("/api/loras/{lid}")
    async def loras_delete(lid: str, delete_file: bool = False):
        return {"deleted": engine.loras.remove(lid, delete_file)}

    @app.post("/api/images/{image_id}/regenerate")
    async def regenerate(image_id: str):
        try:
            return engine.regenerate(image_id).public()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))

    @app.delete("/api/images/{image_id}")
    async def delete_image(image_id: str, all_versions: bool = False):
        if all_versions:
            return {"deleted": library.delete_versions(image_id)}
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
    from .app_routes import register as _register_app_routes
    _register_app_routes(app, state)

    # use the route itself (not a mount) so POST /mcp works without a redirect to /mcp/
    app.router.routes.extend(mcp_app.routes)

    # ------------------------------------------------------------------- web
    @app.exception_handler(ValueError)
    async def value_error(_req, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    async def index_page():
        from fastapi.responses import HTMLResponse
        v = str(int(max((WEB_DIR / f).stat().st_mtime for f in ("app.js", "style.css", "index.html"))))
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8").replace("__V__", v),
                            headers={"Cache-Control": "no-cache"})

    @app.middleware("http")
    async def no_stale_ui(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
