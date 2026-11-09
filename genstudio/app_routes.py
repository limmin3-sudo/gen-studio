"""Settings-screen API: updates, model downloads, ComfyUI, remote access, Codex. Only from this PC.

These change the installation itself, so a phone or laptop coming in over Tailscale gets 403 here even with
the access token.
"""
from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from . import __version__, runtime
from .config import Settings, access_token, write_secret
from .security import is_local_request

CODEX_CONFIG = Path.home() / ".codex" / "config.toml"


def set_remote(settings: Settings, on: bool) -> dict:
    """Remote access on/off: the token gate follows settings.remote_access at once, Tailscale Serve publishes it."""
    settings.remote_access = bool(on)
    runtime.save_prefs(remote_access=bool(on))
    ok, msg = runtime.tailscale_serve(settings.port, on)
    return {"on": settings.remote_access, "ok": ok, "message": msg}


def _remote_info(settings: Settings) -> dict:
    name = runtime.tailscale_name()
    url = f"https://{name}" if name else None
    return {"on": settings.remote_access, "tailscale": bool(runtime.tailscale_exe()), "logged_in": bool(name),
            "url": url, "link": f"{url}/?token={access_token()}" if url and settings.remote_access else None}


def _codex_connected() -> bool:
    try:
        return "[mcp_servers.gen_studio]" in CODEX_CONFIG.read_text(encoding="utf-8")
    except OSError:
        return False


def register(app: FastAPI, state) -> None:
    s: Settings = state.settings
    downloads = runtime.Downloads(s, state.registry)
    state.extra.setdefault("downloads", downloads)

    def local_only(request: Request) -> None:
        if not is_local_request(request):
            raise HTTPException(403, "설정은 Gen Studio가 켜진 PC에서만 바꿀 수 있어요")

    def comfy_proc() -> runtime.ComfyProcess:
        p = state.extra.get("comfy_proc")
        if p is None:
            p = state.extra["comfy_proc"] = runtime.ComfyProcess(s)
        return p

    @app.get("/api/app/info")
    async def app_info(request: Request):
        local_only(request)
        p = comfy_proc()
        alive = await state.comfy.ping() is not None
        missing = downloads.missing()
        v = runtime.local_version()
        return {
            "version": __version__, "commit": v.get("commit"), "updated_at": v.get("updated_at"),
            "last_error": v.get("last_error"), "desktop": "desktop" in state.extra,
            "comfy": {"alive": alive, "installed": p.installed, "managed": p.managed,
                      "state": "running" if alive and p.state not in ("updating",) else p.state, "message": p.message,
                      "url": s.comfy_url, "path": str(s.comfy_path) if s.comfy_path else None},
            "remote": _remote_info(s),
            "codex": {"connected": _codex_connected(), "config": str(CODEX_CONFIG),
                      "url": f"http://127.0.0.1:{s.port}/mcp"},
            "models": {"missing": missing, "missing_bytes": sum(m["size"] for m in missing)},
            "downloads": downloads.state,
            "logs": str(runtime.log_dir()),
        }

    @app.get("/api/app/latest")
    async def app_latest(request: Request):
        local_only(request)
        latest = await runtime.latest_version()
        mine = runtime.local_version().get("commit")
        return {"latest": latest, "current": mine,
                "update_available": bool(latest and latest["commit"] != mine)}

    @app.post("/api/app/update")
    async def app_update(request: Request):
        local_only(request)
        desk = state.extra.get("desktop")
        if desk:
            desk.restart_for_update()
        else:
            runtime.spawn_updater(s.port)
            threading.Thread(target=lambda: (time.sleep(1.0), os._exit(0)), daemon=True).start()
        return {"ok": True, "message": "업데이트를 위해 잠시 껐다 다시 켭니다. 1~2분 기다려 주세요."}

    @app.post("/api/app/comfy/{action}")
    async def app_comfy(action: str, request: Request):
        local_only(request)
        p = comfy_proc()
        jobs = {"start": p.start, "restart": p.restart, "update": p.update}
        if action not in jobs:
            raise HTTPException(404, "알 수 없는 동작")
        if state.engine._running:
            raise HTTPException(409, "작업이 끝난 뒤에 해 주세요")
        p.state = "updating" if action == "update" else "starting"
        threading.Thread(target=jobs[action], daemon=True).start()
        return {"state": p.state}

    @app.post("/api/app/remote")
    async def app_remote(body: dict, request: Request):
        local_only(request)
        r = set_remote(s, bool(body.get("on")))
        return {**r, **_remote_info(s)}

    @app.get("/api/app/remote/qr")
    async def app_remote_qr(request: Request):
        local_only(request)
        info = _remote_info(s)
        if not info["link"]:
            raise HTTPException(404, "원격 접속이 꺼져 있어요")
        import qrcode
        img = qrcode.make(info["link"], box_size=8, border=2)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/app/codex")
    async def app_codex(request: Request):
        local_only(request)
        import importlib.util
        spec = importlib.util.spec_from_file_location("connect_codex", runtime.cfg.ROOT / "scripts" / "connect_codex.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        url = f"http://127.0.0.1:{s.port}/mcp"
        action = mod.upsert(CODEX_CONFIG, mod.build_section(url, remote=False))
        return {"connected": True, "message": f"Codex 설정에 Gen Studio를 {action}. Codex 앱을 껐다 켜면 보여요.",
                "config": str(CODEX_CONFIG)}

    @app.post("/api/app/models/download")
    async def app_models_download(request: Request):
        local_only(request)
        try:
            return downloads.start()
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/app/models/added")
    async def app_models_added(request: Request):
        local_only(request)
        from . import user_models
        md = s.models_dir
        return {"models": [{**e, "installed": state.registry.by_id[e["id"]].installed(md) if e["id"] in state.registry.by_id else None}
                           for e in user_models.entries()]}

    @app.post("/api/app/models/add")
    async def app_models_add(body: dict, request: Request):
        """Paste a Civitai checkpoint link: it joins the model list and starts downloading."""
        local_only(request)
        from . import user_models
        try:
            e = await user_models.add_from_civitai(state.registry, str(body.get("url") or ""),
                                                   transport=state.extra.get("civitai_transport"))
        except ValueError as err:
            raise HTTPException(400, str(err))
        started = False
        if s.models_dir and not downloads.state.get("running"):
            downloads.start()
            started = True
        return {"model": e, "downloading": started}

    @app.delete("/api/app/models/added/{model_id}")
    async def app_models_remove(model_id: str, request: Request):
        local_only(request)
        from . import user_models
        return {"removed": user_models.remove(state.registry, model_id)}

    @app.get("/api/app/models/unused")
    async def app_models_unused(request: Request):
        local_only(request)
        files = runtime.unused_model_files(state.registry, s.models_dir)
        return {"files": files, "bytes": sum(f["size"] for f in files)}

    @app.post("/api/app/models/unused/delete")
    async def app_models_unused_delete(body: dict, request: Request):
        local_only(request)
        if state.engine._running:
            raise HTTPException(409, "작업이 끝난 뒤에 지워 주세요")
        allowed = runtime.unused_model_files(state.registry, s.models_dir)
        n = runtime.delete_model_files(s.models_dir, list(body.get("paths") or []), allowed)
        return {"deleted": n}

    @app.post("/api/app/models/cancel")
    async def app_models_cancel(request: Request):
        local_only(request)
        downloads.cancel()
        return {"ok": True}

    @app.post("/api/app/civitai-key")
    async def app_civitai_key(body: dict, request: Request):
        local_only(request)
        key = str(body.get("key") or "").strip()
        if len(key) < 10:
            raise HTTPException(400, "API 키가 너무 짧아요")
        write_secret("civitai_api_key", key)
        return {"ok": True}

    @app.post("/api/app/open-logs")
    async def app_open_logs(request: Request):
        local_only(request)
        if runtime.WIN:
            os.startfile(str(runtime.log_dir()))  # noqa: S606
        return {"path": str(runtime.log_dir())}

    @app.post("/api/app/show")
    async def app_show(request: Request):
        local_only(request)
        desk = state.extra.get("desktop")
        if desk:
            desk.show()
        return {"ok": bool(desk)}
