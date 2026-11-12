"""Small async client for the ComfyUI HTTP + WebSocket API."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Awaitable, Callable

import httpx

ProgressFn = Callable[[float, str], Awaitable[None] | None]


class ComfyError(RuntimeError):
    pass


class ComfyUnavailable(ComfyError):
    pass


def _fmt_node_errors(body: dict) -> str:
    err = body.get("error") or {}
    lines = [err.get("message", "ComfyUI가 작업을 거부했어요")]
    for nid, ne in (body.get("node_errors") or {}).items():
        for e in ne.get("errors", []):
            lines.append(f"- {ne.get('class_type', nid)}: {e.get('message')} {e.get('details', '')}".strip())
    return "\n".join(lines)


class ComfyClient:
    def __init__(self, base_url: str, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30):
        self.base = base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self._http = httpx.AsyncClient(base_url=self.base, timeout=timeout, transport=transport)
        self.use_websocket = transport is None

    async def close(self) -> None:
        await self._http.aclose()

    async def _req(self, method: str, url: str, **kw) -> httpx.Response:
        try:
            r = await self._http.request(method, url, **kw)
        except httpx.TransportError as e:
            raise ComfyUnavailable(f"ComfyUI에 연결할 수 없어요 ({self.base}). ComfyUI가 켜져 있는지 확인하세요.") from e
        return r

    async def ping(self) -> dict | None:
        try:
            r = await self._req("GET", "/system_stats")
            return r.json() if r.status_code == 200 else None
        except ComfyUnavailable:
            return None

    async def object_info(self) -> dict:
        r = await self._req("GET", "/object_info")
        r.raise_for_status()
        return r.json()

    async def upload_image(self, data: bytes, filename: str, subfolder: str = "gen-studio") -> str:
        mime = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}.get(
            filename[filename.rfind("."):].lower(), "image/png")
        files = {"image": (filename, data, mime)}
        r = await self._req("POST", "/upload/image", files=files, data={"subfolder": subfolder, "overwrite": "true", "type": "input"})
        if r.status_code != 200:
            raise ComfyError(f"이미지 업로드 실패 ({r.status_code}): {r.text[:200]}")
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    async def queue(self, graph: dict) -> str:
        r = await self._req("POST", "/prompt", json={"prompt": graph, "client_id": self.client_id})
        if r.status_code != 200:
            try:
                raise ComfyError(_fmt_node_errors(r.json()))
            except json.JSONDecodeError:
                raise ComfyError(f"ComfyUI 오류 ({r.status_code}): {r.text[:300]}")
        return r.json()["prompt_id"]

    async def history(self, prompt_id: str) -> dict | None:
        r = await self._req("GET", f"/history/{prompt_id}")
        if r.status_code != 200:
            return None
        return r.json().get(prompt_id)

    async def view(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        r = await self._req("GET", "/view", params={"filename": filename, "subfolder": subfolder, "type": type_})
        r.raise_for_status()
        return r.content

    async def free(self) -> None:
        try:
            await self._req("POST", "/free", json={"unload_models": True, "free_memory": True})
        except ComfyError:
            pass

    async def interrupt(self) -> None:
        await self._req("POST", "/interrupt", json={})

    # ------------------------------------------------------------------ run
    async def run(self, graph: dict, on_progress: ProgressFn | None = None, timeout: float = 1800) -> list[dict]:
        """Queue a graph, wait for it, and return the output image descriptors."""
        outputs = await self.run_outputs(graph, on_progress, timeout)
        images = [img for _nid, img in iter_images(outputs)]
        if not images:
            raise ComfyError("생성은 끝났지만 저장된 이미지가 없어요")
        return images

    async def run_outputs(self, graph: dict, on_progress: ProgressFn | None = None, timeout: float = 1800) -> dict:
        """Queue a graph, wait for it, and return ComfyUI's per-node outputs (images, text ...)."""
        ws = None
        if self.use_websocket:
            try:
                import websockets
                url = self.base.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
                ws = await websockets.connect(url, max_size=None, open_timeout=10)
            except Exception:
                ws = None  # fall back to polling
        try:
            pid = await self.queue(graph)
            await _call(on_progress, 0.0, "대기 중")
            if ws is not None:
                await self._wait_ws(ws, pid, on_progress, timeout)
            else:
                await self._wait_poll(pid, on_progress, timeout)
        finally:
            if ws is not None:
                await ws.close()
        h = await self.history(pid)
        if not h:
            raise ComfyError("결과를 찾을 수 없어요")
        status = h.get("status") or {}
        if status.get("status_str") == "error":
            raise ComfyError(_history_error(h))
        return h.get("outputs") or {}

    async def _wait_ws(self, ws, pid: str, on_progress: ProgressFn | None, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        nodes_total, nodes_done = None, 0
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ComfyError("시간 초과: 생성이 너무 오래 걸려요")
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(left, 5))
            except asyncio.TimeoutError:
                h = await self.history(pid)          # missed message? check directly
                if h and (h.get("status") or {}).get("completed") is not None:
                    return
                continue
            if isinstance(raw, bytes):
                continue                              # preview frames
            msg = json.loads(raw)
            t, d = msg.get("type"), msg.get("data") or {}
            if d.get("prompt_id") not in (None, pid):
                continue
            if t == "progress":
                v, m = d.get("value", 0), d.get("max", 1) or 1
                await _call(on_progress, min(0.99, v / m), f"그리는 중 {v}/{m}")
            elif t == "executing":
                if d.get("node") is None and d.get("prompt_id") == pid:
                    return
            elif t == "execution_cached":
                nodes_done += len(d.get("nodes") or [])
            elif t == "execution_error":
                raise ComfyError(f"{d.get('node_type', '')}: {d.get('exception_message', '실행 오류')}".strip(": "))
            elif t == "execution_interrupted":
                raise ComfyError("작업이 중단됐어요")
            elif t == "execution_success" and d.get("prompt_id") == pid:
                return

    async def _wait_poll(self, pid: str, on_progress: ProgressFn | None, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            h = await self.history(pid)
            if h and (h.get("outputs") or (h.get("status") or {}).get("completed") is not None):
                return
            await asyncio.sleep(0.5)
        raise ComfyError("시간 초과: 생성이 너무 오래 걸려요")


def iter_images(outputs: dict):
    """(node_id, image descriptor) for every saved output image, in node order."""
    for nid in sorted(outputs, key=lambda k: int(k) if str(k).isdigit() else 0):
        for key in ("images", "gifs", "videos"):
            for item in outputs[nid].get(key, []) or []:
                if isinstance(item, dict) and item.get("filename") and item.get("type", "output") == "output":
                    yield nid, item


def output_text(outputs: dict) -> str:
    for nid in outputs:
        t = outputs[nid].get("text")
        if t:
            return t[0] if isinstance(t, list) else str(t)
    return ""


def _history_error(h: dict) -> str:
    for m in (h.get("status") or {}).get("messages", []):
        if m and m[0] == "execution_error":
            d = m[1]
            return f"{d.get('node_type', '')}: {d.get('exception_message', '실행 오류')}"
    return "ComfyUI 실행 오류"


async def _call(fn: ProgressFn | None, p: float, text: str) -> None:
    if fn is None:
        return
    r = fn(p, text)
    if asyncio.iscoroutine(r):
        await r
