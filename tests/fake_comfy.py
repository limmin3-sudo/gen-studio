"""In-memory stand-in for the ComfyUI HTTP API, good enough to drive the job pipeline in tests."""
from __future__ import annotations

import io
import json
import uuid

import httpx
from PIL import Image

from genstudio.workflows import REQUIRED_NODES


class FakeComfy:
    def __init__(self, fail_with: str | None = None):
        self.prompts: dict[str, dict] = {}
        self.uploads: dict[str, bytes] = {}
        self.freed = 0
        self.fail_with = fail_with

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _render(self, graph: dict) -> bytes:
        w = h = 64
        for node in graph.values():
            if node["class_type"] in ("EmptyLatentImage", "EmptySD3LatentImage"):
                w, h = node["inputs"]["width"], node["inputs"]["height"]
            if node["class_type"] == "LoadImage":
                data = self.uploads.get(node["inputs"]["image"])
                if data:
                    with Image.open(io.BytesIO(data)) as im:
                        w, h = im.size
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (120, 90, 200)).save(buf, "PNG")
        return buf.getvalue()

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4080 SUPER"}]})
        if path == "/object_info":
            return httpx.Response(200, json={n: {} for n in REQUIRED_NODES})
        if path == "/free":
            self.freed += 1
            return httpx.Response(200)
        if path == "/interrupt":
            return httpx.Response(200)
        if path == "/upload/image":
            boundary = request.headers["content-type"].split("boundary=", 1)[1].encode()
            part = next(p for p in request.content.split(b"--" + boundary) if b'filename="' in p)
            name = part.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
            data = part.split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n")
            self.uploads[f"gen-studio/{name}"] = data
            return httpx.Response(200, json={"name": name, "subfolder": "gen-studio", "type": "input"})
        if path == "/prompt" and request.method == "POST":
            graph = json.loads(request.content)["prompt"]
            for nid, node in graph.items():
                for v in node["inputs"].values():
                    if isinstance(v, list) and len(v) == 2 and v[0] not in graph:
                        return httpx.Response(400, json={"error": {"message": f"bad link in node {nid}"}, "node_errors": {}})
            pid = uuid.uuid4().hex
            self.prompts[pid] = graph
            return httpx.Response(200, json={"prompt_id": pid, "number": len(self.prompts)})
        if path.startswith("/history/"):
            pid = path.rsplit("/", 1)[1]
            if pid not in self.prompts:
                return httpx.Response(200, json={})
            if self.fail_with:
                return httpx.Response(200, json={pid: {"outputs": {}, "status": {"status_str": "error", "completed": False,
                    "messages": [["execution_error", {"node_type": "KSampler", "exception_message": self.fail_with}]]}}})
            save = next(k for k, n in self.prompts[pid].items() if n["class_type"] == "SaveImage")
            return httpx.Response(200, json={pid: {
                "outputs": {save: {"images": [{"filename": f"{pid}.png", "subfolder": "gen-studio", "type": "output"}]}},
                "status": {"status_str": "success", "completed": True}}})
        if path == "/view":
            pid = request.url.params["filename"].removesuffix(".png")
            return httpx.Response(200, content=self._render(self.prompts[pid]), headers={"content-type": "image/png"})
        return httpx.Response(404)
