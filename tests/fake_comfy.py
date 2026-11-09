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
        self.views: list[dict] = []
        self.fail_with = fail_with

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    describe_reply = ('```json\n[{"label": "woman", "ko": "여성", "parent": null},'
                      ' {"label": "blonde hair", "ko": "금발 머리", "parent": "woman"},'
                      ' {"label": "white collared shirt", "ko": "흰 칼라 셔츠", "parent": "woman"},'
                      ' {"label": "epaulets", "ko": "견장", "parent": "white collared shirt"},'
                      ' {"label": "unicorn", "ko": "유니콘", "parent": null},'
                      ' {"label": "background", "ko": "배경", "parent": null}]\n```')

    @staticmethod
    def translate_reply(instruction: str) -> str:
        """Fake Qwen3-VL rewrite: tags for Illustrious, a sentence otherwise."""
        if "Danbooru" in instruction:
            return "1girl, solo, silver hair, white shirt, cafe, window light"
        return "Prompt: A woman sitting by a window in soft natural light."

    def _is_mask(self, graph: dict, save_id: str) -> bool:
        src = graph[save_id]["inputs"]["images"][0]
        return graph[src]["class_type"] == "MaskToImage"

    def _image_size(self, graph: dict) -> tuple[int, int]:
        for n in graph.values():
            if n["class_type"] == "LoadImage":
                data = self.uploads.get(n["inputs"]["image"])
                if data:
                    with Image.open(io.BytesIO(data)) as im:
                        return im.size
        return 256, 256

    def _masks(self, graph: dict, save_id: str) -> list[bytes]:
        """Fake SAM 3: boxes per label; 'epaulets' finds two, unknown labels find nothing."""
        w, h = self._image_size(graph)
        det = graph[graph[save_id]["inputs"]["images"][0]]["inputs"]["mask"][0]
        d = graph[det]["inputs"]
        boxes = []
        if graph[det]["class_type"] == "RemoveBackground":
            boxes = [(w * .25, h * .1, w * .75, h)]
        elif "positive_coords" in d:
            pts = json.loads(graph[d["positive_coords"][0]]["inputs"]["value"])
            boxes = [(p["x"] - 20, p["y"] - 20, p["x"] + 20, p["y"] + 20) for p in pts]
        else:
            label = graph[d["conditioning"][0]]["inputs"]["text"]
            table = {"woman": [(w * .3, h * .1, w * .7, h * .95)], "blonde hair": [(w * .4, h * .1, w * .6, h * .25)],
                     "white collared shirt": [(w * .35, h * .3, w * .65, h * .55)],
                     "epaulets": [(w * .36, h * .3, w * .44, h * .33), (w * .56, h * .3, w * .64, h * .33)],
                     "background": [(0, 0, w, h * .08)], "person": [(w * .3, h * .1, w * .7, h * .95)],
                     "face": [(w * .45, h * .12, w * .55, h * .2)],
                     "hand": [(w * .3, h * .5, w * .34, h * .54), (w * .66, h * .5, w * .7, h * .54), (0, 0, 2, 2)]}
            boxes = table.get(label, [])
        out = []
        for b in boxes or [None]:
            m = Image.new("RGB", (w, h), 0)
            if b:
                m.paste((255, 255, 255), tuple(int(v) for v in b))
            buf = io.BytesIO()
            m.save(buf, "PNG")
            out.append(buf.getvalue())
        return out

    def _render(self, graph: dict) -> bytes:
        w = h = 64
        for node in graph.values():
            if node["class_type"] in ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyHunyuanLatentVideo"):
                w, h = node["inputs"]["width"], node["inputs"]["height"]
            if node["class_type"] == "LoadImage":
                data = self.uploads.get(node["inputs"]["image"])
                if data:
                    with Image.open(io.BytesIO(data)) as im:
                        w, h = im.size
        ups = sum(1 for n in graph.values() if n["class_type"] == "ImageUpscaleWithModel")
        if ups:                                   # 1MP, then x2 per Real-ESRGAN step
            k = (1_000_000 / (w * h)) ** 0.5 * 2 ** ups
            w, h = round(w * k), round(h * k)
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
            sub = next((p.split(b"\r\n\r\n", 1)[1].strip().decode() for p in request.content.split(b"--" + boundary)
                        if b'name="subfolder"' in p), "gen-studio")
            self.uploads[f"{sub}/{name}"] = data
            return httpx.Response(200, json={"name": name, "subfolder": sub, "type": "input"})
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
            graph = self.prompts[pid]
            outputs = {}
            for nid, n in graph.items():
                if n["class_type"] == "SaveImage":
                    k = len(self._masks(graph, nid)) if self._is_mask(graph, nid) else 1
                    outputs[nid] = {"images": [{"filename": f"{pid}_{nid}_{j}.png", "subfolder": "gen-studio", "type": "output"}
                                               for j in range(k)]}
                if n["class_type"] == "PreviewAny":
                    gen = graph[n["inputs"]["source"][0]]["inputs"]
                    reply = (self.describe_reply if "images.image1" in gen and "image 1" not in gen["prompt"]
                             else self.translate_reply(gen["prompt"]))
                    outputs[nid] = {"text": [reply]}
            return httpx.Response(200, json={pid: {"outputs": outputs, "status": {"status_str": "success", "completed": True}}})
        if path == "/view" and request.url.params["filename"].endswith(".safetensors"):
            self.views.append(dict(request.url.params))
            return httpx.Response(200, content=b"trained-lora")
        if path == "/view":
            pid, nid, j = request.url.params["filename"].removesuffix(".png").rsplit("_", 2)
            graph = self.prompts[pid]
            data = self._masks(graph, nid)[int(j)] if self._is_mask(graph, nid) else self._render(graph)
            return httpx.Response(200, content=data, headers={"content-type": "image/png"})
        return httpx.Response(404)
