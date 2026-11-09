"""LoRA library: register, detect the family, download from Civitai, and describe how to apply.

Files live in ComfyUI/models/loras; our notes about them in outputs/loras.json.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import struct
import threading
import time
from pathlib import Path

import httpx

from .config import get_secret

# Civitai "baseModel" -> our family id
CIVITAI_FAMILY = [
    (re.compile(r"z\s*image", re.I), "zimage"),
    (re.compile(r"qwen", re.I), "qwen21"),
    (re.compile(r"wan", re.I), "wan22"),
    (re.compile(r"illustrious|noobai|sdxl|pony", re.I), "illustrious"),
]
FAMILY_NAME = {"zimage": "Z-Image (실사)", "illustrious": "Illustrious·SDXL (반실사)", "qwen21": "Qwen-Image 2.1",
               "wan22": "Wan 2.2"}
UA = {"User-Agent": "gen-studio/0.1"}


def family_from_base(base: str | None) -> str | None:
    for rx, fam in CIVITAI_FAMILY:
        if base and rx.search(base):
            return fam
    return None


def guess_family_from_file(path: Path) -> tuple[str | None, str]:
    """Look inside a .safetensors LoRA: training metadata first, then tensor names."""
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(min(n, 50_000_000)))
    except (OSError, ValueError, struct.error):
        return None, "파일을 읽을 수 없어요"
    meta = header.get("__metadata__") or {}
    hint = " ".join(str(meta.get(k, "")) for k in ("ss_base_model_version", "modelspec.architecture",
                                                     "ss_sd_model_name", "base_model"))
    fam = family_from_base(hint)
    if fam:
        return fam, f"학습 정보: {hint.strip()[:60]}"
    keys = " ".join(list(header)[:400])
    if "lora_unet_input_blocks" in keys or "lora_te1_" in keys or "lora_te2_" in keys:
        return "illustrious", "SDXL 형식"
    if re.search(r"blocks\.\d+\.(self_attn|cross_attn|ffn)", keys):
        return "wan22", "Wan 형식"
    if "img_mod" in keys or "txt_mod" in keys or "transformer_blocks" in keys:
        return "qwen21", "Qwen 형식"
    if re.search(r"layers\.\d+\.(attention|feed_forward)", keys):
        return "zimage", "Z-Image 형식"
    return None, "계열을 알아내지 못했어요"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def parse_civitai_url(url: str) -> tuple[int | None, int | None]:
    m = re.search(r"civitai\.com/models/(\d+)", url)
    v = re.search(r"modelVersionId=(\d+)", url) or re.search(r"/api/download/models/(\d+)", url)
    return (int(m.group(1)) if m else None), (int(v.group(1)) if v else None)


class Loras:
    def __init__(self, path: Path, loras_dir: Path | None, transport: httpx.AsyncBaseTransport | None = None):
        self.path = path
        self.dir = loras_dir
        self._transport = transport
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self.downloads: dict[str, dict] = {}
        try:
            for x in json.loads(path.read_text(encoding="utf-8")):
                self._items[x["id"]] = x
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            pass

    # ------------------------------------------------------------ storage
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(list(self._items.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    def list(self) -> list[dict]:
        out = []
        for x in sorted(self._items.values(), key=lambda x: x["name"].lower()):
            present = (self.dir / x["file"]).exists() if self.dir else None
            out.append({**x, "present": present, "family_name": FAMILY_NAME.get(x.get("family") or "", "알 수 없음")})
        return out

    def get(self, key: str) -> dict | None:
        if key in self._items:
            return self._items[key]
        low = re.sub(r"[\s_\-.]+", "", key.lower())
        hits = [x for x in self._items.values()
                if low in (re.sub(r"[\s_\-.]+", "", x["name"].lower()), re.sub(r"[\s_\-.]+", "", x["file"].lower()))
                or low in re.sub(r"[\s_\-.]+", "", x["name"].lower())]
        return hits[0] if len(hits) == 1 else None

    def upsert(self, file: str, **fields) -> dict:
        with self._lock:
            lid = next((k for k, v in self._items.items() if v["file"] == file), None) or _slug(Path(file).stem)
            while lid in self._items and self._items[lid]["file"] != file:
                lid += "-2"
            prev = self._items.get(lid, {})
            item = {"id": lid, "name": prev.get("name") or Path(file).stem, "file": file, "family": None,
                    "triggers": [], "strength": 0.8, "stage": "both", "civitai": None, "preview": None, "note": "",
                    "added_at": prev.get("added_at", time.time())}
            item.update({k: v for k, v in prev.items() if k in item})
            item.update({k: v for k, v in fields.items() if v is not None})
            self._items[lid] = item
            self._save()
            return item

    def update(self, key: str, **fields) -> dict:
        x = self.get(key)
        if not x:
            raise KeyError(f"LoRA를 찾을 수 없어요: {key}")
        allowed = {k: v for k, v in fields.items() if k in ("name", "family", "triggers", "strength", "stage", "note")}
        return self.upsert(x["file"], **allowed)

    def remove(self, key: str, delete_file: bool = False) -> bool:
        x = self.get(key)
        if not x:
            return False
        with self._lock:
            self._items.pop(x["id"], None)
            self._save()
        if delete_file and self.dir:
            (self.dir / x["file"]).unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------ detection
    async def _civitai_json(self, url: str) -> dict | None:
        try:
            async with httpx.AsyncClient(headers=UA, timeout=20, transport=self._transport) as c:
                r = await c.get(url)
                return r.json() if r.status_code == 200 else None
        except httpx.HTTPError:
            return None

    def _from_version(self, v: dict) -> dict:
        imgs = [i for i in v.get("images") or [] if i.get("type", "image") == "image"]
        return {"family": family_from_base(v.get("baseModel")), "triggers": [t.strip(" ,") for t in v.get("trainedWords") or []][:6],
                "civitai": {"model_id": v.get("modelId"), "version_id": v.get("id"), "base_model": v.get("baseModel"),
                            "url": f"https://civitai.com/models/{v.get('modelId')}?modelVersionId={v.get('id')}"},
                "preview": imgs[0]["url"] if imgs else None,
                "name": (v.get("model") or {}).get("name")}

    async def identify(self, file: str) -> dict:
        """Civitai by file hash; falls back to looking inside the file."""
        path = self.dir / file
        info: dict = {}
        sha = await asyncio.to_thread(sha256_file, path)
        v = await self._civitai_json(f"https://civitai.com/api/v1/model-versions/by-hash/{sha}")
        if v and v.get("id"):
            info = self._from_version(v)
            info["note"] = "Civitai에서 확인"
        if not info.get("family"):
            fam, why = await asyncio.to_thread(guess_family_from_file, path)
            info["family"] = fam
            info["note"] = (info.get("note", "") + " · " if info.get("note") else "") + why
        return self.upsert(file, sha256=sha, **info)

    async def scan(self) -> list[dict]:
        """Register LoRA files that are in the folder but not in the list yet."""
        if not self.dir or not self.dir.exists():
            return []
        known = {x["file"] for x in self._items.values()}
        added = []
        for p in sorted(self.dir.rglob("*.safetensors")):
            rel = str(p.relative_to(self.dir)).replace("\\", "/")
            if rel not in known:
                added.append(await self.identify(rel))
        return added

    # ------------------------------------------------------------ download
    async def add_from_civitai(self, url: str) -> str:
        """Starts a background download; returns a download id to poll."""
        if not self.dir:
            raise ValueError("ComfyUI 폴더가 설정되지 않았어요")
        mid, vid = parse_civitai_url(url)
        if not (mid or vid):
            raise ValueError("Civitai 모델 주소를 붙여넣어 주세요 (civitai.com/models/...)")
        if vid:
            v = await self._civitai_json(f"https://civitai.com/api/v1/model-versions/{vid}")
        else:
            m = await self._civitai_json(f"https://civitai.com/api/v1/models/{mid}")
            v = (m or {}).get("modelVersions", [None])[0]
            if v is not None:
                v = {**v, "modelId": mid, "model": {"name": (m or {}).get("name"), "type": (m or {}).get("type")}}
        if not v:
            raise ValueError("Civitai에서 정보를 가져오지 못했어요")
        mtype = ((v.get("model") or {}).get("type") or "LORA").upper()
        if mtype not in ("LORA", "LOCON", "DORA", "LYCORIS"):
            raise ValueError(f"LoRA가 아니에요 ({mtype})")
        files = v.get("files") or []
        f = next((x for x in files if x.get("primary")), files[0] if files else None)
        if not f:
            raise ValueError("받을 파일이 없어요")
        did = f"d{int(time.time() * 1000)}"
        self.downloads[did] = {"id": did, "status": "running", "file": f["name"], "done": 0,
                               "total": int(f.get("sizeKB", 0) * 1024), "error": None, "lora_id": None}
        asyncio.create_task(self._download(did, f.get("downloadUrl") or f"https://civitai.com/api/download/models/{v['id']}",
                                           f["name"], self._from_version(v)))
        return did

    async def _download(self, did: str, url: str, name: str, info: dict) -> None:
        st = self.downloads[did]
        dest = self.dir / name
        part = dest.with_name(dest.name + ".part")
        headers = dict(UA)
        key = get_secret("civitai_api_key", "CIVITAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(60, read=120), follow_redirects=True,
                                         transport=self._transport) as c:
                async with c.stream("GET", url) as r:
                    if r.status_code in (401, 403):
                        raise RuntimeError("권한이 없어요. Civitai API 키가 필요하거나, 성인 콘텐츠 보기를 켜야 해요.")
                    r.raise_for_status()
                    st["total"] = int(r.headers.get("content-length", 0)) or st["total"]
                    with open(part, "wb") as out:
                        async for chunk in r.aiter_bytes(1024 * 1024):
                            out.write(chunk)
                            st["done"] += len(chunk)
            part.replace(dest)
            item = self.upsert(name, note="Civitai에서 받음", **{k: v for k, v in info.items() if v})
            st.update(status="done", lora_id=item["id"])
        except Exception as e:
            part.unlink(missing_ok=True)
            st.update(status="error", error=str(e))

    async def add_file(self, filename: str, data: bytes) -> dict:
        if not self.dir:
            raise ValueError("ComfyUI 폴더가 설정되지 않았어요")
        name = Path(filename).name
        if not name.endswith(".safetensors"):
            raise ValueError(".safetensors 파일만 넣을 수 있어요")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / name).write_bytes(data)
        return await self.identify(name)


def _slug(s: str) -> str:
    return re.sub(r"[^\w가-힣-]+", "-", s.strip().lower()).strip("-") or f"lora-{int(time.time())}"


def apply_triggers(prompt: str, loras: list[dict]) -> str:
    """Put each LoRA's trigger words in front of the prompt once."""
    words = []
    low = prompt.lower()
    for x in loras:
        for t in x.get("triggers") or []:
            if t and t.lower() not in low and t not in words:
                words.append(t)
    return (", ".join(words) + ", " + prompt) if words else prompt
