"""Image library: every generated, edited or imported image plus its metadata.

Layout on disk:
    outputs/2026-11-09/<id>.png     the image (ComfyUI metadata kept inside)
    outputs/2026-11-09/<id>.json    our metadata (prompt, model, parent, ...)
    outputs/.thumbs/<id>.jpg        small preview for the gallery
"""
from __future__ import annotations

import io
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, PngImagePlugin

MAX_IMPORT_PIXELS = 2_100_000   # ~1448x1448; keeps edits within 16GB VRAM


def new_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


class Library:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".thumbs").mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self._scan()

    def _scan(self) -> None:
        for meta in self.root.glob("*/*.json"):
            if meta.name.endswith(".segments.json"):      # segment lists live next to images; not image records
                continue
            try:
                rec = json.loads(meta.read_text(encoding="utf-8"))
                if not isinstance(rec, dict) or "id" not in rec or "file" not in rec:
                    continue
                if (self.root / rec["file"]).exists():
                    self._items[rec["id"]] = rec
            except (OSError, ValueError, KeyError, TypeError):
                continue

    # ------------------------------------------------------------ write
    def add(self, png: bytes, meta: dict) -> dict:
        iid = meta.get("id") or new_id()
        day = datetime.now().strftime("%Y-%m-%d")
        folder = self.root / day
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{iid}.png").write_bytes(png)
        with Image.open(io.BytesIO(png)) as im:
            w, h = im.size
            self._thumb(im, iid)
        parent = self._items.get(meta.get("parent_id") or "")
        rec = {
            "id": iid,
            "file": f"{day}/{iid}.png",
            "created_at": time.time(),
            "width": w,
            "height": h,
            "root_id": parent["root_id"] if parent else iid,
            **{k: v for k, v in meta.items() if k not in ("id",)},
        }
        (folder / f"{iid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        with self._lock:
            self._items[iid] = rec
        return rec

    def add_video(self, video: bytes, poster_png: bytes, meta: dict, ext: str = ".mp4") -> dict:
        """A generated clip: the file plus a poster frame for the gallery thumbnail."""
        iid = meta.get("id") or new_id()
        day = datetime.now().strftime("%Y-%m-%d")
        folder = self.root / day
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{iid}{ext}").write_bytes(video)
        with Image.open(io.BytesIO(poster_png)) as im:
            w, h = im.size
            self._thumb(im, iid)
        parent = self._items.get(meta.get("parent_id") or "")
        rec = {"id": iid, "file": f"{day}/{iid}{ext}", "created_at": time.time(), "width": w, "height": h,
               "media": "video", "root_id": parent["root_id"] if parent else iid,
               **{k: v for k, v in meta.items() if k != "id"}}
        (folder / f"{iid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        with self._lock:
            self._items[iid] = rec
        return rec

    def import_image(self, data: bytes, filename: str = "image.png") -> dict:
        """Bring an outside image in: fix orientation, keep size workable, remember any AI metadata."""
        with Image.open(io.BytesIO(data)) as im:
            embedded = _embedded_metadata(im)
            try:
                from PIL import ImageOps
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
            w, h = im.size
            scale = min(1.0, (MAX_IMPORT_PIXELS / (w * h)) ** 0.5)
            nw, nh = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
            if (nw, nh) != (w, h):
                im = im.resize((nw, nh), Image.LANCZOS)
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                im = bg
            buf = io.BytesIO()
            im.save(buf, "PNG")
        return self.add(buf.getvalue(), {
            "kind": "import",
            "source": "import",
            "label": "불러온 원본",
            "original_name": filename,
            "original_size": [w, h],
            "embedded": embedded,
            "prompt": embedded.get("prompt_text", ""),
        })

    def _thumb(self, im: Image.Image, iid: str) -> None:
        t = flat_rgb(im)
        t.thumbnail((320, 320))
        t.save(self.root / ".thumbs" / f"{iid}.jpg", "JPEG", quality=82)

    def update(self, iid: str, **fields) -> dict:
        rec = self._items[iid]
        rec.update(fields)
        (self.root / rec["file"]).with_suffix(".json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        return rec

    def save_side_file(self, iid: str, suffix: str, data: bytes) -> str:
        """Stores an extra file next to the image (e.g. the edit mask) and returns its library-relative path."""
        p = (self.root / self._items[iid]["file"]).with_suffix(suffix)
        p.write_bytes(data)
        return str(p.relative_to(self.root)).replace("\\", "/")

    def export(self, iid: str, dest: Path, fmt: str | None = None, max_side: int | None = None,
               width: int | None = None, height: int | None = None, quality: int = 90, overwrite: bool = False) -> dict:
        """Write a copy somewhere else (e.g. a project's assets folder), optionally resized and converted."""
        src = self.path(iid)
        if not src:
            raise KeyError("이미지를 찾을 수 없어요")
        dest = Path(dest).expanduser()
        fmt = (fmt or dest.suffix.lstrip(".") or "png").lower().replace("jpeg", "jpg")
        if fmt not in ("png", "webp", "jpg"):
            raise ValueError("형식은 png, webp, jpg 중 하나예요")
        if not dest.suffix:
            dest = dest.with_suffix("." + fmt)
        if dest.exists() and not overwrite:
            raise ValueError(f"이미 파일이 있어요: {dest} (덮어쓰려면 overwrite)")
        with Image.open(src) as im:
            alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
            if alpha and fmt in ("png", "webp"):
                im = im.convert("RGBA")                 # keep the cut-out transparent
            elif alpha:
                rgba = im.convert("RGBA")               # jpg has no transparency: put it on white
                im = Image.new("RGB", rgba.size, (255, 255, 255))
                im.paste(rgba, mask=rgba.split()[3])
            else:
                im = im.convert("RGB")
            if width and height:
                im = im.resize((int(width), int(height)), Image.LANCZOS)
            elif width or height:
                r = im.width / im.height
                im = im.resize((int(width or round(height * r)), int(height or round(width / r))), Image.LANCZOS)
            elif max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            kw = {"webp": {"quality": quality, "method": 6}, "jpg": {"quality": quality, "optimize": True}, "png": {}}[fmt]
            im.save(dest, {"jpg": "JPEG", "webp": "WEBP", "png": "PNG"}[fmt], **kw)
            size = im.size
        return {"path": str(dest), "format": fmt, "width": size[0], "height": size[1], "bytes": dest.stat().st_size}

    def delete_versions(self, iid: str) -> int:
        """Delete an image together with all its versions."""
        ids = [r["id"] for r in self.versions(iid)]
        return sum(1 for x in ids if self.delete(x))

    def delete(self, iid: str) -> bool:
        rec = self._items.pop(iid, None)
        if not rec:
            return False
        if rec["root_id"] == iid:
            # the first version goes away: the earliest remaining one becomes the new first, so the set stays together
            rest = sorted((r for r in self._items.values() if r["root_id"] == iid), key=lambda r: r["created_at"])
            if rest:
                new_root = rest[0]["id"]
                for r in rest:
                    self.update(r["id"], root_id=new_root, **({"parent_id": None} if r["id"] == new_root else {}))
        p = self.root / rec["file"]
        for f in (p, p.with_suffix(".json"), p.with_suffix(".mask.png"), p.with_suffix(".segments.json"),
                  self.root / ".thumbs" / f"{iid}.jpg"):
            f.unlink(missing_ok=True)
        seg_dir = p.with_suffix(".segments")
        if seg_dir.is_dir():
            for f in seg_dir.glob("*.png"):
                f.unlink()
            seg_dir.rmdir()
        return True

    def usage(self) -> dict:
        """How many pictures/videos are kept and how much disk they take (thumbnails included)."""
        n = len(self._items)
        size = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file()
                   and f.name not in ("characters.json", "loras.json", "chips.json"))
        return {"count": n, "bytes": size}

    def clear_all(self) -> dict:
        """Delete every picture, video, version, mask, segment file and thumbnail in the library."""
        before = self.usage()
        for iid in list(self._items):
            self.delete(iid)
        media = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".json")
        for f in sorted(self.root.rglob("*"), key=lambda x: -len(x.parts)):     # deepest first
            if f.is_dir():
                if not any(f.iterdir()):
                    f.rmdir()
            elif f.parent != self.root or (f.suffix.lower() in media and f.name not in (
                    "characters.json", "loras.json")):
                f.unlink(missing_ok=True)              # day folders, thumbnails, stray pictures at the top
        for d in sorted((d for d in self.root.iterdir() if d.is_dir()), key=lambda x: -len(x.parts)):
            if not any(d.rglob("*")):
                d.rmdir()
        (self.root / ".thumbs").mkdir(exist_ok=True)
        return before

    # ------------------------------------------------------------- read
    def get(self, iid: str) -> dict | None:
        return self._items.get(iid)

    def path(self, iid: str) -> Path | None:
        rec = self._items.get(iid)
        return self.root / rec["file"] if rec else None

    def thumb_path(self, iid: str) -> Path | None:
        p = self.root / ".thumbs" / f"{iid}.jpg"
        return p if p.exists() else self.path(iid)

    def list(self, limit: int = 60, offset: int = 0, roots_only: bool = False) -> list[dict]:
        items = sorted(self._items.values(), key=lambda r: r["created_at"], reverse=True)
        if roots_only:
            items = [r for r in items if r["root_id"] == r["id"]]
        return items[offset: offset + limit]

    def versions(self, iid: str) -> list[dict]:
        rec = self._items.get(iid)
        if not rec:
            return []
        root = rec["root_id"]
        return sorted((r for r in self._items.values() if r["root_id"] == root), key=lambda r: r["created_at"])


def _embedded_metadata(im: Image.Image) -> dict:
    """Reads generation info other tools leave in PNG text chunks (ComfyUI, A1111/Forge, NovelAI)."""
    info = getattr(im, "info", {}) or {}
    out: dict = {}
    if "prompt" in info or "workflow" in info:
        out["tool"] = "ComfyUI"
        try:
            graph = json.loads(info.get("prompt", "{}"))
            texts = [n["inputs"]["text"] for n in graph.values()
                     if isinstance(n, dict) and n.get("class_type", "").startswith("CLIPTextEncode")
                     and isinstance(n.get("inputs", {}).get("text"), str)]
            if texts:
                out["prompt_text"] = texts[0]
        except (ValueError, AttributeError, KeyError):
            pass
    elif "parameters" in info:
        out["tool"] = "A1111/Forge"
        out["prompt_text"] = str(info["parameters"]).split("\nNegative prompt:")[0].strip()
    elif info.get("Software", "").lower().startswith("novelai") or "Comment" in info:
        out["tool"] = info.get("Software", "NovelAI")
        out["prompt_text"] = str(info.get("Description", ""))
    return out


def flat_rgb(im: Image.Image, backdrop: tuple[int, int, int] = (238, 238, 238)) -> Image.Image:
    """RGB copy for previews; transparent cut-outs sit on a light gray instead of showing the removed background."""
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        out = Image.new("RGB", rgba.size, backdrop)
        out.paste(rgba, mask=rgba.split()[3])
        return out
    return im.convert("RGB")


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    pi = PngImagePlugin.PngInfo()
    im.save(buf, "PNG", pnginfo=pi)
    return buf.getvalue()
