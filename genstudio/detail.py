"""Detail pass (face / eyes / hands): find each one with SAM 3, redraw it enlarged, paste it back softly.

Small faces in a full-body picture get only a few dozen pixels, so the model draws them mushy. Cropping around
each face, scaling the crop up to ~1024px, redrawing it lightly (denoise ~0.4) and shrinking it back gives the
face the same detail as a close-up while the rest of the picture stays untouched.
"""
from __future__ import annotations

import io
from typing import Literal

from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

# target -> (SAM 3 word, padding around the box as a share of its size, prompt words, label)
TARGETS = {
    "face": ("face", 0.45, "detailed face, natural skin texture, detailed eyes, sharp focus", "얼굴"),
    "eyes": ("eye", 1.2, "detailed eyes, clear iris, detailed eyelashes, sharp focus", "눈"),
    "hands": ("hand", 0.4, "detailed hand, five fingers, natural hand, sharp focus", "손"),
}
TAG_WORDS = {
    "face": "detailed face, beautiful detailed eyes, sharp focus",
    "eyes": "beautiful detailed eyes, detailed pupils, sharp focus",
    "hands": "detailed hands, five fingers, sharp focus",
}
WORK_SIDE = 1024          # crops are redrawn at about this size
MAX_SIDE = 1536
MIN_AREA = 0.0004         # ignore specks smaller than this share of the picture
MAX_PER_TARGET = 6


class DetailRequest(BaseModel):
    image_id: str
    targets: list[Literal["face", "eyes", "hands"]] = Field(default_factory=lambda: ["face"])
    strength: float = Field(0.4, ge=0.15, le=0.8, description="다시 그리는 정도 (낮을수록 원래 얼굴 유지)")
    prompt: str = Field("", description="덧붙일 말 (예: 'smiling', '푸른 눈')")
    model: str = "auto"


def load_mask(data: bytes, size: tuple[int, int]) -> Image.Image:
    with Image.open(io.BytesIO(data)) as m:
        m = m.convert("L")
        return m if m.size == size else m.resize(size, Image.NEAREST)


def crop_box(mask: Image.Image, pad: float) -> tuple[int, int, int, int] | None:
    """Box around the white area, padded, clamped to the picture. None for empty / tiny masks."""
    bb = mask.point(lambda v: 255 if v > 127 else 0).getbbox()
    if not bb:
        return None
    W, H = mask.size
    x0, y0, x1, y1 = bb
    if (x1 - x0) * (y1 - y0) < MIN_AREA * W * H:
        return None
    side = max(x1 - x0, y1 - y0)
    p = int(side * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side / 2 + p                      # square crops redraw best
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    return x0, y0, x1, y1


def work_size(w: int, h: int) -> tuple[int, int]:
    s = WORK_SIDE / max(w, h)
    s = max(s, 1.0) if max(w, h) <= MAX_SIDE else MAX_SIDE / max(w, h)
    return max(64, round(w * s / 16) * 16), max(64, round(h * s / 16) * 16)


def png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def paste_back(base: Image.Image, redrawn: bytes, box: tuple[int, int, int, int], mask_crop: Image.Image) -> Image.Image:
    """Shrink the redrawn crop back and blend it in only where the part is, with a soft edge."""
    x0, y0, x1, y1 = box
    with Image.open(io.BytesIO(redrawn)) as r:
        r = r.convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS)
    soft = mask_crop.filter(ImageFilter.MaxFilter(9)).filter(
        ImageFilter.GaussianBlur(max(2, (x1 - x0) // 40)))
    base.paste(r, (x0, y0), soft)
    return base


def prompt_for(target: str, family: str, extra: str) -> str:
    words = TAG_WORDS[target] if family == "illustrious" else TARGETS[target][2]
    return ", ".join(x for x in (extra.strip(), words) if x)
