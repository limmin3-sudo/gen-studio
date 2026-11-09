"""Segments: named masks of the parts of an image (person, hair, shirt ...).

Stored next to the image:
    outputs/2026-11-09/<id>.segments/<n>.png    one black/white mask per segment
    outputs/2026-11-09/<id>.segments.json       labels, groups, boxes
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

DESCRIBE_PROMPT = """You help a photo editor split a picture into parts a user may want to select and edit.
List the distinct visible things in the image as a JSON array. Each item:
{"label": "<1-4 word English noun phrase that names it precisely, e.g. 'blonde hair', 'white collared shirt'>",
 "ko": "<the same in natural Korean>",
 "parent": "<label of the thing it belongs to, or null>"}
Include every person (as "woman", "man", "girl", "boy" or "person"), then their hair, face, each clothing item
(top, bottom, legwear such as stockings, tights or socks, and shoes), visible body parts people often edit (legs,
arms), accessories and anything they hold, then notable objects, furniture, vehicles, and finally "background".
Use one item per kind of thing even if there are several (e.g. one "epaulets" item for both). Never split left
and right: use "arms", "legs", "hands", "feet", "shoes", "eyes" once each.
At most 16 items. Reply with the JSON array only."""

# used when no vision-language model is installed
FALLBACK_LABELS = [
    ("person", "사람", None), ("hair", "머리카락", "person"), ("face", "얼굴", "person"), ("shirt", "셔츠", "person"),
    ("jacket", "재킷", "person"), ("coat", "코트", "person"), ("dress", "원피스", "person"), ("skirt", "치마", "person"),
    ("pants", "바지", "person"), ("shoes", "신발", "person"), ("hat", "모자", "person"), ("glasses", "안경", "person"),
    ("bag", "가방", "person"), ("necklace", "목걸이", "person"), ("belt", "벨트", "person"), ("chair", "의자", None),
    ("table", "테이블", None), ("plant", "식물", None), ("car", "자동차", None), ("building", "건물", None),
    ("sky", "하늘", None), ("background", "배경", None),
]

# Korean words people type into the "find" box -> what SAM 3 understands
KO_LABELS = {
    "사람": "person", "여자": "woman", "여성": "woman", "남자": "man", "남성": "man", "아이": "child", "머리": "hair",
    "머리카락": "hair", "얼굴": "face", "눈": "eyes", "입술": "lips", "목": "neck", "팔": "arms", "손": "hands",
    "다리": "legs", "허벅지": "thighs", "발": "feet", "가슴": "chest", "배": "belly", "허리": "waist", "엉덩이": "hips",
    "셔츠": "shirt", "블라우스": "blouse", "티셔츠": "t-shirt", "재킷": "jacket", "자켓": "jacket", "코트": "coat",
    "후드": "hoodie", "니트": "sweater", "스웨터": "sweater", "원피스": "dress", "드레스": "dress", "치마": "skirt",
    "스커트": "skirt", "바지": "pants", "청바지": "jeans", "반바지": "shorts", "레깅스": "leggings", "스타킹": "stockings",
    "타이츠": "tights", "양말": "socks", "신발": "shoes", "구두": "high heels", "힐": "high heels", "부츠": "boots",
    "운동화": "sneakers", "속옷": "underwear", "브라": "bra", "수영복": "swimsuit", "넥타이": "tie", "벨트": "belt",
    "모자": "hat", "안경": "glasses", "선글라스": "sunglasses", "목걸이": "necklace", "귀걸이": "earrings",
    "팔찌": "bracelet", "시계": "watch", "반지": "ring", "장갑": "gloves", "가방": "bag", "견장": "epaulets",
    "명찰": "name tag", "배지": "badge", "배경": "background", "하늘": "sky", "벽": "wall", "바닥": "floor",
    "창문": "window", "의자": "chair", "테이블": "table", "책상": "desk", "소파": "sofa", "침대": "bed", "식물": "plant",
    "꽃": "flowers", "자동차": "car", "건물": "building", "나무": "tree", "물": "water", "컵": "cup", "휴대폰": "phone",
}


def korean_to_label(word: str) -> str | None:
    w = word.strip()
    if w in KO_LABELS:
        return KO_LABELS[w]
    for ko, en in sorted(KO_LABELS.items(), key=lambda kv: -len(kv[0])):
        if ko in w:
            rest = w.replace(ko, "").strip()
            return en if not rest else None
    return None


MIN_AREA = 0.0008      # ignore specks smaller than this share of the image
DUP_IOU = 0.92


_SIDE = re.compile(r"^(left|right)\s+", re.I)
_PLURAL = {"foot": "feet", "arm": "arms", "leg": "legs", "hand": "hands", "shoe": "shoes", "eye": "eyes", "ear": "ears",
           "sleeve": "sleeves", "glove": "gloves", "boot": "boots", "sock": "socks", "earring": "earrings",
           "shoulder": "shoulders", "thigh": "thighs", "knee": "knees", "epaulet": "epaulets"}


def merge_sides(label: str, ko: str) -> tuple[str, str]:
    """'left arm' + '왼팔' -> 'arms' + '팔': both sides become one group that SAM 3 splits into instances."""
    if not _SIDE.match(label):
        return label, ko
    rest = _SIDE.sub("", label)
    words = rest.split()
    words[-1] = _PLURAL.get(words[-1], words[-1] if words[-1].endswith("s") else words[-1] + "s")
    ko = re.sub(r"^(왼쪽|오른쪽|왼|오른)\s*", "", ko or "").strip()
    return " ".join(words), ko


def parse_labels(text: str, limit: int = 16) -> list[dict]:
    """Pull the JSON array out of a model reply (tolerates code fences and chatter)."""
    m = re.search(r"\[.*\]", text or "", re.S)
    items: list = []
    if m:
        try:
            items = json.loads(m.group(0))
        except ValueError:
            items = []
    out, seen = [], set()
    for it in items:
        if isinstance(it, str):
            it = {"label": it}
        if not isinstance(it, dict):
            continue
        label = re.sub(r"[^a-zA-Z0-9 '\-]", "", str(it.get("label", ""))).strip().lower()
        label = " ".join(label.split()[:4])
        label, ko = merge_sides(label, str(it.get("ko") or "").strip()[:30])
        if not label or label in seen:
            continue
        seen.add(label)
        parent = it.get("parent")
        parent = str(parent).strip().lower() if parent not in (None, "", "null", "none") else None
        out.append({"label": label, "ko": ko, "parent": parent})
        if len(out) >= limit:
            break
    return out


def _mask_l(data: bytes, size: tuple[int, int] | None = None) -> Image.Image:
    with Image.open(io.BytesIO(data)) as im:
        m = im.convert("L")
        if size and m.size != size:
            m = m.resize(size, Image.NEAREST)
        return m.point(lambda v: 255 if v > 127 else 0)


def _iou(a: Image.Image, b: Image.Image) -> float:
    inter = ImageChops.multiply(a, b).histogram()[255]
    union = ImageChops.lighter(a, b).histogram()[255]
    return inter / union if union else 0.0


class SegmentStore:
    def __init__(self, library):
        self.library = library

    def _paths(self, iid: str) -> tuple[Path, Path]:
        img = self.library.path(iid)
        if not img:
            raise KeyError("이미지를 찾을 수 없어요")
        return img.with_suffix(".segments"), img.with_suffix(".segments.json")

    def load(self, iid: str) -> list[dict]:
        _, js = self._paths(iid)
        try:
            return json.loads(js.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return []

    def _write(self, iid: str, segs: list[dict]) -> None:
        _, js = self._paths(iid)
        js.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_detections(self, iid: str, detections: list[tuple[dict, list[bytes]]], source: str) -> list[dict]:
        """detections: [(label info, [mask png, ...]), ...] in label order. Replaces earlier automatic segments."""
        rec = self.library.get(iid)
        size = (rec["width"], rec["height"])
        folder, _ = self._paths(iid)
        folder.mkdir(exist_ok=True)
        keep = [s for s in self.load(iid) if s.get("source") == "click"]      # clicked parts survive a re-scan
        kept_masks = [_mask_l((folder / s["file"]).read_bytes(), size) for s in keep]
        segs: list[dict] = list(keep)
        n = max([int(s["id"].split("-")[-1]) for s in keep] + [0])
        total = size[0] * size[1]
        for info, masks in detections:
            found = []
            for data in masks:
                m = _mask_l(data, size)
                box = m.getbbox()
                if not box or m.histogram()[255] / total < MIN_AREA:
                    continue
                if any(_iou(m, k) > DUP_IOU for k in kept_masks):
                    continue
                kept_masks.append(m)
                found.append(m)
            for k, m in enumerate(found):
                n += 1
                sid = f"s-{n}"
                m.save(folder / f"{n}.png")
                segs.append({
                    "id": sid, "label": info["label"], "ko": info.get("ko") or "", "parent": info.get("parent"),
                    "group": info["label"] if len(found) > 1 else None, "instance": k + 1 if len(found) > 1 else None,
                    "bbox": list(m.getbbox()), "area": round(m.histogram()[255] / total, 4),
                    "file": f"{n}.png", "source": source,
                })
        self._write(iid, segs)
        return segs

    def add_click(self, iid: str, mask_png: bytes, label: str = "selection", ko: str = "선택한 부분") -> dict | None:
        rec = self.library.get(iid)
        size = (rec["width"], rec["height"])
        m = _mask_l(mask_png, size)
        if not m.getbbox():
            return None
        folder, _ = self._paths(iid)
        folder.mkdir(exist_ok=True)
        segs = self.load(iid)
        n = max([int(s["id"].split("-")[-1]) for s in segs] + [0]) + 1
        m.save(folder / f"{n}.png")
        seg = {"id": f"s-{n}", "label": label, "ko": ko, "parent": None, "group": None, "instance": None,
               "bbox": list(m.getbbox()), "area": round(m.histogram()[255] / (size[0] * size[1]), 4),
               "file": f"{n}.png", "source": "click"}
        segs.append(seg)
        self._write(iid, segs)
        return seg

    def resolve(self, iid: str, keys: list[str]) -> list[dict]:
        """Accepts segment ids, labels, Korean names or group names; a group or label selects all its instances."""
        segs = self.load(iid)
        chosen: list[dict] = []
        for key in keys:
            key = str(key)
            k = key.strip().lower()
            if k.isdigit() and 1 <= int(k) <= len(segs):        # number shown on the overview image
                hits = [segs[int(k) - 1]]
                chosen += [h for h in hits if h not in chosen]
                continue
            hits = [s for s in segs if s["id"] == key] or \
                   [s for s in segs if k in (s["label"], (s.get("ko") or "").lower(), s.get("group") or "")]
            if not hits:
                hits = [s for s in segs if k and (k in s["label"] or (s.get("ko") and k in s["ko"].lower()))]
            if not hits:
                raise KeyError(f"세그먼트를 찾을 수 없어요: {key}")
            chosen += [h for h in hits if h not in chosen]
        return chosen

    def union_mask(self, iid: str, keys: list[str]) -> bytes:
        rec = self.library.get(iid)
        folder, _ = self._paths(iid)
        acc = Image.new("L", (rec["width"], rec["height"]), 0)
        for s in self.resolve(iid, keys):
            acc = ImageChops.lighter(acc, _mask_l((folder / s["file"]).read_bytes(), acc.size))
        buf = io.BytesIO()
        Image.merge("RGB", (acc, acc, acc)).save(buf, "PNG")
        return buf.getvalue()

    def alpha_mask(self, iid: str, sid: str) -> bytes:
        """White where selected, transparent elsewhere: for CSS mask-image highlighting."""
        folder, _ = self._paths(iid)
        s = self.resolve(iid, [sid])[0]
        m = Image.open(folder / s["file"]).convert("L")
        im = Image.new("RGBA", m.size, (255, 255, 255, 0))
        im.putalpha(m)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def cutout(self, iid: str, key: str, thumb: int | None = None) -> bytes:
        """The selected part on a transparent background, cropped to its box."""
        folder, _ = self._paths(iid)
        segs = self.resolve(iid, [key])
        acc = None
        for s in segs:
            m = Image.open(folder / s["file"]).convert("L")
            acc = m if acc is None else ImageChops.lighter(acc, m)
        with Image.open(self.library.path(iid)) as im:
            rgba = im.convert("RGBA")
        rgba.putalpha(acc)
        rgba = rgba.crop(acc.getbbox())
        if thumb:
            rgba.thumbnail((thumb, thumb))
        buf = io.BytesIO()
        rgba.save(buf, "PNG")
        return buf.getvalue()

    def overview(self, iid: str, max_side: int = 1024) -> bytes:
        """The picture with every segment tinted and numbered, so a chat model can point at parts."""
        segs = self.load(iid)
        folder, _ = self._paths(iid)
        with Image.open(self.library.path(iid)) as im:
            base = im.convert("RGBA")
        palette = [(255, 99, 71), (65, 105, 225), (50, 205, 50), (255, 215, 0), (186, 85, 211), (0, 206, 209),
                   (255, 140, 0), (220, 20, 60), (124, 252, 0), (30, 144, 255)]
        draw_items = []
        for i, s in enumerate(segs, 1):
            if s["label"] == "background":
                continue
            m = Image.open(folder / s["file"]).convert("L")
            color = palette[(i - 1) % len(palette)]
            layer = Image.new("RGBA", base.size, color + (0,))
            layer.putalpha(m.point(lambda v: 90 if v else 0))
            base = Image.alpha_composite(base, layer)
            draw_items.append((i, s, color))
        d = ImageDraw.Draw(base)
        try:
            font = ImageFont.load_default(size=max(14, base.width // 45))
        except TypeError:
            font = ImageFont.load_default()
        for i, s, color in draw_items:
            x0, y0, x1, y1 = s["bbox"]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            txt = str(i)
            tw = d.textlength(txt, font=font)
            r = int(max(tw, 10) * 0.9) + 6
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (235,), outline=(255, 255, 255, 255), width=2)
            d.text((cx - tw / 2, cy - r * 0.62), txt, fill=(0, 0, 0, 255), font=font)
        out = base.convert("RGB")
        out.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=88)
        return buf.getvalue()

    def delete_all(self, iid: str) -> None:
        folder, js = self._paths(iid)
        js.unlink(missing_ok=True)
        if folder.exists():
            for f in folder.glob("*.png"):
                f.unlink()
            folder.rmdir()
