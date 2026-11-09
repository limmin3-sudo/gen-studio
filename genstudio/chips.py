"""The editor's own quick-insert chips ("내 칩"): a short label and the sentence it adds to the edit box.

The owner writes them; they live in data/chips.json on this PC only. The defaults are a few general edits.
"""
from __future__ import annotations

import json

from . import config as cfg

MAX_CHIPS = 80
MAX_LABEL = 24
MAX_TEXT = 600

DEFAULTS = [
    {"label": "검은 티로", "text": "선택한 옷을 검은 티셔츠로 바꾸기, 몸에 자연스럽게 맞게"},
    {"label": "청바지로", "text": "선택한 하의를 청바지로 바꾸기, 자연스러운 원단 질감"},
    {"label": "옷 자국 정리", "text": "남은 옷 경계와 자국을 깔끔하게 정리하고 이어지는 부분을 자연스럽게"},
    {"label": "피부 자연스럽게", "text": "피부 질감을 자연스럽게, 주변 피부색과 조명에 맞게"},
    {"label": "경계 다듬기", "text": "고친 부분과 나머지 그림의 경계를 매끄럽게"},
    {"label": "조명 맞추기", "text": "원래 그림의 조명과 그림자에 정확히 맞추기"},
]


def _path():
    return cfg.DATA_DIR / "chips.json"


def load() -> list[dict]:
    try:
        items = json.loads(_path().read_text(encoding="utf-8"))
        return items if isinstance(items, list) else list(DEFAULTS)
    except (FileNotFoundError, ValueError):
        return [dict(c) for c in DEFAULTS]


def clean(items) -> list[dict]:
    if not isinstance(items, list):
        raise ValueError("칩 목록 형식이 맞지 않아요")
    out, seen = [], set()
    for c in items:
        if not isinstance(c, dict):
            continue
        label = str(c.get("label") or "").strip()[:MAX_LABEL]
        text = str(c.get("text") or "").strip()[:MAX_TEXT]
        if not label or not text or (label, text) in seen:
            continue
        seen.add((label, text))
        out.append({"label": label, "text": text})
    if len(out) > MAX_CHIPS:
        raise ValueError(f"칩은 {MAX_CHIPS}개까지예요")
    return out


def save(items) -> list[dict]:
    items = clean(items)
    _path().parent.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return items


def parse_bulk(text: str) -> list[dict]:
    """'이름 | 문장' per line (a tab works too); lines without a separator use the start of the sentence as the name."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("|", "\t"):
            if sep in line:
                label, body = line.split(sep, 1)
                break
        else:
            label, body = line[:MAX_LABEL], line
        if label.strip() and body.strip():
            out.append({"label": label.strip(), "text": body.strip()})
    return out
