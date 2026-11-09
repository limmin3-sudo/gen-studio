"""Picks a model, size and settings from what the user typed.

Rule-based on purpose: predictable, instant, and easy to tweak.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .registry import ModelSpec, Registry

TEXT_WORDS = ["글자", "글씨", "텍스트", "문구", "간판", "포스터", "로고", "타이포", "현수막", "배너", "메뉴판", "책 표지",
              "앨범 커버", "인포그래픽", "카드뉴스", "썸네일 문구", "명함", "전단", "광고 문구", "자막",
              "typography", "poster", "logo", "signboard", "neon sign", "banner", "infographic", "lettering",
              "book cover", "album cover", "with the text", "that says", "reads \"", "written"]
_QUOTED = re.compile(r'["“”「」『』][^"“”「」『』]{2,40}["“”「」『』]')

STYLE_WORDS = {
    "semi": ["반실사", "반 실사", "2.5d", "2.5 d", "semi-real", "semi real", "semireal", "일러스트풍", "일러스트 풍",
             "illustration", "illustrated", "그림체", "웹툰", "게임 cg", "game cg", "렌더", "3d render"],
    "realistic": ["실사", "사진", "photo", "photograph", "photoreal", "realistic", "real life", "dslr", "film still",
                  "인물사진", "스냅", "raw photo", "35mm", "85mm"],
}

ASPECT_WORDS = {
    "portrait": ["세로", "portrait", "vertical", "전신", "full body", "full-body", "9:16", "2:3", "3:4", "폰 배경", "phone wallpaper"],
    "landscape": ["가로", "landscape", "horizontal", "wide", "와이드", "16:9", "3:2", "풍경", "배경화면", "wallpaper"],
    "square": ["정사각", "square", "1:1", "프로필", "profile picture"],
}

SIZES = {
    "square": (1024, 1024),
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "tall": (768, 1344),
    "wide": (1344, 768),
}

STYLE_LABEL = {"realistic": "실사", "semi": "반실사", "text": "글자·디자인"}
ASPECT_LABEL = {"square": "정사각", "portrait": "세로", "landscape": "가로", "tall": "세로(9:16)", "wide": "가로(16:9)"}


@dataclass
class Decision:
    model: ModelSpec
    style: str
    aspect: str
    width: int
    height: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model.id,
            "model_name": self.model.name,
            "family": self.model.family.id,
            "style": self.style,
            "aspect": self.aspect,
            "width": self.width,
            "height": self.height,
            "reasons": self.reasons,
        }


def _find(text: str, table: dict[str, list[str]]) -> tuple[str | None, str | None]:
    low = text.lower()
    best = (None, None, -1)
    for key, words in table.items():
        for w in words:
            i = low.find(w)
            if i >= 0 and (best[2] < 0 or len(w) > len(best[1] or "")):
                best = (key, w, i)
    return best[0], best[1]


def detect_style(text: str) -> tuple[str | None, str | None]:
    # Lettering needs the model that can spell; check it first.
    t, w0 = _find(text, {"text": TEXT_WORDS})
    if t:
        return "text", w0
    m = _QUOTED.search(text)
    if m:
        return "text", m.group(0)
    # "반실사" contains "실사", so check semi words first and let the longer match win.
    semi, w1 = _find(text, {"semi": STYLE_WORDS["semi"]})
    if semi:
        return "semi", w1
    real, w2 = _find(text, {"realistic": STYLE_WORDS["realistic"]})
    return (real, w2) if real else (None, None)


def detect_aspect(text: str) -> tuple[str | None, str | None]:
    return _find(text, ASPECT_WORDS)


def snap(v: int, step: int = 16) -> int:
    return max(step, int(round(v / step)) * step)


def choose(
    registry: Registry,
    prompt: str,
    *,
    models_dir: Path | None = None,
    style: str = "auto",
    model_id: str | None = None,
    aspect: str = "auto",
    width: int | None = None,
    height: int | None = None,
    default_style: str = "realistic",
) -> Decision:
    reasons: list[str] = []

    # 1) style
    if model_id and model_id != "auto":
        model = registry.get(model_id)
        chosen_style = model.style
        reasons.append("직접 고른 모델")
    else:
        kw_model, kw = _keyword_model(registry, prompt, models_dir) if style == "auto" else (None, None)
        if kw_model:
            model = kw_model
            chosen_style = model.style
            reasons.append(f"'{kw}' → {model.name}")
        elif style in STYLE_LABEL:
            chosen_style = style
            reasons.append(f"{STYLE_LABEL[style]} 선택")
        else:
            found, word = detect_style(prompt)
            if found:
                chosen_style = found
                reasons.append(f"'{word}' → {STYLE_LABEL[found]}")
            else:
                chosen_style = default_style
                reasons.append(f"스타일 언급 없음 → 기본값 {STYLE_LABEL.get(default_style, default_style)}")
        if not kw_model:
            model = pick_model(registry, chosen_style, models_dir, reasons)

    # 2) size
    if width and height:
        w, h, chosen_aspect = snap(width), snap(height), "custom"
        reasons.append(f"크기 {w}×{h}")
    else:
        if aspect in SIZES:
            chosen_aspect = aspect
        else:
            found, word = detect_aspect(prompt)
            chosen_aspect = found or "square"
            if found:
                reasons.append(f"'{word}' → {ASPECT_LABEL[found]}")
        w, h = SIZES[chosen_aspect]

    return Decision(model=model, style=chosen_style, aspect=chosen_aspect, width=w, height=h, reasons=reasons)


def _keyword_model(registry: Registry, prompt: str, models_dir: Path | None):
    low = prompt.lower()
    for m in registry.models:
        for w in m.keywords:
            if w.lower() in low and m.installed(models_dir) is not False:
                return m, w
    return None, None


def pick_model(registry: Registry, style: str, models_dir: Path | None, reasons: list[str]) -> ModelSpec:
    candidates = [m for m in registry.models if m.covers(style)] or list(registry.models)
    # prefer installed models, then the main one; helpers only when nothing else fits
    def score(m: ModelSpec):
        inst = m.installed(models_dir)
        return (0 if inst in (True, None) else 1, {"main": 0, "alt": 1}.get(m.role, 2))
    candidates.sort(key=score)
    best = candidates[0]
    if best.installed(models_dir) is False:
        reasons.append("설치된 모델이 없어 기본 모델로 시도")
    elif best.role != "main" and not any(m.role == "main" and m.covers(style) for m in candidates):
        reasons.append("메인 모델이 없어 보조 모델 사용")
    return best


def model_for_edit(registry: Registry, source_model_id: str | None, models_dir: Path | None,
                   default_style: str = "realistic") -> tuple[ModelSpec, str]:
    """Edits redraw with the same model (or at least the same family) that made the image."""
    if source_model_id and source_model_id in registry.by_id:
        m = registry.by_id[source_model_id]
        if m.installed(models_dir) is not False:
            return m, "원본을 만든 모델"
        same = [x for x in registry.models if x.family is m.family and x.installed(models_dir) is not False]
        if same:
            return same[0], "원본과 같은 계열"
    reasons: list[str] = []
    return pick_model(registry, default_style, models_dir, reasons), "외부 이미지 → 기본 스타일 모델"


_NEG_SPLIT = re.compile(r"\s*(?:--no|negative:|부정:)\s*", re.I)


def split_negative(prompt: str) -> tuple[str, str]:
    """Lets people write 'prompt --no hats, text' in one line."""
    parts = _NEG_SPLIT.split(prompt, maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (prompt.strip(), "")
