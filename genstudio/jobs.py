"""Job queue: runs one generation/edit at a time on the GPU and records the results."""
from __future__ import annotations

import asyncio
import re
import base64
import io
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from . import router, workflows
from .comfy import ComfyClient, ComfyError, iter_images, output_text
from .segments import DESCRIBE_PROMPT, FALLBACK_LABELS, SegmentStore, parse_labels
from .config import Settings
from .library import Library
from .registry import Registry

CLEAR_PROMPT = "empty background, nothing there, seamless continuation of the surroundings, clean"


# ------------------------------------------------------------------ requests
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="그리고 싶은 내용")
    negative: str = ""
    style: Literal["auto", "realistic", "semi", "text"] = "auto"
    model: str = "auto"
    aspect: Literal["auto", "square", "portrait", "landscape", "tall", "wide"] = "auto"
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    seed: int = -1
    count: int = Field(1, ge=1, le=4)
    steps: int | None = Field(None, ge=1, le=80)
    cfg: float | None = Field(None, ge=0, le=20)
    character: str | None = Field(None, description="등록한 캐릭터 이름 또는 id")
    references: list[str] = Field(default_factory=list, description="참고 이미지 id (최대 10장)")
    loras: list[dict] = Field(default_factory=list, description="[{name, strength}] 적용할 LoRA")
    translate: bool | None = Field(None, description="한국어 자동 번역 (비우면 설정값)")
    denoise: float | None = Field(None, ge=0.1, le=1.0,
                                  description="참고 이미지 1장을 다른 화풍으로 다시 그릴 때 바꾸는 정도 (낮을수록 원본 유지)")


ASPECTS = {"1:1": 1.0, "4:5": 0.8, "3:4": 0.75, "2:3": 2 / 3, "9:16": 9 / 16, "5:4": 1.25, "4:3": 4 / 3,
           "3:2": 1.5, "16:9": 16 / 9, "21:9": 21 / 9}


class EditRequest(BaseModel):
    image_id: str
    prompt: str = ""
    negative: str = ""
    # inpaint: redraw only the mask; variation: redraw the whole image lightly;
    # reference: Qwen 2.1 instruction edit with optional extra reference images; outpaint: extend the canvas
    mode: Literal["inpaint", "variation", "reference", "outpaint"] = "inpaint"
    # for inpaint: prompt = draw what the prompt says, auto = let the model decide, clear = remove and blend
    fill: Literal["prompt", "auto", "clear"] = "prompt"
    mask_png: str | None = Field(None, description="base64 PNG, white = area to redraw")
    denoise: float | None = Field(None, ge=0.05, le=1.0)
    model: str = "auto"
    seed: int = -1
    label: str | None = None
    references: list[str] = Field(default_factory=list, description="추가 참고 이미지 id (reference 모드)")
    character: str | None = None
    # outpaint: target aspect ratio (expands only) and where the original sits, or explicit pixel pads
    aspect: str | None = Field(None, description="예: 3:4, 9:16, 16:9")
    anchor: Literal["center", "top", "bottom", "left", "right"] = "center"
    pads: dict[str, int] | None = None
    segments: list[str] = Field(default_factory=list, description="세그먼트 id나 이름. 마스크 대신 쓸 수 있음")
    loras: list[dict] = Field(default_factory=list, description="[{name, strength}] 적용할 LoRA")
    translate: bool | None = None


class CutoutRequest(BaseModel):
    image_id: str
    background: str = Field("transparent", description="transparent | white | black | gray | #rrggbb")
    label: str | None = None


BG_COLORS = {"transparent": None, "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
             "흰색": (255, 255, 255), "검정": (0, 0, 0), "회색": (128, 128, 128), "투명": None}


def parse_bg(value: str) -> tuple[int, int, int] | None:
    v = (value or "transparent").strip().lower()
    if v in BG_COLORS:
        return BG_COLORS[v]
    m = re.fullmatch(r"#?([0-9a-f]{6})", v)
    if not m:
        raise ValueError("배경은 transparent, white, black, gray 또는 #rrggbb 형식이에요")
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def compose_cutout(image: bytes, mask: bytes, background: tuple[int, int, int] | None) -> bytes:
    with Image.open(io.BytesIO(image)) as im, Image.open(io.BytesIO(mask)) as m:
        rgba = im.convert("RGBA")
        alpha = m.convert("L").resize(rgba.size, Image.LANCZOS)
    rgba.putalpha(alpha)
    if background is not None:
        bg = Image.new("RGBA", rgba.size, background + (255,))
        rgba = Image.alpha_composite(bg, rgba).convert("RGB")
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    return buf.getvalue()


class CharacterBuildRequest(BaseModel):
    name: str = Field(..., min_length=1)
    image_id: str
    views: list[Literal["face", "full", "three_quarter", "profile"]] = Field(
        default_factory=lambda: ["face", "full", "three_quarter"])
    aliases: list[str] = Field(default_factory=list)


_SHEET_BASE = ("Image 1 shows a person. Create a clean reference photo of exactly the same person: same face, "
               "eyes, hair, skin, makeup, age and body. {view} Even soft studio lighting, plain light gray background.")
CHARACTER_VIEWS = {
    "face": ("얼굴 정면", "Head-and-shoulders portrait, facing the camera, neutral expression.", (1024, 1024)),
    "full": ("전신", "Full-body photo standing straight and facing the camera, the whole body from head to feet "
                     "visible, same outfit and shoes.", (832, 1216)),
    "three_quarter": ("3/4 측면", "Upper-body photo with the face and body turned three-quarters to the side, "
                                  "same outfit.", (832, 1216)),
    "profile": ("옆모습", "Side profile of the head and shoulders, looking to the side.", (1024, 1024)),
}
DESCRIBE_PERSON = ("Describe the main person in image 1 so another picture can show the same person: apparent age "
                   "range, face shape, eyes, nose, lips, skin tone, hair colour, length and style, body build, and "
                   "outfit. One compact English paragraph under 60 words, no names. Reply with the description only.")


class SegmentRequest(BaseModel):
    image_id: str
    labels: list[str] | None = Field(None, description="찾을 대상 (영어 명사). 비우면 자동으로 찾음")
    points: list[dict] | None = Field(None, description="클릭한 점 [{x, y}] (이미지 픽셀)")
    negative_points: list[dict] | None = None


import re as _re

# "반실사로 만들어줘", "실사화", "그림체 바꿔": the reference is the picture itself, drawn again in another look
RESTYLE_RX = _re.compile(r"(반\s*실사|실사|사진|semi[\s-]?real\w*|photo\w*|realistic)\s*(으로|로|화|느낌|버전|스타일|풍)"
                         r"|그림체|화풍|변환|restyle|convert", _re.I)
RESTYLE_DENOISE = {"semi": 0.55, "realistic": 0.6}
RESTYLE_PIXELS = 1_100_000

RESTYLE_TAGS = """Describe image 1 as a prompt for an Illustrious (SDXL) model: comma-separated English Danbooru-style
tags, most important first (1girl/1boy and count, hair, face, body, clothing, pose, expression, place, lighting,
camera angle and framing). Describe everything that is visible, faithfully. {extra}Reply with the tags only."""
RESTYLE_SENTENCE = """Describe image 1 as a detailed English image-generation prompt in two or three sentences:
subject, face, body, clothing, pose, expression, place, lighting, camera angle and framing. Describe everything that
is visible, faithfully. {extra}Reply with the prompt only."""


def restyle_size(w: int, h: int) -> tuple[int, int]:
    s = (RESTYLE_PIXELS / (w * h)) ** 0.5
    return router.snap(int(w * s)), router.snap(int(h * s))


def restyle_instruction(request: str, family: str) -> str:
    extra = (f"The user's request (it may be Korean): \"{request.strip()}\". Include any change it asks for; "
             "ignore words about the art style. ") if request.strip() else ""
    return (RESTYLE_TAGS if family == "illustrious" else RESTYLE_SENTENCE).format(extra=extra)

# "show more of the body" needs a bigger canvas, not an edit inside the same frame
_EXPAND_RULES = [
    (_re.compile(r"전신|발끝|발까지|신발까지|다리까지|full[\s-]?body|head[\s-]?to[\s-]?toe|whole body", _re.I), "9:16",
     "full body shot, legs and feet visible, shoes"),
    (_re.compile(r"무릎까지|허벅지까지|knees?|thighs?|three[\s-]?quarter", _re.I), "2:3",
     "three-quarter shot, legs visible down to the knees"),
    (_re.compile(r"줌\s*아웃|멀리서|더 넓게|넓게 보이|zoom(ed)?[\s-]?out|wider shot", _re.I), "16:9",
     "wide shot, more of the surroundings"),
]


# a change to what is already there ("thighs firmer", "bigger", "brighter") rather than a new thing in its place
MODIFY_RX = _re.compile(r"더|좀|조금|약간|살짝|하게|키워|줄여|늘려|크게|작게|길게|짧게|진하게|연하게|밝게|어둡게|탄탄|날씬|"
                        r"통통|굵게|얇게|볼륨|bigger|smaller|larger|more |less |slightly|thicker|thinner|longer|shorter|"
                        r"brighter|darker|firmer|toned|slimmer|fuller", _re.I)


# requests that make something bigger need room around the selection, or the new shape is cut at its edge
ENLARGE_RX = _re.compile(r"키워|키우|크게|커지|늘려|늘리|길게|굵게|두껍|풍성|볼륨|넓게|부풀|bigger|larger|longer|thicker|"
                         r"fuller|wider|taller|enlarge|increase|grow", _re.I)


def mask_grow(text: str, width: int, height: int) -> int:
    """Pixels to widen the mask by: generous for 'make it bigger', small otherwise."""
    if ENLARGE_RX.search(text or ""):
        return max(32, int(min(width, height) * 0.08))
    return 12


def qwen_mask_instruction(text: str, fill: str, names: list[str]) -> str:
    what = f"the {', '.join(names)}" if names else "the white area of image 2"
    if fill == "clear":
        return (f"Remove {what} from image 1 and fill that area with the surrounding background so it looks untouched. "
                "Image 2 is a mask: change only its white area, keep everything else in image 1 exactly the same.")
    if fill == "auto":
        return (f"Redraw {what} in image 1 so it fits the picture naturally. "
                "Image 2 is a mask: change only its white area, keep everything else in image 1 exactly the same.")
    room = (" The white area includes some space around it so the changed shape can grow into it; fill any leftover "
            "space with the surrounding background.") if ENLARGE_RX.search(text or "") else ""
    return (f"In image 1, {text.strip()} (this concerns {what}). Image 2 is a mask: change only its white area.{room} "
            "Do only what is asked: do not add, remove or change clothing, accessories, materials or colors that were "
            "not mentioned (for example, bare legs stay bare). Keep the same person, face, pose, background and "
            "lighting exactly as in image 1.")


def expand_intent(text: str, w: int, h: int) -> tuple[str, str, str] | None:
    """(aspect, anchor, prompt addition) when the request asks to see more than the frame holds."""
    for rx, aspect, extra in _EXPAND_RULES:
        if rx.search(text or ""):
            r = ASPECTS[aspect]
            if abs(w / h - r) < 0.02 or (aspect != "16:9" and w / h < r):
                aspect = "9:16" if w / h > 9 / 16 + 0.02 else None
            if aspect is None:
                return None
            anchor = "center" if aspect == "16:9" else "top"
            return aspect, anchor, extra
    return None


def outpaint_pads(w: int, h: int, aspect: str | None, anchor: str = "center", pads: dict | None = None) -> dict:
    """Pixels to add on each side so the image reaches `aspect` without cropping (multiples of 16)."""
    if pads:
        p = {k: max(0, int(pads.get(k, 0))) for k in ("left", "top", "right", "bottom")}
    else:
        if aspect not in ASPECTS:
            raise ValueError(f"비율은 {', '.join(ASPECTS)} 중 하나예요")
        r = ASPECTS[aspect]
        p = {"left": 0, "top": 0, "right": 0, "bottom": 0}
        if w / h < r:                       # needs to get wider
            extra = round(h * r) - w
            if anchor == "left":
                p["right"] = extra
            elif anchor == "right":
                p["left"] = extra
            else:
                p["left"], p["right"] = extra // 2, extra - extra // 2
        elif w / h > r:                     # needs to get taller
            extra = round(w / r) - h
            if anchor == "top":
                p["bottom"] = extra
            elif anchor == "bottom":
                p["top"] = extra
            else:
                p["top"], p["bottom"] = extra // 2, extra - extra // 2
    # round the final canvas up to multiples of 16
    nw, nh = w + p["left"] + p["right"], h + p["top"] + p["bottom"]
    p["right" if p["right"] or not p["left"] else "left"] += (-nw) % 16
    p["bottom" if p["bottom"] or not p["top"] else "top"] += (-nh) % 16
    if not any(p.values()):
        raise ValueError("이미 그 비율이에요")
    if max(w + p["left"] + p["right"], h + p["top"] + p["bottom"]) > 2560:
        raise ValueError("늘린 크기가 너무 커요 (최대 2560px)")
    return p


@dataclass
class Job:
    id: str
    kind: str
    request: dict
    status: str = "queued"           # queued | running | done | error | canceled
    progress: float = 0.0
    message: str = "대기 중"
    decision: dict = field(default_factory=dict)
    result_ids: list[str] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        d["request"] = {k: v for k, v in self.request.items() if k != "mask_png"}
        return d


class Engine:
    def __init__(self, settings: Settings, registry: Registry, library: Library, comfy: ComfyClient,
                 characters: "Characters | None" = None):
        from .characters import Characters
        self.settings = settings
        self.characters = characters or Characters(settings.output_dir / "characters.json")
        self.segments = SegmentStore(library)
        from .loras import Loras
        self.loras = Loras(settings.output_dir / "loras.json",
                           settings.models_dir / "loras" if settings.models_dir else None)
        self.registry = registry
        self.library = library
        self.comfy = comfy
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._last_family: str | None = None
        self._running: str | None = None
        self._done_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------- control
    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass

    def route(self, req: GenerateRequest) -> router.Decision:
        prompt, _neg = router.split_negative(req.prompt)
        return router.choose(
            self.registry, prompt, models_dir=self.settings.models_dir, style=req.style, model_id=req.model,
            aspect=req.aspect, width=req.width, height=req.height, default_style=self.settings.default_style,
        )

    def _reference_model(self, requested: str | None):
        if requested and requested != "auto":
            m = self.registry.get(requested)
            if m.family.encoder != "qwen21":
                raise ValueError(f"{m.name}은(는) 참고 이미지를 받지 못해요. Qwen-Image 2.1을 쓰세요.")
            return m
        m = next((x for x in self.registry.models if x.family.encoder == "qwen21"
                  and x.installed(self.settings.models_dir) is not False), None)
        if not m:
            raise ValueError("참고 이미지 기능에는 Qwen-Image 2.1이 필요해요. install.bat으로 받아 주세요.")
        return m

    def resolve_refs(self, refs: list[str], character: str | None) -> tuple[list[str], dict | None]:
        char = None
        ids: list[str] = []
        if character:
            char = self.characters.get(character)
            if not char:
                raise KeyError(f"캐릭터를 찾을 수 없어요: {character}")
            ids += char["ref_ids"]
        ids += refs
        missing = [i for i in ids if not self.library.get(i)]
        if missing:
            raise KeyError(f"참고 이미지를 찾을 수 없어요: {', '.join(missing)}")
        return ids, char

    def resolve_loras(self, wanted: list[dict]) -> tuple[list[dict], str | None]:
        """[{name|id, strength}] -> ([{name: file, strength, stage, triggers, label}], family)."""
        out, fams = [], set()
        for w in wanted or []:
            key = str(w.get("name") or w.get("id") or "")
            x = self.loras.get(key)
            if not x:
                raise KeyError(f"LoRA를 찾을 수 없어요: {key}")
            if not x.get("family"):
                raise ValueError(f"'{x['name']}'이(가) 어느 모델용인지 몰라요. LoRA 목록에서 계열을 정해 주세요.")
            fams.add(x["family"])
            out.append({"name": x["file"], "label": x["name"], "strength": float(w.get("strength", x.get("strength", 0.8))),
                        "stage": x.get("stage") or "both", "triggers": x.get("triggers") or []})
        if len(fams) > 1:
            raise ValueError("서로 다른 모델용 LoRA는 같이 쓸 수 없어요")
        return out, (fams.pop() if fams else None)

    def _model_for_family(self, family: str, requested: str | None):
        if requested and requested != "auto":
            m = self.registry.get(requested)
            if m.family.id != family:
                raise ValueError(f"고른 LoRA는 {family} 계열용이라 {m.name}에는 쓸 수 없어요")
            return m
        cands = [m for m in self.registry.models if m.family.id == family]
        if not cands:
            raise ValueError("이 LoRA에 맞는 모델이 없어요")
        cands.sort(key=lambda m: (m.installed(self.settings.models_dir) is False, m.role != "main"))
        return cands[0]

    def plan(self, req: GenerateRequest) -> dict:
        """Which model and size a generate request will use (also shown live in the UI)."""
        d = self.route(req)
        decision = d.as_dict()
        loras, lfam = self.resolve_loras(req.loras)
        if loras:
            if lfam != d.model.family.id:
                m = self._model_for_family(lfam, req.model)
                decision.update(model_id=m.id, model_name=m.name, family=m.family.id)
                decision["reasons"] = decision["reasons"] + [f"LoRA → {m.name}"]
            decision["loras"] = loras
        if req.character or req.references:
            ids, char = self.resolve_refs(req.references, req.character)
            if len(ids) > 10:
                raise ValueError("참고 이미지는 캐릭터 사진을 포함해 10장까지예요")
            rm = self._restyle_model(req, d, ids, char, lfam)
            if rm:
                # one picture in another look: redraw it with the style's own model instead of Qwen
                src = self.library.get(ids[0])
                w, h = (router.snap(req.width), router.snap(req.height)) if req.width and req.height \
                    else restyle_size(src["width"], src["height"])
                style = rm.style if rm.style in router.STYLE_LABEL else d.style
                decision.update(model_id=rm.id, model_name=rm.name, family=rm.family.id, style=style, ref_ids=ids,
                                restyle=ids[0], width=w, height=h, aspect="custom", character=None,
                                denoise=req.denoise or RESTYLE_DENOISE.get(style, 0.6))
                decision["reasons"] = [f"참고 이미지를 {router.STYLE_LABEL.get(style, style)}(으)로 다시 그리기",
                                       f"→ {rm.name}"]
                return decision
            m = self._reference_model(req.model)
            if lfam and lfam != m.family.id:
                raise ValueError("참고 이미지는 Qwen-Image 2.1로 그려서, 다른 모델용 LoRA와 같이 쓸 수 없어요")
            decision.update(model_id=m.id, model_name=m.name, family=m.family.id, ref_ids=ids,
                            character=char["name"] if char else None)
            decision["reasons"] = [f"캐릭터 '{char['name']}'" if char else f"참고 이미지 {len(ids)}장",
                                   "→ 참고 이미지를 따르는 Qwen-Image 2.1"]
        return decision

    def _restyle_model(self, req: GenerateRequest, d: router.Decision, ids: list[str], char: dict | None,
                       lfam: str | None):
        """A single reference + a different look (반실사로, 실사화, a non-Qwen model or LoRA) -> that model, else None."""
        if char or len(ids) != 1:
            return None
        if req.model and req.model != "auto":
            m = self.registry.get(req.model)
            return None if m.family.encoder == "qwen21" else m
        if lfam and lfam != "qwen21":
            return self._model_for_family(lfam, None)
        prompt, _ = router.split_negative(req.prompt)
        wants = req.style == "semi" or d.style == "semi" or RESTYLE_RX.search(prompt)
        if wants and d.model.family.encoder != "qwen21":
            return d.model
        return None

    def submit_generate(self, req: GenerateRequest) -> Job:
        decision = self.plan(req)
        job = Job(id=uuid.uuid4().hex[:12], kind="generate", request=req.model_dump(), decision=decision)
        return self._enqueue(job)

    def submit_edit(self, req: EditRequest) -> Job:
        src = self.library.get(req.image_id)
        if not src:
            raise KeyError("편집할 이미지를 찾을 수 없어요")
        if req.segments and not req.mask_png:
            if req.mode not in ("inpaint",):
                req = req.model_copy(update={"mode": "inpaint"})
            req = req.model_copy(update={"mask_png": base64.b64encode(
                self.segments.union_mask(req.image_id, req.segments)).decode()})
        if req.mode == "inpaint" and not req.mask_png:
            raise ValueError("다시 그릴 영역(마스크)이 없어요. 브러시로 영역을 칠하거나 세그먼트를 선택하세요.")
        if req.mode == "inpaint" and req.fill == "prompt" and not req.prompt.strip():
            raise ValueError("무엇을 그릴지 입력하거나 '모델이 알아서 채우기'를 고르세요.")
        extra: dict = {}
        if req.mode in ("reference", "variation") and not req.references and req.prompt.strip():
            hit = expand_intent(req.prompt, src["width"], src["height"])
            if hit:
                aspect, anchor, addition = hit
                req = req.model_copy(update={"mode": "outpaint", "aspect": aspect, "anchor": anchor,
                                             "label": req.label or f"캔버스 늘리기 {aspect}: {req.prompt.strip()}"})
                extra["auto_expand"] = {"aspect": aspect, "anchor": anchor, "add": addition}
        if req.mode == "reference":
            if not req.prompt.strip():
                raise ValueError("어떻게 바꿀지 입력하세요. 예: 배경을 우주선 함교로, 같은 인물 유지")
            ids, char = self.resolve_refs(req.references, req.character)
            if len(ids) > 9:
                raise ValueError("추가 참고 이미지는 9장까지예요 (편집할 이미지가 1번)")
            model, why = self._reference_model(req.model), "참고 편집 → Qwen-Image 2.1"
            extra = {"ref_ids": ids, "character": char["name"] if char else None}
        elif req.mode == "outpaint":
            extra["pads"] = outpaint_pads(src["width"], src["height"], req.aspect, req.anchor, req.pads)
        if req.mode == "reference":
            pass                               # model already chosen above
        elif req.mode == "outpaint" and extra.get("auto_expand") and (not req.model or req.model == "auto"):
            model, why = router.model_for_edit(self.registry, src.get("model_id"), self.settings.models_dir,
                                               self.settings.default_style)
            why = f"화면 밖을 그려야 해서 캔버스 늘리기 ({extra['auto_expand']['aspect']}, 원본은 위에 고정)"
        elif req.model and req.model != "auto":
            model, why = self.registry.get(req.model), "직접 고른 모델"
        else:
            model, why = router.model_for_edit(self.registry, src.get("model_id"), self.settings.models_dir,
                                               self.settings.default_style)
        loras, lfam = self.resolve_loras(req.loras)
        if loras and lfam != model.family.id:
            if req.mode == "reference" or (req.model and req.model != "auto"):
                raise ValueError(f"고른 LoRA는 {model.name}에 쓸 수 없어요")
            model, why = self._model_for_family(lfam, None), "LoRA에 맞는 모델"
        decision = {"model_id": model.id, "model_name": model.name, "family": model.family.id, "reasons": [why],
                    "action": _action_label(req), **extra}
        if loras:
            decision["loras"] = loras
        job = Job(id=uuid.uuid4().hex[:12], kind="edit", request=req.model_dump(), decision=decision)
        return self._enqueue(job)

    def recommend(self, task: str, prompt: str = "", image_id: str | None = None, style: str = "auto",
                  aspect: str = "auto") -> dict:
        """Models for a task with the recommended one marked, for manual pickers.

        task: generate | inpaint (mask/segments) | redraw | instruction | expand
        """
        md = self.settings.models_dir
        src = self.library.get(image_id) if image_id else None
        if task == "generate":
            d = self.route(GenerateRequest(prompt=prompt or " ", style=style, aspect=aspect))
            rec_id, why = d.model.id, ", ".join(d.reasons)
        elif task == "instruction":
            m = next((x for x in self.registry.models if x.family.encoder == "qwen21"), None)
            rec_id, why = (m.id if m else None), "문장 편집과 참고 이미지는 Qwen-Image 2.1만 지원"
        else:
            m, why = router.model_for_edit(self.registry, (src or {}).get("model_id"), md, self.settings.default_style)
            rec_id = m.id
            why = f"{why}라서 그림체가 이어짐" if src and src.get("model_id") else why
        items = []
        for m in self.registry.models:
            ok = task != "instruction" or m.family.encoder == "qwen21"
            items.append({"id": m.id, "name": m.name, "best_for": m.best_for, "style": m.style,
                          "installed": m.installed(md), "compatible": ok, "recommended": m.id == rec_id,
                          "note": "" if ok else "이 작업에는 쓸 수 없음"})
        items.sort(key=lambda x: (not x["recommended"], not x["compatible"], x["installed"] is False))
        return {"task": task, "recommended": rec_id, "reason": why, "models": items}

    def vlm_file(self) -> str | None:
        """Qwen3-VL text encoder that ships with Qwen-Image 2.1 doubles as the image describer."""
        for fam in self.registry.families.values():
            if fam.encoder == "qwen21" and fam.clip:
                spec = next((f for f in fam.files if f.name == fam.clip), None)
                if spec and spec.installed(self.settings.models_dir) is not False:
                    return fam.clip
        return None

    def submit_segment(self, req: SegmentRequest) -> Job:
        if not self.library.get(req.image_id):
            raise KeyError("이미지를 찾을 수 없어요")
        if not self.registry.tool_ready("sam3", self.settings.models_dir):
            raise ValueError("세그먼트 모델(SAM 3.1)이 없어요. update.bat을 실행해 받아 주세요.")
        if req.points:
            action, how = "클릭 선택", "SAM 3"
        elif req.labels:
            action, how = "세그먼트 분석", "지정한 대상"
        else:
            action, how = "세그먼트 분석", "Qwen3-VL이 대상 찾기" if self.vlm_file() else "기본 단어 목록"
        job = Job(id=uuid.uuid4().hex[:12], kind="segment", request=req.model_dump(),
                  decision={"action": action, "labeler": how, "model_name": "SAM 3.1"})
        return self._enqueue(job)

    def submit_character_build(self, req: CharacterBuildRequest) -> Job:
        if not self.library.get(req.image_id):
            raise KeyError("이미지를 찾을 수 없어요")
        existing = self.characters.get(req.name)
        if existing and existing["name"] == req.name.strip():
            raise ValueError(f"이미 같은 이름의 캐릭터가 있어요: {req.name}")
        from .characters import MAX_REFS
        views = list(dict.fromkeys(req.views))[: MAX_REFS - 1]
        if not views:
            raise ValueError("만들 사진 종류를 하나 이상 고르세요")
        m = self._reference_model(None)
        job = Job(id=uuid.uuid4().hex[:12], kind="character", request={**req.model_dump(), "views": views},
                  decision={"action": "캐릭터 만들기", "model_id": m.id, "model_name": m.name, "family": m.family.id,
                            "views": [CHARACTER_VIEWS[v][0] for v in views]})
        return self._enqueue(job)

    async def _run_character(self, job: Job) -> None:
        req = CharacterBuildRequest(**job.request)
        src = self.library.get(req.image_id)
        path = self.library.path(req.image_id)
        if not src or not path:
            raise ComfyError("이미지가 사라졌어요")
        model = self.registry.get(job.decision["model_id"])
        name = await self.comfy.upload_image(path.read_bytes(), f"{src['id']}.png")
        description = ""
        vlm = self.vlm_file()
        if vlm:
            job.message = "인물 특징 읽는 중"
            try:
                from . import translate as tr
                description = tr.clean(output_text(await self.comfy.run_outputs(
                    workflows.describe_objects(name, vlm, DESCRIBE_PERSON, max_length=200))), "")
            except ComfyError:
                description = ""
        await self._switch_family(model.family.id)
        n = len(req.views)
        for i, v in enumerate(req.views):
            ko, view, (w, h) = CHARACTER_VIEWS[v]
            job.message = f"{ko} 사진 만드는 중 ({i + 1}/{n})"
            prompt = _SHEET_BASE.format(view=view) + (f" The person: {description}" if description else "")
            seed = random.randint(0, 2**31 - 1)
            graph, extra = workflows.reference(model, [name], prompt, "", seed, width=w, height=h,
                                               models_dir=self.settings.models_dir)
            outs = await self.comfy.run(graph, self._progress(job, i / n, 1 / n))
            data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
            rec = self.library.add(data, {
                "kind": "edit", "source": "gen-studio", "parent_id": src["id"], "prompt": src.get("prompt", ""),
                "final_prompt": extra["final_prompt"], "model_id": model.id, "family": model.family.id, "seed": seed,
                "label": f"캐릭터: {req.name} · {ko}", "edit": {"mode": "character_sheet", "view": v},
            })
            job.result_ids.append(rec["id"])
        char = self.characters.save(req.name, [src["id"], *job.result_ids], description, aliases=req.aliases)
        job.result = {"character": char}

    # ------------------------------------------------------------ detail pass
    def _detail_model(self, src: dict, requested: str):
        if requested and requested != "auto":
            m = self.registry.get(requested)
            if m.family.encoder == "qwen21" or m.family.high_noise:
                raise ValueError(f"{m.name}은(는) 디테일 보정에 쓰지 않아요. Z-Image나 Illustrious를 고르세요.")
            return m
        m, _why = router.model_for_edit(self.registry, src.get("model_id"), self.settings.models_dir,
                                        self.settings.default_style)
        if m.family.encoder == "qwen21" or m.family.high_noise:   # Qwen / Wan pictures: redraw with the style model
            style = "semi" if src.get("style") == "semi" else "realistic"
            m = router.pick_model(self.registry, style, self.settings.models_dir, [])
        return m

    def submit_detail(self, req) -> Job:
        from . import detail as D
        src = self.library.get(req.image_id)
        if not src:
            raise KeyError("이미지를 찾을 수 없어요")
        if not self.registry.tool_ready("sam3", self.settings.models_dir):
            raise ValueError("세그먼트 모델(SAM 3.1)이 없어요. update.bat을 실행해 받아 주세요.")
        targets = list(dict.fromkeys(req.targets)) or ["face"]
        m = self._detail_model(src, req.model)
        names = ", ".join(D.TARGETS[t][3] for t in targets)
        job = Job(id=uuid.uuid4().hex[:12], kind="detail", request={**req.model_dump(), "targets": targets},
                  decision={"action": f"디테일 보정 ({names})", "model_id": m.id, "model_name": m.name,
                            "family": m.family.id})
        return self._enqueue(job)

    async def _run_detail(self, job: Job) -> None:
        from . import detail as D
        req = D.DetailRequest(**job.request)
        src = self.library.get(req.image_id)
        path = self.library.path(req.image_id)
        if not src or not path:
            raise ComfyError("이미지가 사라졌어요")
        model = self.registry.get(job.decision["model_id"])
        with Image.open(path) as im:
            base = im.convert("RGB")
        name = await self.comfy.upload_image(D.png(base), f"{src['id']}-detail.png")
        job.message = "얼굴·손 찾는 중"
        graph, saves = workflows.segment_labels(name, self.registry.tool_file("sam3"),
                                                [D.TARGETS[t][0] for t in req.targets])
        outputs = await self.comfy.run_outputs(graph, self._progress(job, 0, 0.15))
        found: list[tuple[str, Image.Image]] = []
        for nid, d in iter_images(outputs):
            if nid in saves:
                found.append((req.targets[saves[nid]], D.load_mask(
                    await self.comfy.view(d["filename"], d.get("subfolder", "")), base.size)))
        plan = []
        for t in req.targets:
            boxes = [(m, D.crop_box(m, D.TARGETS[t][1])) for tt, m in found if tt == t]
            boxes = sorted([(m, b) for m, b in boxes if b], key=lambda x: -(x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))
            plan += [(t, m, b) for m, b in boxes[: D.MAX_PER_TARGET]]
        if not plan:
            raise ComfyError(f"{', '.join(D.TARGETS[t][3] for t in req.targets)}을(를) 찾지 못했어요")
        extra = await self._maybe_translate(job, req.prompt.strip(), model.family.id, None) if req.prompt.strip() else ""
        await self._switch_family(model.family.id)
        n = len(plan)
        for i, (t, mask, box) in enumerate(plan):
            job.message = f"{D.TARGETS[t][3]} 다시 그리는 중 ({i + 1}/{n})"
            crop = base.crop(box)
            mcrop = mask.crop(box)
            ww, wh = D.work_size(*crop.size)
            img_name = await self.comfy.upload_image(D.png(crop.resize((ww, wh), Image.LANCZOS)), f"{src['id']}-d{i}.png")
            m_name = await self.comfy.upload_image(D.png(mcrop.resize((ww, wh), Image.LANCZOS)), f"{src['id']}-d{i}-m.png")
            graph, _ = workflows.inpaint(model, img_name, m_name, D.prompt_for(t, model.family.id, extra), "",
                                         random.randint(0, 2**31 - 1), denoise=req.strength, grow=8,
                                         models_dir=self.settings.models_dir, prefix="gen-studio/detail")
            outs = await self.comfy.run(graph, self._progress(job, 0.15 + 0.85 * i / n, 0.85 / n))
            base = D.paste_back(base, await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", "")),
                                box, mcrop)
        names = ", ".join(dict.fromkeys(D.TARGETS[t][3] for t, _m, _b in plan))
        rec = self.library.add(D.png(base), {
            "kind": "edit", "source": "gen-studio", "parent_id": src["id"], "prompt": src.get("prompt", ""),
            "model_id": model.id, "family": model.family.id, "style": src.get("style"),
            "label": f"디테일 보정 ({names})", "edit_prompt": req.prompt,
            "edit": {"mode": "detail", "targets": req.targets, "strength": req.strength, "count": n},
        })
        job.result_ids.append(rec["id"])
        job.result = {"fixed": [{"target": t, "box": list(b)} for t, _m, b in plan]}

    # ------------------------------------------------------------ LoRA training
    def submit_train(self, req) -> Job:
        from . import training as T
        ids = list(req.image_ids)
        if req.character:
            ch = self.characters.get(req.character)
            if not ch:
                raise KeyError(f"캐릭터를 찾을 수 없어요: {req.character}")
            ids = list(dict.fromkeys([*ch["ref_ids"], *ids]))
        missing = [i for i in ids if not self.library.get(i)]
        if missing:
            raise KeyError(f"이미지를 찾을 수 없어요: {', '.join(missing)}")
        if not T.MIN_IMAGES <= len(ids) <= T.MAX_IMAGES:
            raise ValueError(f"학습 이미지는 {T.MIN_IMAGES}~{T.MAX_IMAGES}장이 필요해요 (지금 {len(ids)}장)")
        fam = req.family
        if fam == "auto":
            fams = [self.library.get(i).get("family") for i in ids]
            counts = {f: fams.count(f) for f in T.TRAINABLE}
            fam = max(counts, key=lambda f: (counts[f], f == "zimage"))
        cands = [m for m in self.registry.models if m.family.id == fam and m.installed(self.settings.models_dir) is not False]
        if not cands:
            raise ValueError(f"{T.TRAINABLE[fam]['name']} 모델이 설치돼 있지 않아요")
        model = sorted(cands, key=lambda m: m.role != "main")[0]
        if self.loras.get(req.name):
            raise ValueError(f"이미 같은 이름의 LoRA가 있어요: {req.name}")
        if not self.loras.dir:
            raise ValueError("LoRA를 저장할 ComfyUI 폴더를 몰라요. config/settings.yaml의 comfyui path를 확인하세요.")
        trigger = (req.trigger or T.make_trigger(req.name)).strip()
        steps = req.steps or T.default_steps(fam, len(ids))
        decision = {"action": "LoRA 학습", "model_id": model.id, "model_name": model.name, "family": fam,
                    "images": len(ids), "steps": steps, "rank": req.rank, "trigger": trigger,
                    "learning_rate": req.learning_rate or T.TRAINABLE[fam]["lr"], "image_ids": ids,
                    "reasons": [f"{T.TRAINABLE[fam]['name']}용", f"사진 {len(ids)}장 · {steps}스텝",
                                f"트리거 단어 '{trigger}'"]}
        job = Job(id=uuid.uuid4().hex[:12], kind="train", request=req.model_dump(), decision=decision)
        return self._enqueue(job)

    async def _run_train(self, job: Job) -> None:
        from . import training as T
        from . import translate as tr
        req = T.TrainRequest(**job.request)
        d = job.decision
        model = self.registry.get(d["model_id"])
        tag = f"{re.sub(r'[^a-z0-9]+', '-', req.name.lower()).strip('-') or 'lora'}-{job.id[:6]}"
        folder = f"gen-studio-train/{tag}"
        vlm = self.vlm_file()
        n = len(d["image_ids"])
        captions = []
        for i, iid in enumerate(d["image_ids"], 1):
            job.message = f"학습 사진 준비 {i}/{n}"
            job.progress = 0.1 * i / n
            png = T.prepare_image(self.library.path(iid).read_bytes())
            name = await self.comfy.upload_image(png, f"{i:04d}.png", subfolder=folder)
            cap = ""
            if vlm:
                try:
                    cap = tr.clean(output_text(await self.comfy.run_outputs(workflows.describe_objects(
                        name, vlm, T.caption_instruction(req.kind, d["family"], d["trigger"]), max_length=160))), "")
                except ComfyError:
                    cap = ""
            text = ", ".join(x for x in (d["trigger"], cap) if x)
            captions.append(text)
            await self.comfy.upload_image(text.encode("utf-8"), f"{i:04d}.txt", subfolder=folder)
        await self._switch_family(model.family.id)
        job.message = "학습 중 (몇십 분 걸려요)"
        prefix = f"loras/gen-studio/{tag}"
        graph = T.train_graph(model, folder, d["steps"], d["rank"], d["learning_rate"], random.randint(0, 2**31 - 1),
                              prefix, models_dir=self.settings.models_dir)

        async def prog(p, msg):
            job.progress = 0.1 + 0.88 * p
            job.message = f"학습 중 {msg.split()[-1]}" if msg and "/" in msg else "학습 중"
        try:
            await self.comfy.run_outputs(graph, prog, timeout=6 * 3600)
        except ComfyError as e:
            if any(x in str(e) for x in T.TRAIN_NODES):
                raise ComfyError("ComfyUI가 오래돼서 학습 노드가 없어요. ComfyUI_windows_portable\\update\\"
                                 "update_comfyui.bat 을 실행한 뒤 다시 켜 주세요.") from e
            if "out of memory" in str(e).lower():
                raise ComfyError("그래픽 메모리가 부족해요. rank를 8로 낮추거나 다른 프로그램을 닫고 다시 해 보세요.") from e
            raise
        await self.comfy.free()
        self._last_family = None
        data = await self.comfy.view(f"{tag}_{d['steps']}_steps_00001_.safetensors", "loras/gen-studio")
        if not self.loras.dir:
            raise ComfyError("LoRA 폴더를 몰라요 (ComfyUI 경로 설정 확인)")
        self.loras.dir.mkdir(parents=True, exist_ok=True)
        fname = f"{tag}.safetensors"
        (self.loras.dir / fname).write_bytes(data)
        item = self.loras.upsert(fname, name=req.name, family=d["family"], triggers=[d["trigger"]], strength=0.9,
                                 note=f"Gen Studio에서 학습 · 사진 {n}장 · {d['steps']}스텝")
        job.result = {"lora": item, "captions": captions}

    def submit_cutout(self, req: CutoutRequest) -> Job:
        if not self.library.get(req.image_id):
            raise KeyError("이미지를 찾을 수 없어요")
        parse_bg(req.background)
        if not self.registry.tool_ready("bg_removal", self.settings.models_dir):
            raise ValueError("배경 제거 모델(BiRefNet)이 없어요. update.bat을 실행해 받아 주세요.")
        job = Job(id=uuid.uuid4().hex[:12], kind="cutout", request=req.model_dump(),
                  decision={"action": "배경 제거", "model_name": "BiRefNet"})
        return self._enqueue(job)

    async def _run_cutout(self, job: Job) -> None:
        req = CutoutRequest(**job.request)
        src = self.library.get(req.image_id)
        path = self.library.path(req.image_id)
        if not src or not path:
            raise ComfyError("이미지가 사라졌어요")
        data = path.read_bytes()
        name = await self.comfy.upload_image(data, f"{src['id']}.png")
        job.message = "배경을 나누는 중"
        try:
            outputs = await self.comfy.run_outputs(
                workflows.remove_background(name, self.registry.tool_file("bg_removal")), self._progress(job, 0, 1))
        except ComfyError as e:
            if "RemoveBackground" in str(e) or "LoadBackgroundRemovalModel" in str(e):
                raise ComfyError("ComfyUI가 오래돼서 배경 제거 노드가 없어요. "
                                 "ComfyUI_windows_portable\\update\\update_comfyui.bat 을 실행한 뒤 다시 켜 주세요.") from e
            raise
        imgs = list(iter_images(outputs))
        if not imgs:
            raise ComfyError("배경 제거 결과가 없어요")
        _n, d = imgs[0]
        mask = await self.comfy.view(d["filename"], d.get("subfolder", ""))
        color = parse_bg(req.background)
        png = compose_cutout(data, mask, color)
        bg_label = {None: "투명"}.get(color, req.background)
        rec = self.library.add(png, {
            "kind": "edit", "source": "gen-studio", "parent_id": src["id"], "prompt": src.get("prompt", ""),
            "model_id": src.get("model_id"), "family": src.get("family"), "style": src.get("style"),
            "label": req.label or f"배경 제거 ({bg_label})",
            "edit": {"mode": "cutout", "background": req.background},
        })
        job.result_ids.append(rec["id"])

    def _enqueue(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self._done_events[job.id] = asyncio.Event()
        self._queue.put_nowait(job.id)
        self._trim()
        return job

    def _trim(self, keep: int = 200) -> None:
        if len(self.jobs) > keep:
            for jid in sorted(self.jobs, key=lambda j: self.jobs[j].created_at)[: len(self.jobs) - keep]:
                if self.jobs[jid].status in ("done", "error", "canceled"):
                    self.jobs.pop(jid, None)
                    self._done_events.pop(jid, None)

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status in ("done", "error", "canceled"):
            return False
        if job.status == "queued":
            job.status, job.message = "canceled", "취소됨"
            self._finish(job)
            return True
        await self.comfy.interrupt()
        return True

    async def wait(self, job_id: str, timeout: float) -> Job:
        ev = self._done_events.get(job_id)
        if ev:
            try:
                await asyncio.wait_for(ev.wait(), timeout)
            except asyncio.TimeoutError:
                pass
        return self.jobs[job_id]

    def queue_position(self, job_id: str) -> int:
        waiting = [j for j in self.jobs.values() if j.status == "queued"]
        waiting.sort(key=lambda j: j.created_at)
        ids = [j.id for j in waiting]
        return ids.index(job_id) + 1 if job_id in ids else 0

    # -------------------------------------------------------------- worker
    async def _loop(self) -> None:
        while True:
            jid = await self._queue.get()
            job = self.jobs.get(jid)
            if not job or job.status != "queued":
                continue
            job.status, job.started_at, job.message = "running", time.time(), "준비 중"
            self._running = jid
            try:
                if job.kind == "generate":
                    await self._run_generate(job)
                elif job.kind == "segment":
                    await self._run_segment(job)
                elif job.kind == "cutout":
                    await self._run_cutout(job)
                elif job.kind == "character":
                    await self._run_character(job)
                elif job.kind == "train":
                    await self._run_train(job)
                elif job.kind == "detail":
                    await self._run_detail(job)
                else:
                    await self._run_edit(job)
                job.status, job.progress, job.message = "done", 1.0, "완료"
            except ComfyError as e:
                job.status, job.error, job.message = "error", str(e), "실패"
            except Exception as e:  # keep the worker alive no matter what
                job.status, job.error, job.message = "error", f"{type(e).__name__}: {e}", "실패"
            finally:
                self._running = None
                self._finish(job)

    def _finish(self, job: Job) -> None:
        job.finished_at = time.time()
        ev = self._done_events.get(job.id)
        if ev:
            ev.set()

    async def _switch_family(self, family_id: str) -> None:
        # different model families rarely fit in 16GB together; unload before switching
        if self._last_family and self._last_family != family_id:
            await self.comfy.free()
        self._last_family = family_id

    def _progress(self, job: Job, base: float, span: float):
        def cb(p: float, text: str) -> None:
            job.progress = round(base + span * p, 3)
            job.message = text
        return cb

    async def _run_generate(self, job: Job) -> None:
        req = GenerateRequest(**job.request)
        model = self.registry.get(job.decision["model_id"])
        prompt, neg_inline = router.split_negative(req.prompt)
        negative = ", ".join(x for x in (req.negative, neg_inline) if x)
        prompt_ko = prompt
        restyle = job.decision.get("restyle")
        if restyle:
            src_name, prompt = await self._restyle_source(job, restyle, prompt, model)
            ref_names = []
        else:
            prompt = await self._maybe_translate(job, prompt, model.family.id, req.translate)
        await self._switch_family(model.family.id)
        if not restyle:
            ref_names = await self._upload_refs(job.decision.get("ref_ids") or [])
        loras = job.decision.get("loras") or []
        if loras:
            from .loras import apply_triggers
            prompt = apply_triggers(prompt, loras)
        if job.decision.get("character"):
            prompt = _with_identity(prompt, job.decision["character"], self.characters.get(job.decision["character"]))
        n = req.count
        for i in range(n):
            seed = req.seed + i if req.seed >= 0 else random.randint(0, 2**31 - 1)
            if restyle:
                graph, extra = workflows.img2img(
                    model, src_name, prompt, negative, seed, job.decision["denoise"], models_dir=self.settings.models_dir,
                    steps=req.steps, cfg=req.cfg, loras=loras)
            elif ref_names:
                graph, extra = workflows.reference(
                    model, ref_names, prompt, negative, seed, width=job.decision["width"], height=job.decision["height"],
                    models_dir=self.settings.models_dir, cfg=req.cfg, loras=loras)
            else:
                graph, extra = workflows.txt2img(
                    model, prompt, negative, job.decision["width"], job.decision["height"], seed,
                    models_dir=self.settings.models_dir, steps=req.steps, cfg=req.cfg, loras=loras)
            outs = await self.comfy.run(graph, self._progress(job, i / n, 1 / n))
            data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
            rec = self.library.add(data, {
                "kind": "generate", "source": "gen-studio", "prompt": prompt_ko, "prompt_en": prompt if prompt != prompt_ko else None,
                "negative": negative,
                "final_prompt": extra["final_prompt"], "model_id": model.id, "family": model.family.id,
                "style": job.decision["style"], "seed": seed, "steps": req.steps or model.family.steps,
                "cfg": req.cfg if req.cfg is not None else model.family.cfg,
                "label": f"{router.STYLE_LABEL.get(job.decision['style'], '')} 변환".strip() if restyle else "원본",
                "denoise": job.decision.get("denoise") if restyle else None,
                "ref_ids": job.decision.get("ref_ids") or [], "character": job.decision.get("character"),
                "loras": [{"name": l["label"], "strength": l["strength"]} for l in loras],
            })
            job.result_ids.append(rec["id"])

    async def _run_edit(self, job: Job) -> None:
        req = EditRequest(**job.request)
        src = self.library.get(req.image_id)
        src_path = self.library.path(req.image_id)
        if not src or not src_path:
            raise ComfyError("편집할 이미지가 사라졌어요")
        model = self.registry.get(job.decision["model_id"])
        await self._switch_family(model.family.id)
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
        image_name = await self.comfy.upload_image(src_path.read_bytes(), f"{src['id']}.png")

        base_prompt = src.get("prompt_en") or src.get("prompt", "") or ""
        req0 = req                              # what the user typed, for labels and history
        if req.mode != "reference" and req.prompt.strip():
            req = req.model_copy(update={"prompt": await self._maybe_translate(job, req.prompt.strip(), model.family.id,
                                                                               req.translate)})
        loras = job.decision.get("loras") or []
        lw = {"loras": loras} if loras else {}
        if req.mode == "inpaint":
            if req.fill == "auto":
                prompt = base_prompt            # model fills to match the whole picture
            elif req.fill == "clear":
                prompt = CLEAR_PROMPT
            else:
                prompt = req.prompt.strip()
            mask = _normalize_mask(req.mask_png, (src["width"], src["height"]))
            mask_bytes = mask
            mask_name = await self.comfy.upload_image(mask, f"{src['id']}-mask-{job.id}.png")
            if model.family.encoder == "qwen21":
                names = self._segment_names(src["id"], req0.segments)
                graph, extra = workflows.masked_reference(
                    model, image_name, mask_name, qwen_mask_instruction(req0.prompt, req.fill, names), req.negative, seed,
                    grow=mask_grow(req0.prompt, src["width"], src["height"]), models_dir=self.settings.models_dir, **lw)
            else:
                denoise = req.denoise
                if req.fill == "prompt" and denoise is None and MODIFY_RX.search(req0.prompt):
                    # keep the shape and identity; add the change on top of the original description
                    prompt = ", ".join(x for x in (base_prompt, req.prompt.strip()) if x)
                    denoise = 0.6
                graph, extra = workflows.inpaint(model, image_name, mask_name, prompt, req.negative, seed,
                                                 denoise=denoise or 1.0, grow=mask_grow(req0.prompt, src["width"], src["height"]),
                                                 models_dir=self.settings.models_dir, **lw)
        elif req.mode == "reference":
            mask_bytes = None
            refs = await self._upload_refs(job.decision.get("ref_ids") or [])
            prompt = req.prompt.strip()
            if job.decision.get("character"):
                prompt = _with_identity(prompt, job.decision["character"], self.characters.get(job.decision["character"]),
                                        editing=True)
            graph, extra = workflows.reference(model, [image_name, *refs], prompt, req.negative, seed,
                                               models_dir=self.settings.models_dir, **lw)
        elif req.mode == "outpaint":
            mask_bytes = None
            auto = job.decision.get("auto_expand")
            if model.family.encoder == "qwen21":
                prompt = ", ".join(x for x in (auto["add"] if auto else "", req0.prompt.strip()) if x)
            else:
                prompt = ", ".join(x for x in (base_prompt, auto["add"]) if x) if auto else (req.prompt.strip() or base_prompt)
            graph, extra = workflows.outpaint(model, image_name, job.decision["pads"], prompt, req.negative, seed,
                                              models_dir=self.settings.models_dir, **lw)
        else:
            mask_bytes = None
            prompt = req.prompt.strip() or base_prompt
            graph, extra = workflows.img2img(model, image_name, prompt, req.negative, seed,
                                             req.denoise or 0.55, models_dir=self.settings.models_dir, **lw)
        outs = await self.comfy.run(graph, self._progress(job, 0, 1))
        data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
        rec = self.library.add(data, {
            "loras": [{"name": l["label"], "strength": l["strength"]} for l in loras],
            "kind": "edit", "source": src.get("source", "gen-studio"), "parent_id": src["id"],
            "prompt": src.get("prompt", "") if req.mode in ("inpaint", "reference", "outpaint") else req0.prompt,
            "edit_prompt": req0.prompt, "edit_prompt_en": req.prompt if req.prompt != req0.prompt else None,
            "final_prompt": extra["final_prompt"], "model_id": model.id, "family": model.family.id, "seed": seed,
            "edit": {"mode": req.mode, "fill": req.fill, "denoise": req.denoise, "references": req.references,
                     "character": req.character, "aspect": req.aspect, "anchor": req.anchor,
                     "pads": job.decision.get("pads")},
            "label": req0.label or _action_label(req0),
        })
        if mask_bytes:
            self.library.update(rec["id"], mask_file=self.library.save_side_file(rec["id"], ".mask.png", mask_bytes))
        job.result_ids.append(rec["id"])

    async def _restyle_source(self, job: Job, ref_id: str, request: str, model) -> tuple[str, str]:
        """Upload the reference at the working size and describe it (plus the user's request) for the target model."""
        path = self.library.path(ref_id)
        if not path:
            raise ComfyError(f"참고 이미지가 사라졌어요: {ref_id}")
        w, h = job.decision["width"], job.decision["height"]
        with Image.open(path) as im:
            buf = io.BytesIO()
            im.convert("RGB").resize((w, h), Image.LANCZOS).save(buf, "PNG")
        name = await self.comfy.upload_image(buf.getvalue(), f"restyle-{ref_id}-{w}x{h}.png")
        ref = self.library.get(ref_id) or {}
        fallback = ", ".join(x for x in (ref.get("prompt_en") or ref.get("prompt") or "", request) if x)
        vlm = self.vlm_file()
        if not vlm:
            return name, fallback
        job.message = "참고 이미지를 읽는 중"
        try:
            outputs = await self.comfy.run_outputs(
                workflows.describe_objects(name, vlm, restyle_instruction(request, model.family.id)))
        except ComfyError:
            return name, fallback
        from . import translate as tr
        text = tr.clean(output_text(outputs), fallback)
        job.result["caption"] = text
        return name, text

    async def _maybe_translate(self, job: Job, text: str, family: str, flag: bool | None) -> str:
        """Korean request -> English prompt for models that need it (uses the Qwen3-VL already downloaded)."""
        from . import translate as tr
        mode = "always" if flag is True else "never" if flag is False else self.settings.translate
        if not text or not tr.should_translate(text, family, mode):
            return text
        vlm = self.vlm_file()
        if not vlm:
            return text
        job.message = "한국어를 영어 프롬프트로 바꾸는 중"
        try:
            outputs = await self.comfy.run_outputs(workflows.rewrite_text(vlm, tr.instruction(text, family)))
        except ComfyError:
            return text                       # translation is a nicety; never block the picture on it
        result = tr.clean(output_text(outputs), text)
        job.result.setdefault("translated", []).append({"from": text, "to": result})
        return result

    async def _run_segment(self, job: Job) -> None:
        req = SegmentRequest(**job.request)
        src_path = self.library.path(req.image_id)
        image_name = await self.comfy.upload_image(src_path.read_bytes(), f"{req.image_id}.png")
        sam = self.registry.tool_file("sam3")
        if req.points:
            graph = workflows.segment_points(image_name, sam, req.points, req.negative_points)
            outputs = await self.comfy.run_outputs(graph, self._progress(job, 0, 1))
            masks = [await self.comfy.view(d["filename"], d.get("subfolder", "")) for _n, d in iter_images(outputs)]
            seg = self.segments.add_click(req.image_id, masks[0]) if masks else None
            if not seg:
                raise ComfyError("그 위치에서 물체를 찾지 못했어요. 조금 다른 곳을 눌러 보세요.")
            job.result = {"segments": self.segments.load(req.image_id), "new": [seg["id"]]}
            return

        if req.labels:
            from .segments import korean_to_label
            from .translate import has_korean
            infos = []
            for l in (x.strip() for x in req.labels if x.strip()):
                en = l.lower()
                if has_korean(l):
                    en = korean_to_label(l) or ""
                    vlm = self.vlm_file()
                    if not en and vlm:          # anything the word list does not know: ask Qwen3-VL
                        outputs = await self.comfy.run_outputs(workflows.rewrite_text(
                            vlm, f"Give the English noun phrase (1-3 words) for this thing in a photo: {l}. Reply with it only."))
                        en = re.sub(r"[^a-zA-Z '\-]", "", output_text(outputs)).strip().lower()
                    if not en:
                        raise ComfyError(f"'{l}'을(를) 찾을 단어로 바꾸지 못했어요. 영어로 적어 주세요.")
                infos.append({"label": en, "ko": l if has_korean(l) else "", "parent": None})
        else:
            infos = []
            vlm = self.vlm_file()
            if vlm:
                job.message = "무엇이 있는지 살펴보는 중"
                outputs = await self.comfy.run_outputs(workflows.describe_objects(image_name, vlm, DESCRIBE_PROMPT),
                                                       self._progress(job, 0, 0.4))
                infos = parse_labels(output_text(outputs))
            if not infos:
                infos = [{"label": a, "ko": b, "parent": c} for a, b, c in FALLBACK_LABELS]
        graph, saves = workflows.segment_labels(image_name, sam, [i["label"] for i in infos])
        outputs = await self.comfy.run_outputs(graph, self._progress(job, 0.4, 0.6))
        per_label: dict[int, list[bytes]] = {}
        for nid, d in iter_images(outputs):
            if nid in saves:
                per_label.setdefault(saves[nid], []).append(await self.comfy.view(d["filename"], d.get("subfolder", "")))
        detections = [(infos[i], per_label.get(i, [])) for i in range(len(infos))]
        segs = self.segments.save_detections(req.image_id, detections, "labels" if req.labels else "auto")
        job.result = {"segments": segs, "labels": [i["label"] for i in infos]}

    def _segment_names(self, image_id: str, keys: list[str]) -> list[str]:
        if not keys:
            return []
        try:
            segs = self.segments.resolve(image_id, keys)
        except KeyError:
            return []
        names: list[str] = []
        for s in segs:
            if s["label"] not in ("selection",) and s["label"] not in names:
                names.append(s["label"])
        return names

    async def _upload_refs(self, ids: list[str]) -> list[str]:
        names = []
        for rid in ids:
            p = self.library.path(rid)
            if not p:
                raise ComfyError(f"참고 이미지가 사라졌어요: {rid}")
            names.append(await self.comfy.upload_image(p.read_bytes(), f"ref-{rid}.png"))
        return names

    def regenerate(self, image_id: str) -> Job:
        """Same request as the one that made this image, with a new seed."""
        rec = self.library.get(image_id)
        if not rec:
            raise KeyError("이미지를 찾을 수 없어요")
        if rec.get("kind") == "generate":
            return self.submit_generate(GenerateRequest(
                prompt=rec.get("prompt") or " ", negative=rec.get("negative", ""), model=rec.get("model_id") or "auto",
                width=rec["width"], height=rec["height"], steps=rec.get("steps"), cfg=rec.get("cfg"),
                references=[r for r in rec.get("ref_ids") or []
                            if not rec.get("character") or r not in (self.characters.get(rec["character"]) or {}).get("ref_ids", [])],
                character=rec.get("character"), loras=rec.get("loras") or [], denoise=rec.get("denoise")))
        if rec.get("kind") == "edit" and rec.get("parent_id"):
            edit = rec.get("edit") or {}
            if edit.get("mode") == "character_sheet":
                raise ValueError("캐릭터 사진은 캐릭터 만들기로 다시 만들어 주세요")
            if edit.get("mode") == "detail":
                from .detail import DetailRequest
                return self.submit_detail(DetailRequest(image_id=rec["parent_id"], targets=edit.get("targets") or ["face"],
                                                        strength=edit.get("strength") or 0.4,
                                                        prompt=rec.get("edit_prompt") or ""))
            if edit.get("mode") == "cutout":
                return self.submit_cutout(CutoutRequest(image_id=rec["parent_id"],
                                                        background=edit.get("background") or "transparent"))
            mask_b64 = None
            if rec.get("mask_file"):
                mask_b64 = base64.b64encode((self.library.root / rec["mask_file"]).read_bytes()).decode()
            return self.submit_edit(EditRequest(
                image_id=rec["parent_id"], prompt=rec.get("edit_prompt", ""), mode=edit.get("mode", "variation"),
                fill=edit.get("fill", "prompt"), mask_png=mask_b64, denoise=edit.get("denoise"),
                model=rec.get("model_id") or "auto", label=rec.get("label"), references=edit.get("references") or [],
                character=edit.get("character"), aspect=edit.get("aspect"), anchor=edit.get("anchor") or "center",
                pads=edit.get("pads") if edit.get("mode") == "outpaint" else None, loras=rec.get("loras") or []))
        raise ValueError("불러온 원본 이미지는 재생성할 수 없어요. 편집으로 바꿔 보세요.")


def _with_identity(prompt: str, name: str, char: dict | None, editing: bool = False) -> str:
    """Tell Qwen which reference is the person to keep."""
    plural = "s" if char and len(char.get("ref_ids", [])) > 1 else ""
    who = f"{name} (the person in the reference image{plural})"
    desc = f" {char['description']}." if char and char.get("description") else ""
    keep = f"Keep {who} exactly the same: same face, hair, body and identity.{desc}"
    return f"{prompt}\n{keep}" if editing else f"{keep} {prompt}"


def _action_label(req: EditRequest) -> str:
    if req.mode == "reference":
        return f"참고 편집: {req.prompt.strip()}"
    if req.mode == "outpaint":
        return f"캔버스 늘리기 {req.aspect or ''}".strip()
    if req.mode == "variation":
        return f"전체 다시 그리기: {req.prompt.strip()}" if req.prompt.strip() else "전체 다시 그리기"
    if req.fill == "auto":
        return "모델이 알아서 채우기"
    if req.fill == "clear":
        return "지우고 비우기"
    return f"부분 다시 그리기: {req.prompt.strip()}"


def _normalize_mask(b64: str | None, size: tuple[int, int]) -> bytes:
    """Accepts any PNG (alpha or white strokes) and returns a black/white RGB mask at the image size."""
    if not b64:
        raise ValueError("마스크가 비어 있어요")
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            m = im.convert("RGBA").split()[3]
        else:
            m = im.convert("L")
        m = m.resize(size, Image.NEAREST).point(lambda v: 255 if v > 20 else 0)
        if m.getbbox() is None:
            raise ValueError("다시 그릴 영역이 비어 있어요")
        buf = io.BytesIO()
        Image.merge("RGB", (m, m, m)).save(buf, "PNG")
        return buf.getvalue()
