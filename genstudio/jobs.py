"""Job queue: runs one generation/edit at a time on the GPU and records the results."""
from __future__ import annotations

import asyncio
import re
import base64
import io
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

import logging

from . import poses, router, workflows
from .comfy import ComfyClient, ComfyError, iter_images, output_text
from .segments import DESCRIBE_PROMPT, FALLBACK_LABELS, SegmentStore, parse_labels
from .config import Settings
from .library import Library
from .registry import Registry

CLEAR_PROMPT = "empty background, nothing there, seamless continuation of the surroundings, clean"


# ------------------------------------------------------------------ requests
class Region(BaseModel):
    box: list[float] = Field(..., min_length=4, max_length=4)
    prompt: str = Field(..., min_length=1)

    def clamp(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in self.box)
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def region_position(box: tuple[float, float, float, float]) -> str:
    """'on the left', 'at the top right' ... from where the box sits."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h = "left" if cx < 0.38 else "right" if cx > 0.62 else ""
    v = "top" if cy < 0.33 else "bottom" if cy > 0.67 else ""
    if h and v:
        return f"at the {v} {h}"
    if h:
        return f"on the {h}"
    if v:
        return f"at the {v}"
    return "in the center"


UNCHANGED = 4.0          # mean pixel difference (0-255) inside the mask below which an edit "did nothing"


def masked_change(before: bytes, after: bytes, mask_png: bytes | None) -> float:
    """How much the selected area changed: mean absolute difference inside the mask, on small copies."""
    if not mask_png:
        return 255.0
    from PIL import ImageChops, ImageStat
    with Image.open(io.BytesIO(before)) as a, Image.open(io.BytesIO(after)) as b, Image.open(io.BytesIO(mask_png)) as m:
        size = (256, max(1, round(256 * a.height / a.width)))
        a2, b2 = a.convert("L").resize(size), b.convert("L").resize(size)
        m2 = m.convert("L").resize(size).point(lambda v: 255 if v > 127 else 0)
        if not m2.getbbox():
            return 255.0
        return ImageStat.Stat(ImageChops.difference(a2, b2), mask=m2).mean[0]


def fit_on_black(png: bytes, w: int, h: int) -> bytes:
    """Fit a pose map inside w x h without stretching; the rest stays black (= no body there)."""
    with Image.open(io.BytesIO(png)) as im:
        im = im.convert("RGB")
        s = min(w / im.width, h / im.height)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
        buf = io.BytesIO()
        canvas.save(buf, "PNG")
        return buf.getvalue()


def region_mask(box: tuple[float, float, float, float], w: int, h: int) -> bytes:
    from PIL import ImageDraw, ImageFilter
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle([box[0] * w, box[1] * h, box[2] * w, box[3] * h], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(max(2, min(w, h) // 80)))
    buf = io.BytesIO()
    m.save(buf, "PNG")
    return buf.getvalue()


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
    pose: str | None = Field(None, description="이 이미지의 포즈대로 그리기 (사진 또는 포즈 맵 id)")
    pose_strength: float = Field(0.85, ge=0.2, le=1.5, description="포즈를 따르는 정도")
    regions: list[Region] = Field(default_factory=list, max_length=6,
                                  description="영역별 프롬프트: box는 0~1 비율 [왼쪽, 위, 오른쪽, 아래]")


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


class Keyframe(BaseModel):
    image_id: str = Field(..., description="이 순간에 나올 장면 (이미지 id)")
    at: float = Field(..., description="몇 초에 (0 = 처음, -1 = 마지막)")


class VideoRequest(BaseModel):
    image_id: str = Field(..., description="첫 장면")
    prompt: str = Field("", description="어떻게 움직일지 (비우면 자연스러운 움직임)")
    seconds: int = Field(5, ge=2, le=10, description="5초가 가장 자연스러움. 중간 장면이 있으면 10초까지 자연스러움")
    quality: Literal["480p", "720p"] = "720p"
    width: int | None = Field(None, ge=256, le=1920, description="직접 크기 (16의 배수로 맞춤). 하나만 주면 비율 유지")
    height: int | None = Field(None, ge=256, le=1920)
    end_image_id: str | None = Field(None, description="마지막 장면")
    mid_image_id: str | None = Field(None, description="가운데 장면 (앞 절반과 뒤 절반을 따로 만들어 이어 붙임)")
    negative: str = Field("", description="나오면 안 되는 것")
    steps: int = Field(4, ge=2, le=30, description="4가 기본 (가속 모델). 늘리면 느려지고 조금 더 정교해질 수 있음")
    smooth: bool = Field(True, description="프레임 보간으로 16fps → 32fps (모델이 있을 때)")
    smooth_multiplier: int = Field(2, ge=2, le=4, description="보간 배수: 2 = 32fps, 3 = 48fps, 4 = 64fps")
    seed: int = -1


VIDEO_AREA = {"480p": 832 * 480, "720p": 1280 * 720}
VIDEO_FPS = 16


def video_files(specs) -> list[str]:
    """[high, low, text encoder, vae, (lora high, lora low)] from the tool's file list, whatever its order."""
    def pick(cond):
        return next((f.name for f in specs if cond(f)), None)
    unets = [f for f in specs if f.dir == "diffusion_models"]
    high = next((f.name for f in unets if "high" in f.name.lower()), unets[0].name if unets else None)
    low = next((f.name for f in unets if "low" in f.name.lower()), unets[-1].name if unets else None)
    out = [high, low, pick(lambda f: f.dir == "text_encoders"), pick(lambda f: f.dir == "vae")]
    lh = pick(lambda f: f.dir == "loras" and "high" in f.name.lower())
    ll = pick(lambda f: f.dir == "loras" and "low" in f.name.lower())
    return out + [lh, ll] if lh and ll else out


def ltx_files(specs) -> list[str]:
    """[transformer, text encoder, video vae, audio vae, upscaler] from the tool's file list, whatever its order."""
    def pick(cond):
        return next((f.name for f in specs if cond(f)), None)
    return [pick(lambda f: f.dir == "diffusion_models"), pick(lambda f: f.dir == "text_encoders"),
            pick(lambda f: f.dir == "vae" and "audio" not in f.name), pick(lambda f: f.dir == "vae" and "audio" in f.name),
            pick(lambda f: f.dir == "latent_upscale_models")]


def video_size(w: int, h: int, quality: str) -> tuple[int, int]:
    k = (VIDEO_AREA[quality] / (w * h)) ** 0.5
    return max(256, round(w * k / 16) * 16), max(256, round(h * k / 16) * 16)


class LtxRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="장면·움직임·카메라·소리(대사, 효과음, 음악)까지 자세히")
    image_id: str | None = Field(None, description="첫 장면으로 쓸 이미지 (없으면 글만으로 만듦)")
    end_image_id: str | None = Field(None, description="마지막 장면")
    mid_image_id: str | None = Field(None, description="가운데 장면")
    keyframes: list[Keyframe] = Field(default_factory=list, description="원하는 초에 원하는 장면 (여러 개)")
    seconds: int = Field(5, ge=2, le=20, description="10초 넘으면 오래 걸리고 메모리가 빠듯함")
    quality: Literal["540p", "720p"] = "720p"
    orientation: Literal["auto", "landscape", "portrait", "square"] = "auto"
    width: int | None = Field(None, ge=256, le=1920, description="직접 크기 (64의 배수로 맞춤). 하나만 주면 비율 유지")
    height: int | None = Field(None, ge=256, le=1920)
    fps: int = Field(24, ge=12, le=50)
    image_strength: float = Field(0.7, ge=0.1, le=1.0, description="장면 사진을 얼마나 그대로 따를지")
    audio: bool = Field(True, description="소리(대사·효과음·음악)까지 만들기")
    negative: str = ""
    seed: int = -1


class VideoFrameRequest(BaseModel):
    video_id: str
    at: float | Literal["first", "last"] = Field("last", description="'first', 'last', 또는 몇 초")


class JoinVideosRequest(BaseModel):
    video_ids: list[str] = Field(..., min_length=2, max_length=20, description="이어 붙일 영상 순서대로")
    audio: bool | None = Field(None, description="소리 이어 붙이기 (비우면 모든 영상에 소리가 있을 때만)")


LTX_AREA = {"540p": 960 * 544, "720p": 1280 * 720}
LTX_SHAPES = {"landscape": (16, 9), "portrait": (9, 16), "square": (1, 1)}
LTX_FPS = 24


def custom_size(w: int, h: int, width: int | None, height: int | None, step: int) -> tuple[int, int]:
    """Size asked for directly; one side alone keeps the picture's shape."""
    if width and not height:
        height = width * h / w
    elif height and not width:
        width = height * w / h
    return max(256, round(width / step) * step), max(256, round(height / step) * step)


def ltx_guides(req: "LtxRequest") -> list[tuple[str, float]]:
    """Every pinned picture as (image id, second), in time order; -1 = the last frame."""
    g = [(req.image_id, 0.0)] if req.image_id else []
    if req.mid_image_id:
        g.append((req.mid_image_id, req.seconds / 2))
    g += [(k.image_id, float(k.at)) for k in req.keyframes]
    if req.end_image_id:
        g.append((req.end_image_id, -1.0))
    return sorted(g, key=lambda x: float("inf") if x[1] < 0 or x[1] >= req.seconds else x[1])


def ltx_frame(at: float, seconds: int, fps: int) -> int:
    return -1 if at < 0 or at >= seconds else round(at * fps)


def ltx_size(w: int, h: int, quality: str) -> tuple[int, int]:
    """Same picture shape, about 0.5MP (540p) or 0.9MP (720p); multiples of 64 so the half-size first pass fits."""
    k = (LTX_AREA[quality] / (w * h)) ** 0.5
    return max(256, round(w * k / 64) * 64), max(256, round(h * k / 64) * 64)


class UpscaleRequest(BaseModel):
    image_id: str
    scale: Literal[2, 4] = 2
    refine: bool | None = Field(None, description="키운 뒤 디테일을 살짝 다시 그리기 (비우면 자동: 켬)")
    denoise: float | None = Field(None, ge=0.1, le=0.6)
    prompt: str = ""


class PoseSaveRequest(BaseModel):
    people: list = Field(..., description="사람마다 {body: 관절 18개, right_hand/left_hand: 21개, face: 68개, feet: 6개}"
                                            " 각 점 [x, y] (0~1) 또는 null. 관절 18개 목록만 줘도 됨")
    width: int = Field(768, ge=256, le=2048)
    height: int = Field(1152, ge=256, le=2048)
    name: str = Field("", max_length=60, description="자세 이름 (저장해 두고 다시 쓰기)")
    parent_id: str | None = Field(None, description="어느 사진에서 시작했는지")
    edit: dict | None = Field(None, description="편집기가 다시 열 때 쓰는 값 (깊이, 카메라, 손 모양)")


class PoseKey(BaseModel):
    pose_id: str = Field(..., description="뼈대 자세(make_pose / 편집기) 또는 자세를 따라 할 사진")
    at: float = Field(..., description="몇 초에 이 자세 (0 = 처음, -1 = 마지막)")


class PoseVideoRequest(BaseModel):
    image_id: str = Field(..., description="움직일 사람·장면 사진")
    poses: list[PoseKey] = Field(..., min_length=1, max_length=6)
    prompt: str = Field("", description="장면·움직임·카메라·소리")
    engine: Literal["ltx", "wan"] = "ltx"
    keep_start: bool = Field(True, description="원래 사진을 첫 장면으로 (0초에 자세가 있으면 그 자세가 첫 장면)")
    seconds: int = Field(5, ge=2, le=20)
    quality: Literal["480p", "540p", "720p"] = "720p"
    fps: int = Field(24, ge=12, le=50, description="LTX만")
    audio: bool = Field(True, description="LTX만")
    image_strength: float = Field(0.7, ge=0.1, le=1.0, description="LTX만")
    smooth: bool = Field(True, description="Wan만: 프레임 보간")
    negative: str = ""
    seed: int = -1


class ReposeRequest(BaseModel):
    image_id: str
    pose: str | None = Field(None, description="이 사진(또는 포즈 맵)의 자세로 바꾸기")
    prompt: str = Field("", description="말로 하는 자세 설명, 또는 포즈와 함께 덧붙일 말")


REPOSE_KEEP = ("Keep the same person: same face, hair, body, skin and clothing (or lack of it), and keep the same "
               "background, lighting and image style.")


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

log = logging.getLogger("genstudio.jobs")

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
# asking for a different body posture: a masked redraw can't move limbs outside their old outline, so these
# requests go to "자세 바꾸기" (whole picture redrawn, same person/clothes/background)
POSE_RX = _re.compile(r"자세|포즈|앉[아게은혀히는]|앉도록|누워|눕[게혀는]|엎드|무릎\s*꿇|쪼그려|쪼그리|"
                      r"다리\s*를?\s*(\S+\s+){0,2}(벌|꼬|모아|들어|올려|뻗)|양\s*다리|팔\s*을?\s*(\S+\s+){0,3}(들|올려|벌|뻗|모아)|"
                      r"뒤\s*돌|돌아\s*서|돌아\s*보|옆으로\s*(누|서|돌)|일어서|기대[어게]|점프|"
                      r"[Mm]\s*자|\bsit(s|ting)?\b|\bseated\b|\bkneel|\blying\b|\blie(s)? (down|on)|\bsquat|"
                      r"(spread|cross|open|raise|lift)\w*\s+(\w+\s+){0,2}(legs?|arms?|thighs?)|"
                      r"(legs?|arms?|thighs?|knees?)\s+(\w+\s+){0,2}(spread|apart|open|wide|crossed|raised|up|bent|together)|"
                      r"\bm[- ]?shape|on all fours|bent over|bend(s|ing)? over|turn(s|ing)?\s+around|facing away|"
                      r"\bpose\b|\bposture\b|\bposition\b|hands? on (her|his|their) (hips|knees|head)", _re.I)

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
        self._task: asyncio.Task | None = None          # the running job, so a cancel can stop it between graphs too
        self._cancel: set[str] = set()
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
        for i in ids:
            self._need_still(i)
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
        if req.pose:
            self._plan_pose(req, decision)
        if req.character or req.references:
            ids, char = self.resolve_refs(req.references, req.character)
            if len(ids) > 10:
                raise ValueError("참고 이미지는 캐릭터 사진을 포함해 10장까지예요")
            rm = self._restyle_model(req, d, ids, char, lfam)
            if rm and req.pose:
                raise ValueError("그림체 바꾸기와 포즈 지정은 같이 쓸 수 없어요")
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
            if req.pose:
                if len(ids) >= 10:
                    raise ValueError("참고 이미지가 10장이면 포즈 이미지를 더 넣을 수 없어요")
                decision["pose"]["kind"] = "qwen"
                decision["reasons"].append("포즈 이미지도 참고")
        return decision

    def _plan_pose(self, req: GenerateRequest, decision: dict) -> None:
        """Pick how the pose is applied for the chosen model and size the picture like the pose image."""
        rec = self.library.get(req.pose)
        if not rec:
            raise KeyError(f"포즈 이미지를 찾을 수 없어요: {req.pose}")
        self._need_still(req.pose)
        is_map = rec.get("kind") == "pose"
        if not is_map and not self.registry.tool_ready("pose_detect", self.settings.models_dir):
            raise ValueError("포즈 인식 모델(SDPose)이 없어요. update.bat을 실행해 받아 주세요.")
        fam = decision["family"]
        auto = not req.model or req.model == "auto"
        if fam == "wan22" and not auto:
            raise ValueError("Wan 2.2는 포즈를 따라 그리지 못해요. 다른 모델을 고르세요.")
        if fam == "wan22" or (fam == "qwen21" and auto and not (req.references or req.character)):
            # a ControlNet follows the skeleton exactly; Qwen only sees it as a picture (kept for reference work)
            m = self.helper_model("semi" if decision.get("style") == "semi" else "realistic")
            if m is not None and self.registry.tool_ready(
                    "control_zimage" if m.family.id == "zimage" else "control_sdxl_pose", self.settings.models_dir):
                decision.update(model_id=m.id, model_name=m.name, family=m.family.id)
                fam = m.family.id
            elif fam == "wan22":
                raise ValueError("포즈를 따라 그릴 모델이 없어요")
        kind = {"zimage": "zimage", "illustrious": "sdxl", "qwen21": "qwen"}.get(fam)
        tool = {"zimage": "control_zimage", "sdxl": "control_sdxl_pose"}.get(kind)
        if tool and not self.registry.tool_ready(tool, self.settings.models_dir):
            raise ValueError("포즈용 컨트롤넷 모델이 없어요. update.bat을 실행해 받아 주세요.")
        if not (req.width and req.height) and req.aspect == "auto":
            r = rec["width"] / rec["height"]
            name = min(router.SIZES, key=lambda k: abs(router.SIZES[k][0] / router.SIZES[k][1] - r))
            decision.update(aspect=name, width=router.SIZES[name][0], height=router.SIZES[name][1])
        decision["pose"] = {"image": req.pose, "kind": kind, "is_map": is_map, "strength": req.pose_strength,
                            "file": self.registry.tool_file(tool) if tool else None}
        decision["reasons"] = decision["reasons"] + ["포즈 따라 그리기"]

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
        if req.regions:
            for r in req.regions:
                x0, y0, x1, y1 = r.clamp()
                if (x1 - x0) * (y1 - y0) < 0.005:
                    raise ValueError("영역이 너무 작아요")
            decision["regions"] = len(req.regions)
            decision["reasons"] = decision["reasons"] + [f"영역 {len(req.regions)}개 따로 그리기"]
        job = Job(id=uuid.uuid4().hex[:12], kind="generate", request=req.model_dump(), decision=decision)
        return self._enqueue(job)

    def submit_edit(self, req: EditRequest) -> Job:
        self._need_still(req.image_id)
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
        if req.mode == "inpaint" and req.fill == "prompt" and req.prompt.strip() and POSE_RX.search(req.prompt):
            job = self.submit_repose(ReposeRequest(image_id=req.image_id, prompt=req.prompt.strip()))
            job.decision["reasons"] = ["자세를 바꾸는 요청이라 고른 영역만이 아니라 그림 전체를 다시 그려요"]
            return job
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
        self._need_still(req.image_id)
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
        self._need_still(req.image_id)
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

    # ------------------------------------------------------------ image -> video
    def _need_still(self, iid: str) -> None:
        rec = self.library.get(iid)
        if rec and rec.get("media") == "video":
            raise ValueError("영상은 편집할 수 없어요. 영상을 만든 원본 이미지에서 작업해 주세요.")

    # ------------------------------------------------------------ LTX-2.5: Codex's high-quality video with sound
    def submit_ltx(self, req: LtxRequest) -> Job:
        return self._enqueue(self._ltx_job(req))

    def _ltx_job(self, req: LtxRequest) -> Job:
        if not self.registry.tool_ready("video_ltx", self.settings.models_dir):
            raise ValueError("LTX-2.5 영상 모델이 없어요. Gen Studio 설정 → 'Codex 전용 고성능 영상'에서 받아 주세요.")
        guides = ltx_guides(req)
        for iid, _ in guides:
            if not self.library.get(iid):
                raise KeyError(f"이미지를 찾을 수 없어요: {iid}")
            self._need_still(iid)
        first = self.library.get(guides[0][0]) if guides else None
        if req.orientation != "auto" or not first:
            w, h = LTX_SHAPES["landscape" if req.orientation == "auto" else req.orientation]
        else:
            w, h = first["width"], first["height"]
        w, h = custom_size(w, h, req.width, req.height, 64) if req.width or req.height else ltx_size(w, h, req.quality)
        frames = req.seconds * req.fps + 1
        mode = "keyframes" if any(at != 0 for _, at in guides) else "image" if guides else "text"
        job = Job(id=uuid.uuid4().hex[:12], kind="ltx", request=req.model_dump(),
                  decision={"action": f"LTX 영상 {req.seconds}초" + (f" · 장면 {len(guides)}개" if mode == "keyframes" else ""),
                            "model_name": "LTX-2.5 22B (영상+소리)" if req.audio else "LTX-2.5 22B",
                            "family": "ltx", "width": w, "height": h, "frames": frames, "fps": req.fps, "mode": mode,
                            "guides": [{"image_id": i, "frame": ltx_frame(at, req.seconds, req.fps)} for i, at in guides]})
        return job

    async def _upload_fit(self, iid: str, w: int, h: int, tag: str) -> tuple[str, bytes]:
        """The picture cropped to cover w×h, uploaded to ComfyUI. Returns (name, png)."""
        path = self.library.path(iid)
        if not path:
            raise ComfyError(f"이미지가 사라졌어요: {iid}")
        with Image.open(path) as im:
            im = im.convert("RGB")
            s = max(w / im.width, h / im.height)                  # cover the frame, crop the overflow
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            x, y = (im.width - w) // 2, (im.height - h) // 2
            im = im.crop((x, y, x + w, y + h))
            buf = io.BytesIO()
            im.save(buf, "PNG")
        png = buf.getvalue()
        return await self.comfy.upload_image(png, f"{tag}-{iid}-{w}x{h}.png"), png

    async def _run_ltx(self, job: Job) -> None:
        req = LtxRequest(**job.request)
        d = job.decision
        w, h = d["width"], d["height"]
        fps = d.get("fps") or LTX_FPS
        guides = ltx_guides(req)
        names = [(await self._upload_fit(iid, w, h, "ltx-key"))[0] for iid, _ in guides]
        src = self.library.get(req.image_id) if req.image_id else None
        await self.comfy.free()                                   # a 22B model and a 12B text encoder: start clean
        self._last_family = None
        job.message = "LTX 영상 만드는 중 (수 분~수십 분)"
        files = dict(zip(("transformer", "text_encoder", "video_vae", "audio_vae", "upscaler"),
                         ltx_files(self.registry.tools["video_ltx"])))
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
        if d.get("mode") == "keyframes":
            graph = workflows.ltx_keyframes(req.prompt.strip(), files, w, h, req.seconds, seed,
                                            [(n, g["frame"]) for n, g in zip(names, d["guides"])],
                                            negative=req.negative, fps=fps, strength=req.image_strength, audio=req.audio)
        else:
            graph = workflows.ltx_video(req.prompt.strip(), files, w, h, req.seconds, seed,
                                        image_name=names[0] if names else None, negative=req.negative, fps=fps,
                                        strength=req.image_strength, audio=req.audio)
        outs = await self.comfy.run(graph, self._progress(job, 0, 1), timeout=4 * 3600)
        clip = next((o for o in outs if str(o.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov"))), None)
        poster_item = next((o for o in outs if str(o.get("filename", "")).lower().endswith(".png")), None)
        if not clip:
            raise ComfyError("영상 파일이 저장되지 않았어요")
        data = await self.comfy.view(clip["filename"], clip.get("subfolder", ""))
        poster = await self.comfy.view(poster_item["filename"], poster_item.get("subfolder", "")) if poster_item else None
        if poster is None:
            buf = io.BytesIO()
            Image.new("RGB", (w, h), (20, 20, 20)).save(buf, "PNG")
            poster = buf.getvalue()
        await self.comfy.free()
        rec = self.library.add_video(data, poster, {
            "kind": "video", "source": "gen-studio", "parent_id": src["id"] if src else None,
            "prompt": req.prompt, "edit_prompt": req.prompt, "final_prompt": req.prompt, "model_id": None,
            "family": "ltx", "seed": seed, "label": f"LTX 영상 {req.seconds}초", "fps": fps,
            "frames": d["frames"], "duration": round((d["frames"] - 1) / fps, 2), "audio": req.audio,
            "keyframes": d.get("guides") or [],
            "edit": {"mode": "ltx", "seconds": req.seconds, "quality": req.quality, "orientation": req.orientation,
                     "negative": req.negative, "request": {**req.model_dump(), "seed": -1}},
        }, ext="." + clip["filename"].rsplit(".", 1)[-1].lower())
        job.result_ids.append(rec["id"])

    def submit_video(self, req: VideoRequest) -> Job:
        return self._enqueue(self._video_job(req))

    def _video_job(self, req: VideoRequest) -> Job:
        src = self.library.get(req.image_id)
        if not src:
            raise KeyError("이미지를 찾을 수 없어요")
        if src.get("media") == "video":
            raise ValueError("영상에서 다시 영상을 만들 수는 없어요. 원본 이미지를 고르세요.")
        if not self.registry.tool_ready("video_i2v", self.settings.models_dir):
            raise ValueError("영상 모델(Wan 2.2 I2V)이 없어요. update.bat을 실행해 받아 주세요 (약 31GB).")
        for iid in (req.end_image_id, req.mid_image_id):
            if iid and not self.library.get(iid):
                raise KeyError(f"이미지를 찾을 수 없어요: {iid}")
            if iid:
                self._need_still(iid)
        if req.mid_image_id and not req.end_image_id:
            raise ValueError("가운데 장면을 쓰려면 마지막 장면도 골라 주세요")
        w, h = (custom_size(src["width"], src["height"], req.width, req.height, 16) if req.width or req.height
                else video_size(src["width"], src["height"], req.quality))
        frames = req.seconds * VIDEO_FPS + 1
        if req.mid_image_id:
            frames = 2 * workflows.wan_half_length(frames) - 1
        smooth = req.smooth and self.registry.tool_ready("frame_interp", self.settings.models_dir)
        mult = req.smooth_multiplier if smooth else 1
        keys = " · 처음·가운데·끝" if req.mid_image_id else " · 처음·끝" if req.end_image_id else ""
        job = Job(id=uuid.uuid4().hex[:12], kind="video", request=req.model_dump(),
                  decision={"action": f"영상 {req.seconds}초{keys}", "model_name": "Wan 2.2 I2V 14B",
                            "family": "wan22_i2v", "width": w, "height": h, "frames": frames, "fps": VIDEO_FPS * mult,
                            "smooth": smooth, "multiplier": mult, "steps": req.steps})
        return job

    async def _run_video(self, job: Job) -> None:
        req = VideoRequest(**job.request)
        src = self.library.get(req.image_id)
        path = self.library.path(req.image_id)
        if not src or not path:
            raise ComfyError("이미지가 사라졌어요")
        d = job.decision
        motion = req.prompt.strip()
        base = src.get("prompt_en") or src.get("prompt") or ""
        text = motion or "natural, subtle movement, the camera holds steady"
        text = await self._maybe_translate(job, text, "wan22", None)
        if base and not motion:
            base = await self._maybe_translate(job, base, "wan22", None)
            text = f"{base}. {text}"
        w, h = d["width"], d["height"]
        name, poster = await self._upload_fit(src["id"], w, h, "video-start")
        end_name = (await self._upload_fit(req.end_image_id, w, h, "video-end"))[0] if req.end_image_id else None
        mid_name = (await self._upload_fit(req.mid_image_id, w, h, "video-mid"))[0] if req.mid_image_id else None
        await self._switch_family("wan22_i2v")
        job.message = "영상 만드는 중 (몇 분 걸려요)"
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
        files = video_files(self.registry.tools["video_i2v"])
        frames = req.seconds * VIDEO_FPS + 1                      # mid frame: two halves of this, sharing one frame
        neg = await self._maybe_translate(job, req.negative.strip(), "wan22", None) if req.negative.strip() else ""
        graph = workflows.image_to_video(name, files, text, neg, w, h, frames, seed, VIDEO_FPS,
                                         interp_file=self.registry.tool_file("frame_interp") if d.get("smooth") else None,
                                         interp_multiplier=d.get("multiplier") or 2, end_name=end_name,
                                         mid_name=mid_name, steps=req.steps)
        outs = await self.comfy.run(graph, self._progress(job, 0, 1), timeout=3 * 3600)
        item = next((o for o in outs if str(o.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov"))), outs[0])
        data = await self.comfy.view(item["filename"], item.get("subfolder", ""))
        ext = "." + item["filename"].rsplit(".", 1)[-1].lower()
        rec = self.library.add_video(data, poster, {
            "kind": "video", "source": "gen-studio", "parent_id": src["id"], "prompt": src.get("prompt", ""),
            "edit_prompt": req.prompt, "final_prompt": text, "model_id": None, "family": "wan22_i2v", "seed": seed,
            "label": f"영상 {req.seconds}초", "fps": d["fps"],
            "frames": (d["frames"] - 1) * (d.get("multiplier") or 1) + 1 if d.get("smooth") else d["frames"],
            "duration": round((d["frames"] - 1) / VIDEO_FPS, 2),
            "keyframes": [i for i in (req.mid_image_id, req.end_image_id) if i],
            "edit": {"mode": "video", "seconds": req.seconds, "quality": req.quality,
                     "request": {**req.model_dump(), "seed": -1}},
        }, ext=ext)
        job.result_ids.append(rec["id"])

    # ------------------------------------------------------------ poses -> video
    def submit_pose_video(self, req: PoseVideoRequest) -> Job:
        """Each pose becomes a still of the same person (자세 바꾸기), then the stills become keyframes of one clip."""
        src = self.library.get(req.image_id)
        if not src:
            raise KeyError("이미지를 찾을 수 없어요")
        self._need_still(req.image_id)
        for k in req.poses:
            if not self.library.get(k.pose_id):
                raise KeyError(f"자세를 찾을 수 없어요: {k.pose_id}")
            self._need_still(k.pose_id)
        self._reference_model(None)                       # Qwen redraws the person in each pose
        tool = "video_ltx" if req.engine == "ltx" else "video_i2v"
        if not self.registry.tool_ready(tool, self.settings.models_dir):
            raise ValueError("LTX-2.5 영상 모델이 없어요. 설정에서 받아 주세요." if req.engine == "ltx"
                             else "영상 모델(Wan 2.2 I2V)이 없어요. 설정에서 받아 주세요.")
        keys = sorted(req.poses, key=lambda k: float("inf") if k.at < 0 or k.at >= req.seconds else k.at)
        starts = req.keep_start and not any(k.at == 0 for k in keys)
        if req.engine == "wan" and len(keys) > (2 if starts else 3):
            raise ValueError("Wan은 처음·가운데·끝 장면까지만 돼요. 자세를 줄이거나 engine=ltx로 해 주세요.")
        if not starts and keys[0].at != 0 and req.engine == "wan":
            raise ValueError("Wan은 첫 장면이 필요해요: keep_start=true로 두거나 0초 자세를 넣어 주세요.")
        job = Job(id=uuid.uuid4().hex[:12], kind="posevideo", request=req.model_dump(),
                  decision={"action": f"자세 영상 · 자세 {len(keys)}개", "model_name": "Qwen 자세 바꾸기 → "
                            + ("LTX-2.5" if req.engine == "ltx" else "Wan 2.2"),
                            "keys": [k.model_dump() for k in keys], "keep_start": starts})
        return self._enqueue(job)

    async def _run_pose_video(self, job: Job) -> None:
        req = PoseVideoRequest(**job.request)
        keys = [PoseKey(**k) for k in job.decision["keys"]]
        stills = []
        for i, k in enumerate(keys):
            job.message = f"자세 장면 {i + 1}/{len(keys)} 만드는 중"
            job.progress = 0.6 * i / len(keys)
            sub = self._repose_job(ReposeRequest(image_id=req.image_id, pose=k.pose_id, prompt=req.prompt))
            await self._run_repose(sub)
            stills.append(sub.result_ids[0])
        job.message, job.progress = "영상 만드는 중", 0.6
        timed = list(zip(stills, keys))
        first = req.image_id if job.decision["keep_start"] else timed.pop(0)[0]
        if req.engine == "ltx":
            end = next((sid for sid, k in timed if k.at < 0 or k.at >= req.seconds), None)
            mids = [{"image_id": sid, "at": k.at} for sid, k in timed if sid != end]
            sub = self._ltx_job(LtxRequest(prompt=req.prompt or "natural movement", image_id=first, end_image_id=end,
                                           keyframes=mids, seconds=req.seconds,
                                           quality="540p" if req.quality in ("480p", "540p") else "720p",
                                           fps=req.fps, audio=req.audio, image_strength=req.image_strength,
                                           negative=req.negative, seed=req.seed))
            await self._run_ltx(sub)
        else:
            end = timed[-1][0] if timed else None
            mid = timed[0][0] if len(timed) == 2 else None
            sub = self._video_job(VideoRequest(image_id=first, prompt=req.prompt, end_image_id=end, mid_image_id=mid,
                                               seconds=min(req.seconds, 10),
                                               quality="480p" if req.quality == "480p" else "720p",
                                               smooth=req.smooth, negative=req.negative, seed=req.seed))
            await self._run_video(sub)
        job.result_ids += sub.result_ids + stills
        job.result = {"video_id": sub.result_ids[0], "stills": stills}

    # ------------------------------------------------------------ clips: take a frame out, join clips
    def _need_clip(self, vid: str) -> dict:
        rec = self.library.get(vid)
        if not rec:
            raise KeyError(f"영상을 찾을 수 없어요: {vid}")
        if rec.get("media") != "video":
            raise ValueError(f"영상이 아니에요: {vid}")
        return rec

    def _clip_info(self, rec: dict) -> dict:
        """Frames, fps and sound of the saved file itself (what a player shows); the record as a fallback."""
        from .mp4info import probe_file
        path = self.library.path(rec["id"])
        info = probe_file(path) if path else None
        if info:
            return info
        fps = float(rec.get("fps") or VIDEO_FPS)
        frames = int(rec.get("frames") or round((rec.get("duration") or 1) * fps) + 1)
        return {"frames": frames, "fps": fps, "duration": round(frames / fps, 3), "audio": bool(rec.get("audio"))}

    async def _upload_clip(self, rec: dict) -> str:
        path = self.library.path(rec["id"])
        if not path:
            raise ComfyError(f"영상 파일이 사라졌어요: {rec['id']}")
        # LoadVideo only lists files directly in ComfyUI's input folder
        return await self.comfy.upload_image(path.read_bytes(), f"gs-clip-{rec['id']}{path.suffix}", subfolder="")

    def submit_video_frame(self, req: VideoFrameRequest) -> Job:
        rec = self._need_clip(req.video_id)
        info = self._clip_info(rec)
        fps, frames = info["fps"], info["frames"]
        if req.at == "first":
            idx = 0
        elif req.at == "last" or float(req.at) < 0:
            idx = -1
        else:
            idx = round(float(req.at) * fps)
            if frames and idx >= frames:
                idx = -1
        where = "첫" if idx == 0 else "마지막" if idx == -1 else f"{idx / fps:.1f}초"
        job = Job(id=uuid.uuid4().hex[:12], kind="frame", request=req.model_dump(),
                  decision={"action": f"영상의 {where} 장면 꺼내기", "model_name": "-", "index": idx})
        return self._enqueue(job)

    async def _run_video_frame(self, job: Job) -> None:
        req = VideoFrameRequest(**job.request)
        rec = self._need_clip(req.video_id)
        idx = job.decision["index"]
        if idx == -1:
            idx = max(0, self._clip_info(rec)["frames"] - 1)
        name = await self._upload_clip(rec)
        outs = await self.comfy.run(workflows.video_frame(name, idx), self._progress(job, 0, 1))
        data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
        new = self.library.add(data, {
            "kind": "edit", "source": "gen-studio", "parent_id": rec["id"], "prompt": rec.get("prompt", ""),
            "final_prompt": rec.get("final_prompt", ""), "model_id": None, "family": rec.get("family"),
            "label": job.decision["action"].removesuffix(" 꺼내기"), "edit": {"mode": "frame", "index": idx},
        })
        job.result_ids.append(new["id"])

    def submit_join(self, req: JoinVideosRequest) -> Job:
        recs = [self._need_clip(v) for v in req.video_ids]
        infos = [self._clip_info(r) for r in recs]
        audio = any(i["audio"] for i in infos) if req.audio is None else req.audio
        fps = max(i["fps"] for i in infos)
        frames = sum(len(workflows.retime_indices(i["frames"], i["fps"], fps)) for i in infos)
        job = Job(id=uuid.uuid4().hex[:12], kind="join", request=req.model_dump(),
                  decision={"action": f"영상 {len(recs)}개 이어 붙이기", "model_name": "-", "fps": fps,
                            "audio": audio, "width": recs[0]["width"], "height": recs[0]["height"],
                            "frames": frames, "duration": round(frames / fps, 3),
                            "clips": [{"id": r["id"], **i} for r, i in zip(recs, infos)]})
        return self._enqueue(job)

    async def _run_join(self, job: Job) -> None:
        req = JoinVideosRequest(**job.request)
        d = job.decision
        recs = [self._need_clip(v) for v in req.video_ids]
        clips = [{**c, "name": await self._upload_clip(r)} for r, c in zip(recs, d["clips"])]
        graph, plan = workflows.join_videos(clips, d["audio"])
        outs = await self.comfy.run(graph, self._progress(job, 0, 1), timeout=3600)
        clip = next((o for o in outs if str(o.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov"))), None)
        poster_item = next((o for o in outs if str(o.get("filename", "")).lower().endswith(".png")), None)
        if not clip:
            raise ComfyError("영상 파일이 저장되지 않았어요")
        data = await self.comfy.view(clip["filename"], clip.get("subfolder", ""))
        if poster_item:
            poster = await self.comfy.view(poster_item["filename"], poster_item.get("subfolder", ""))
        else:
            buf = io.BytesIO()
            Image.new("RGB", (d["width"], d["height"]), (20, 20, 20)).save(buf, "PNG")
            poster = buf.getvalue()
        first = recs[0]
        rec = self.library.add_video(data, poster, {
            "kind": "video", "source": "gen-studio", "parent_id": first["id"], "prompt": first.get("prompt", ""),
            "edit_prompt": "", "final_prompt": first.get("final_prompt", ""), "model_id": None,
            "family": first.get("family"), "label": f"이어 붙인 영상 ({len(recs)}개)", "fps": plan["fps"],
            "frames": plan["frames"], "duration": plan["duration"],
            "audio": d["audio"], "joined": req.video_ids, "edit": {"mode": "join", "video_ids": req.video_ids},
        }, ext="." + clip["filename"].rsplit(".", 1)[-1].lower())
        job.result_ids.append(rec["id"])

    # ------------------------------------------------------------ upscale
    def submit_upscale(self, req: UpscaleRequest) -> Job:
        self._need_still(req.image_id)
        src = self.library.get(req.image_id)
        if not src:
            raise KeyError("이미지를 찾을 수 없어요")
        if not self.registry.tool_ready("upscale", self.settings.models_dir):
            raise ValueError("업스케일 모델(Real-ESRGAN)이 없어요. update.bat을 실행해 받아 주세요.")
        m = None
        mp = src["width"] * src["height"] / 1_000_000
        big = mp > 1.5                          # already upscaled: keep its size, just Real-ESRGAN on top
        if req.refine is not False and not big:
            style = "semi" if src.get("style") == "semi" or src.get("family") == "illustrious" else "realistic"
            m = self.helper_model(style)
        w, h = src["width"], src["height"]
        k = (1.0 if big else (1_000_000 / (w * h)) ** 0.5) * req.scale
        decision = {"action": f"업스케일 {req.scale}배", "model_name": f"Real-ESRGAN{' + ' + m.name if m else ''}",
                    "model_id": m.id if m else None, "family": m.family.id if m else None,
                    "width": round(w * k), "height": round(h * k), "megapixels": round(mp, 2) if big else 1.0,
                    "denoise": req.denoise or (0.25 if m and m.family.id == "illustrious" else 0.33)}
        job = Job(id=uuid.uuid4().hex[:12], kind="upscale", request=req.model_dump(), decision=decision)
        return self._enqueue(job)

    async def _run_upscale(self, job: Job) -> None:
        req = UpscaleRequest(**job.request)
        src = self.library.get(req.image_id)
        path = self.library.path(req.image_id)
        if not src or not path:
            raise ComfyError("이미지가 사라졌어요")
        d = job.decision
        model = self.registry.get(d["model_id"]) if d.get("model_id") else None
        prompt = req.prompt.strip() or src.get("prompt_en") or src.get("prompt") or ""
        if model:
            prompt = await self._maybe_translate(job, prompt, model.family.id, None)
            await self._switch_family(model.family.id)
        name = await self.comfy.upload_image(path.read_bytes(), f"{src['id']}.png")
        job.message = "키우는 중"
        graph, extra = workflows.upscale(name, self.registry.tool_file("upscale"), req.scale, model, prompt,
                                         d["denoise"], random.randint(0, 2**31 - 1), self.settings.models_dir,
                                         megapixels=d.get("megapixels", 1.0))
        outs = await self.comfy.run(graph, self._progress(job, 0, 1), timeout=3600)
        data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
        rec = self.library.add(data, {
            "kind": "edit", "source": "gen-studio", "parent_id": src["id"], "prompt": src.get("prompt", ""),
            "model_id": src.get("model_id"), "family": src.get("family"), "style": src.get("style"),
            "label": f"업스케일 {req.scale}배", "edit_prompt": req.prompt, "final_prompt": extra["final_prompt"],
            "edit": {"mode": "upscale", "scale": req.scale, "refine": model is not None, "denoise": d["denoise"]},
        })
        job.result_ids.append(rec["id"])

    # ------------------------------------------------------------ detail pass
    def helper_model(self, style: str):
        """A model that can redraw parts with a regular sampler (Z-Image / Illustrious), for detail passes, upscale
        refinement, pose ControlNet and training when the main model is Qwen. None when none is installed."""
        md = self.settings.models_dir
        cands = [m for m in self.registry.models if m.covers(style) and m.family.encoder != "qwen21"
                 and not m.family.high_noise and m.installed(md) is not False]
        if not cands:
            cands = [m for m in self.registry.models if m.family.encoder != "qwen21" and not m.family.high_noise
                     and m.installed(md) is not False and (style != "semi") == (m.family.id == "zimage")]
        cands.sort(key=lambda m: {"helper": 0, "main": 1}.get(m.role, 2))
        return cands[0] if cands else None

    def _detail_model(self, src: dict, requested: str):
        if requested and requested != "auto":
            m = self.registry.get(requested)
            if m.family.encoder == "qwen21" or m.family.high_noise:
                raise ValueError(f"{m.name}은(는) 디테일 보정에 쓰지 않아요. Z-Image나 Illustrious를 고르세요.")
            return m
        m, _why = router.model_for_edit(self.registry, src.get("model_id"), self.settings.models_dir,
                                        self.settings.default_style)
        if m.family.encoder == "qwen21" or m.family.high_noise:   # Qwen / Wan pictures: redraw with a helper model
            style = "semi" if src.get("style") == "semi" else "realistic"
            m = self.helper_model(style)
            if m is None:
                raise ValueError("디테일 보정에 쓸 Z-Image·Illustrious 모델이 없어요. 설정에서 모델을 받아 주세요.")
        return m

    def submit_detail(self, req) -> Job:
        self._need_still(req.image_id)
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

    # ------------------------------------------------------------ change the pose of the person in a picture
    def submit_repose(self, req: ReposeRequest) -> Job:
        return self._enqueue(self._repose_job(req))

    def _repose_job(self, req: ReposeRequest) -> Job:
        src = self.library.get(req.image_id)
        if not src:
            raise KeyError("이미지를 찾을 수 없어요")
        self._need_still(req.image_id)
        if not req.pose and not req.prompt.strip():
            raise ValueError("자세 사진을 넣거나, 아래 입력칸에 어떤 자세로 바꿀지 적어 주세요")
        pose = None
        if req.pose:
            rec = self.library.get(req.pose)
            if not rec:
                raise KeyError(f"자세 사진을 찾을 수 없어요: {req.pose}")
            self._need_still(req.pose)
            is_map = rec.get("kind") == "pose"
            if not is_map and not self.registry.tool_ready("pose_detect", self.settings.models_dir):
                raise ValueError("포즈 인식 모델(SDPose)이 없어요. 설정에서 모델을 받아 주세요.")
            pose = {"image": req.pose, "is_map": is_map}
        m = self._reference_model(None)
        job = Job(id=uuid.uuid4().hex[:12], kind="repose", request=req.model_dump(),
                  decision={"action": "자세 바꾸기", "model_id": m.id, "model_name": m.name, "family": m.family.id,
                            "pose": pose})
        return job

    async def _run_repose(self, job: Job) -> None:
        req = ReposeRequest(**job.request)
        src = self.library.get(req.image_id)
        path = self.library.path(req.image_id)
        if not src or not path:
            raise ComfyError("이미지가 사라졌어요")
        model = self.registry.get(job.decision["model_id"])
        names = [await self.comfy.upload_image(path.read_bytes(), f"{src['id']}.png")]
        extra = req.prompt.strip()
        if job.decision.get("pose"):
            pose_id, pose_png = await self._pose_map(job, job.decision["pose"])
            pose_png = fit_on_black(pose_png, src["width"], src["height"])
            names.append(await self.comfy.upload_image(pose_png, f"repose-{pose_id}-{src['width']}.png"))
            instruction = ("Redraw the person from image 1 in exactly the pose shown in image 2 (a skeleton drawing: "
                           "follow the positions of the head, arms, hands, legs and feet; do not draw the skeleton). "
                           + (f"{extra}. " if extra else "") + REPOSE_KEEP)
        else:
            instruction = f"Change the pose of the person in image 1: {extra}. " + REPOSE_KEEP
        await self._switch_family(model.family.id)
        job.message = "자세 바꾸는 중"
        seed = random.randint(0, 2**31 - 1)
        graph, info = workflows.reference(model, names, instruction, "", seed, models_dir=self.settings.models_dir)
        outs = await self.comfy.run(graph, self._progress(job, 0.1, 0.9))
        data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
        rec = self.library.add(data, {
            "kind": "edit", "source": "gen-studio", "parent_id": src["id"], "prompt": src.get("prompt", ""),
            "edit_prompt": req.prompt, "final_prompt": info["final_prompt"], "model_id": model.id,
            "family": model.family.id, "style": src.get("style"), "seed": seed, "label": "자세 바꾸기",
            "edit": {"mode": "repose", "pose": req.pose},
        })
        job.result_ids.append(rec["id"])

    def submit_cutout(self, req: CutoutRequest) -> Job:
        self._need_still(req.image_id)
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
        # running: stop ComfyUI's current graph and the job itself (uploads, translation, the next graph ...)
        self._cancel.add(job_id)
        job.message = "취소하는 중"
        try:
            await self.comfy.interrupt()
        except Exception:  # noqa: BLE001 - ComfyUI gone: the job stops anyway
            pass
        if self._task and self._running == job_id and not self._task.done():
            self._task.cancel()
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
                self._task = asyncio.create_task(self._run_job(job))
                await self._task
                job.status, job.progress, job.message = "done", 1.0, "완료"
            except asyncio.CancelledError:
                if jid not in self._cancel:
                    raise                                  # the worker itself is stopping
                job.status, job.message = "canceled", "취소됨"
            except ComfyError as e:
                if jid in self._cancel:
                    job.status, job.message = "canceled", "취소됨"
                else:
                    job.status, job.error, job.message = "error", str(e), "실패"
                    log.warning("작업 실패 %s (%s): %s", job.id, job.kind, e)
            except Exception as e:  # keep the worker alive no matter what
                job.status, job.error, job.message = "error", f"{type(e).__name__}: {e}", "실패"
                log.exception("작업 오류 %s (%s)", job.id, job.kind)
            finally:
                self._running = None
                self._task = None
                self._cancel.discard(jid)
                self._finish(job)

    async def _run_job(self, job: Job) -> None:
        """One job from start to end; runs as its own task so cancel() can stop it anywhere."""
        await self._ease_memory(job)
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
        elif job.kind == "upscale":
            await self._run_upscale(job)
        elif job.kind == "video":
            await self._run_video(job)
        elif job.kind == "ltx":
            await self._run_ltx(job)
        elif job.kind == "frame":
            await self._run_video_frame(job)
        elif job.kind == "readpose":
            await self._run_read_pose(job)
        elif job.kind == "posevideo":
            await self._run_pose_video(job)
        elif job.kind == "join":
            await self._run_join(job)
        elif job.kind == "repose":
            await self._run_repose(job)
        else:
            await self._run_edit(job)

    def _finish(self, job: Job) -> None:
        job.finished_at = time.time()
        ev = self._done_events.get(job.id)
        if ev:
            ev.set()

    async def _ease_memory(self, job: Job) -> None:
        """With 32GB of RAM, models ComfyUI keeps around (Qwen 7GB + its 9GB text encoder, SAM 3, Z-Image, Wan ...)
        fill memory and Windows starts swapping to disk: steps go from ~2s to over a minute. Unload them first."""
        try:
            import psutil
            vm = psutil.virtual_memory()
        except Exception:  # noqa: BLE001
            return
        if vm.available < max(8 * 1024**3, vm.total * 0.25):
            job.message = "메모리 비우는 중"
            log.warning("RAM 여유 %.1fGB: 모델을 내려놓고 시작", vm.available / 1024**3)
            await self.comfy.free()
            self._last_family = None

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
        pose = job.decision.get("pose")
        control, pose_id = None, None
        if pose:
            pose_id, pose_png = await self._pose_map(job, pose)
            pose_png = fit_on_black(pose_png, job.decision["width"], job.decision["height"])
            pose_name = await self.comfy.upload_image(pose_png, f"pose-{pose_id}-{job.decision['width']}.png")
            if pose["kind"] == "qwen":
                ref_names = [*ref_names, pose_name]
                prompt += (f" The person or people take exactly the pose shown in image {len(ref_names)} "
                           "(a skeleton drawing; do not draw the skeleton itself).")
            else:
                control = {"kind": pose["kind"], "image": pose_name, "file": pose["file"], "strength": pose["strength"]}
        regions = []
        for r in req.regions:
            box = r.clamp()
            regions.append((box, await self._maybe_translate(job, r.prompt.strip(), model.family.id, req.translate)))
        base_prompt = prompt
        if regions:
            # first pass already knows the layout; then each box is redrawn with its own prompt
            sep = ", " if model.family.id == "illustrious" else ". "
            prompt = prompt + sep + sep.join(f"{region_position(b)}: {t}" for b, t in regions)
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
                    models_dir=self.settings.models_dir, steps=req.steps, cfg=req.cfg, loras=loras, control=control)
            span = 1 / n / (1 + len(regions))
            outs = await self.comfy.run(graph, self._progress(job, i / n, span))
            data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
            for k, (box, text) in enumerate(regions, 1):
                job.message = f"영역 {k}/{len(regions)} 그리는 중"
                data = await self._draw_region(job, model, data, box, text, base_prompt, negative, seed + k, loras,
                                               self._progress(job, i / n + span * k, span))
            rec = self.library.add(data, {
                "regions": [{"box": list(b), "prompt": r.prompt} for (b, _t), r in zip(regions, req.regions)] or None,
                "pose_id": pose_id, "pose_strength": pose["strength"] if pose else None,
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

    async def _pose_map(self, job: Job, pose: dict, need_joints: bool = False) -> tuple[str, bytes]:
        """The skeleton picture: as given, or made from the photo once and kept in the library for reuse."""
        rec = self.library.get(pose["image"])
        if pose["is_map"]:
            return rec["id"], self.library.path(rec["id"]).read_bytes()
        done = next((r for r in self.library.list(limit=100000) if r.get("kind") == "pose" and r.get("parent_id") == rec["id"]
                     and (r.get("keypoints") or not need_joints) and not r.get("pose_name")), None)
        if done:
            return done["id"], self.library.path(done["id"]).read_bytes()
        job.message = "포즈 읽는 중"
        with Image.open(self.library.path(rec["id"])) as im:
            im = im.convert("RGB")
            im.thumbnail((1024, 1024), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "PNG")
        name = await self.comfy.upload_image(buf.getvalue(), f"posesrc-{rec['id']}.png")
        files = self.registry.tools["pose_detect"]
        outputs = await self.comfy.run_outputs(workflows.pose_map(name, files[0].name, files[1].name))
        imgs = [d for _n, d in iter_images(outputs)]
        if not imgs:
            raise ComfyError("포즈 그림이 저장되지 않았어요")
        data = await self.comfy.view(imgs[0]["filename"], imgs[0].get("subfolder", ""))
        joints = poses.from_sdpose(output_text(outputs))
        prec = self.library.add(data, {"kind": "pose", "source": "gen-studio", "parent_id": rec["id"],
                                       "label": "포즈", "prompt": rec.get("prompt", ""),
                                       **({"keypoints": joints} if joints else {})})
        return prec["id"], data

    # ------------------------------------------------------------ skeleton editor
    def save_pose(self, req: PoseSaveRequest) -> dict:
        """A pose made by moving joints (editor or Codex): drawn as a pose map and kept in the library."""
        people = poses.clean(req.people)
        if req.parent_id and not self.library.get(req.parent_id):
            raise KeyError(f"이미지를 찾을 수 없어요: {req.parent_id}")
        png = poses.render(people, req.width, req.height)
        name = req.name.strip()
        kp = {"people": people}
        if req.edit is not None:
            if len(json.dumps(req.edit)) > 200_000:
                raise ValueError("편집 정보가 너무 커요")
            kp["edit"] = req.edit
        return self.library.add(png, {"kind": "pose", "source": "gen-studio", "parent_id": req.parent_id,
                                      "label": f"자세: {name}" if name else "뼈대 자세", "prompt": "",
                                      "pose_name": name or None, "keypoints": kp})

    def saved_poses(self) -> list[dict]:
        return [r for r in self.library.list(limit=100000) if r.get("kind") == "pose" and r.get("pose_name")]

    def submit_read_pose(self, image_id: str) -> Job:
        """Read the joints of every person in a picture (to start the editor from a photo)."""
        rec = self.library.get(image_id)
        if not rec:
            raise KeyError("이미지를 찾을 수 없어요")
        self._need_still(image_id)
        is_map = rec.get("kind") == "pose"
        if is_map and not rec.get("keypoints"):
            raise ValueError("이 포즈 그림에는 관절 정보가 없어요. 원래 사진에서 읽어 주세요.")
        if not is_map and not self.registry.tool_ready("pose_detect", self.settings.models_dir):
            raise ValueError("포즈 인식 모델(SDPose)이 없어요. 설정에서 모델을 받아 주세요.")
        job = Job(id=uuid.uuid4().hex[:12], kind="readpose", request={"image_id": image_id},
                  decision={"action": "뼈대 읽기", "model_name": "SDPose", "is_map": is_map})
        return self._enqueue(job)

    async def _run_read_pose(self, job: Job) -> None:
        iid = job.request["image_id"]
        pid, _ = await self._pose_map(job, {"image": iid, "is_map": job.decision["is_map"]}, need_joints=True)
        prec = self.library.get(pid)
        if not prec.get("keypoints"):
            raise ComfyError("사진에서 사람 뼈대를 찾지 못했어요. 사람이 크게 나온 사진으로 해 보세요.")
        job.result_ids.append(pid)
        job.result = {"pose_id": pid, "keypoints": prec["keypoints"]}

    async def _draw_region(self, job: Job, model, png: bytes, box, text: str, base_prompt: str, negative: str,
                           seed: int, loras: list[dict], progress) -> bytes:
        with Image.open(io.BytesIO(png)) as im:
            w, h = im.size
        img_name = await self.comfy.upload_image(png, f"region-{job.id}.png")
        mask_name = await self.comfy.upload_image(region_mask(box, w, h), f"region-{job.id}-mask.png")
        lw = {"loras": loras} if loras else {}
        if model.family.encoder == "qwen21":
            instruction = (f"In the white area of image 2 ({region_position(box)} of image 1), draw {text}. Make it fit the "
                           "scene naturally: same lighting, perspective and style. Image 2 is a mask: change only its "
                           "white area and keep everything else in image 1 exactly the same.")
            graph, _ = workflows.masked_reference(model, img_name, mask_name, instruction, negative, seed, grow=0,
                                                  models_dir=self.settings.models_dir, **lw)
        else:
            sep = ", " if model.family.id == "illustrious" else ". "
            graph, _ = workflows.inpaint(model, img_name, mask_name, text + sep + base_prompt, negative, seed,
                                         denoise=0.75, grow=0, models_dir=self.settings.models_dir, **lw)
        outs = await self.comfy.run(graph, progress)
        return await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))

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
                ask = req0.prompt
                if req.fill == "prompt" and req0.prompt.strip():
                    ask = await self._edit_english(job, req0.prompt.strip())
                    if ask != req0.prompt:
                        req = req.model_copy(update={"prompt": ask})
                graph, extra = workflows.masked_reference(
                    model, image_name, mask_name, qwen_mask_instruction(ask, req.fill, names), req.negative, seed,
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
        if (req.mode == "inpaint" and req.fill == "prompt" and model.family.encoder == "qwen21"
                and masked_change(src_path.read_bytes(), data, mask_bytes) < UNCHANGED):
            # Qwen sometimes hands the selected area back untouched; redraw it once with the helper model
            helper = self.helper_model("semi" if src.get("style") == "semi" else "realistic")
            if helper is not None:
                job.message = f"그대로 나와서 {helper.name}(으)로 다시 그리는 중"
                log.warning("편집 %s: 선택 영역이 거의 안 바뀜 → %s로 다시 그림", job.id, helper.id)
                await self._switch_family(helper.family.id)
                text = await self._result_prompt(job, req0.prompt.strip(), helper.family.id)
                graph, extra = workflows.inpaint(helper, image_name, mask_name, ", ".join(x for x in (text, base_prompt) if x),
                                                 req.negative, seed, denoise=0.9,
                                                 grow=mask_grow(req0.prompt, src["width"], src["height"]),
                                                 models_dir=self.settings.models_dir)
                outs = await self.comfy.run(graph, self._progress(job, 0.5, 0.5))
                data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
                model = helper
                job.result["fallback"] = helper.name
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

    async def enhance_edit(self, text: str, image_id: str | None, segments: list[str]) -> dict:
        """Short edit request -> detailed English edit instruction (shown to the user to edit before running)."""
        from . import translate as tr
        vlm = self.vlm_file()
        if not vlm:
            raise ValueError("편집 요청 다듬기에는 Qwen3-VL이 필요해요")
        names = self._segment_names(image_id, segments) if image_id and segments else []
        out = await self.comfy.run_outputs(workflows.rewrite_text(vlm, tr.edit_enhance_instruction(text, names),
                                                                  max_length=200), timeout=180)
        return {"prompt": tr.clean(output_text(out), text), "segments": names}

    async def enhance_prompt(self, req: GenerateRequest) -> dict:
        """Short request -> detailed prompt in the form the chosen model reads best (shown to the user to edit)."""
        from . import translate as tr
        d = self.plan(req)
        vlm = self.vlm_file()
        if not vlm:
            raise ValueError("프롬프트 다듬기에는 Qwen3-VL(Qwen-Image 2.1과 함께 받는 파일)이 필요해요")
        prompt, neg = router.split_negative(req.prompt)
        outputs = await self.comfy.run_outputs(workflows.rewrite_text(vlm, tr.enhance_instruction(prompt, d["family"]),
                                                                       max_length=350), timeout=180)
        text = tr.clean(output_text(outputs), prompt)
        if neg:
            text += f" --no {neg}"
        return {"prompt": text, "model_name": d["model_name"], "model_id": d["model_id"], "family": d["family"],
                "style": d["style"], "aspect": d.get("aspect")}

    async def _edit_english(self, job: Job, text: str) -> str:
        """Korean edit request -> one plain English instruction (Qwen follows that more reliably)."""
        from . import translate as tr
        vlm = self.vlm_file()
        if not tr.has_korean(text) or not vlm:
            return text
        job.message = "요청을 편집 지시로 바꾸는 중"
        try:
            out = await self.comfy.run_outputs(workflows.rewrite_text(vlm, tr.edit_instruction(text), max_length=120))
        except ComfyError:
            return text
        return tr.clean(output_text(out), text)

    async def _result_prompt(self, job: Job, text: str, family: str) -> str:
        """For redraw models: describe what the area should show (a prompt), not the command."""
        from . import translate as tr
        vlm = self.vlm_file()
        if not text or not vlm:
            return text
        try:
            out = await self.comfy.run_outputs(workflows.rewrite_text(vlm, tr.instruction(text, family)))
        except ComfyError:
            return text
        return tr.clean(output_text(out), text)

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
                character=rec.get("character"), loras=rec.get("loras") or [], denoise=rec.get("denoise"),
                regions=rec.get("regions") or [], pose=rec.get("pose_id"), pose_strength=rec.get("pose_strength") or 0.85))
        e = rec.get("edit") or {}
        if e.get("mode") in ("join", "frame"):
            raise ValueError("이어 붙인 영상이나 꺼낸 장면은 같은 결과라서 다시 만들 필요가 없어요")
        if rec.get("kind") == "video" and e.get("request"):
            r = {**e["request"], "seed": -1}
            return self.submit_ltx(LtxRequest(**r)) if e.get("mode") == "ltx" else self.submit_video(VideoRequest(**r))
        if rec.get("kind") == "video" and (rec.get("edit") or {}).get("mode") == "ltx":
            e = rec["edit"]
            return self.submit_ltx(LtxRequest(prompt=rec.get("prompt") or " ", image_id=rec.get("parent_id"),
                                              seconds=e.get("seconds") or 5, quality=e.get("quality") or "720p",
                                              orientation=e.get("orientation") or "auto", negative=e.get("negative") or ""))
        if rec.get("kind") == "video" and rec.get("parent_id"):
            e = rec.get("edit") or {}
            return self.submit_video(VideoRequest(image_id=rec["parent_id"], prompt=rec.get("edit_prompt") or "",
                                                  seconds=e.get("seconds") or 5, quality=e.get("quality") or "480p"))
        if rec.get("kind") == "edit" and rec.get("parent_id"):
            edit = rec.get("edit") or {}
            if edit.get("mode") == "character_sheet":
                raise ValueError("캐릭터 사진은 캐릭터 만들기로 다시 만들어 주세요")
            if edit.get("mode") == "repose":
                return self.submit_repose(ReposeRequest(image_id=rec["parent_id"], pose=edit.get("pose"),
                                                        prompt=rec.get("edit_prompt") or ""))
            if edit.get("mode") == "upscale":
                return self.submit_upscale(UpscaleRequest(image_id=rec["parent_id"], scale=edit.get("scale") or 2,
                                                          refine=edit.get("refine"), denoise=edit.get("denoise"),
                                                          prompt=rec.get("edit_prompt") or ""))
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
