"""LoRA training with ComfyUI's built-in training nodes (TrainLoraNode, experimental in ComfyUI).

Flow: pictures from the library -> resized to training buckets -> each captioned by Qwen3-VL with a trigger word ->
uploaded to ComfyUI's input folder -> one graph (load dataset, encode, bucket, train, save) -> the .safetensors is
fetched back, put in models/loras and registered in the LoRA list with its family and trigger.
"""
from __future__ import annotations

import io
import re
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from .registry import ModelSpec
from .workflows import Graph, _loaders, out

# Families that fit training in 16GB. Qwen-Image (20B) and Wan 2.2 (2x14B) do not.
TRAINABLE = {
    "illustrious": {"lr": 1e-4, "offload": False, "per_image": 100, "name": "반실사 (Illustrious)"},
    "zimage": {"lr": 1e-4, "offload": True, "per_image": 80, "name": "실사 (Z-Image)"},
}
BUCKETS = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (768, 1344), (1344, 768)]
MIN_IMAGES, MAX_IMAGES = 3, 60


class TrainRequest(BaseModel):
    name: str = Field(..., min_length=1, description="LoRA 이름")
    image_ids: list[str] = Field(default_factory=list, description="학습할 이미지 id")
    character: str | None = Field(None, description="이 캐릭터의 사진들로 학습")
    kind: Literal["character", "style"] = "character"
    family: Literal["auto", "illustrious", "zimage"] = "auto"
    trigger: str | None = Field(None, description="이 단어를 프롬프트에 넣으면 LoRA가 켜져요 (비우면 자동)")
    steps: int | None = Field(None, ge=50, le=6000)
    rank: int = Field(16, ge=4, le=64)
    learning_rate: float | None = Field(None, gt=0, le=0.01)


def bucket_for(w: int, h: int) -> tuple[int, int]:
    r = w / h
    return min(BUCKETS, key=lambda b: abs(b[0] / b[1] - r))


def prepare_image(data: bytes) -> bytes:
    """Center-crop and resize to the nearest training bucket (same-size images train in one batch shape)."""
    with Image.open(io.BytesIO(data)) as im:
        rgba = im.convert("RGBA")
        base = Image.new("RGB", rgba.size, (255, 255, 255))
        base.paste(rgba, mask=rgba.split()[3])
        bw, bh = bucket_for(*base.size)
        s = max(bw / base.width, bh / base.height)
        rw, rh = max(bw, round(base.width * s)), max(bh, round(base.height * s))
        base = base.resize((rw, rh), Image.LANCZOS)
        x, y = (rw - bw) // 2, (rh - bh) // 2
        base = base.crop((x, y, x + bw, y + bh))
        buf = io.BytesIO()
        base.save(buf, "PNG")
        return buf.getvalue()


def make_trigger(name: str) -> str:
    """A rare token so the LoRA only switches on when asked: 'Flight Director' -> 'gs_flightdirector'."""
    ascii_part = re.sub(r"[^a-z0-9]+", "", name.lower())
    return "gs_" + (ascii_part[:16] or format(abs(hash(name)) % 10**6, "06d"))


def caption_instruction(kind: str, family: str, trigger: str) -> str:
    form = ("comma-separated English Danbooru-style tags" if family == "illustrious"
            else "one or two plain English sentences")
    if kind == "character":
        skip = (f"Do NOT describe the person's face, eyes, hair or body type: those are learned as \"{trigger}\". "
                "Describe only what changes between pictures: pose, expression, clothing, place, lighting, framing.")
    else:
        skip = (f"Do NOT describe the art style, colours or rendering: that is learned as \"{trigger}\". "
                "Describe only the content: subjects, clothing, pose, place, framing.")
    return f"Write a training caption for image 1 as {form}. {skip} Reply with the caption only."


def default_steps(family: str, n_images: int) -> int:
    return max(600, min(2400, TRAINABLE[family]["per_image"] * n_images))


def train_graph(model: ModelSpec, folder: str, steps: int, rank: int, lr: float, seed: int, prefix: str,
                models_dir=None) -> dict:
    g = Graph()
    ms, c, v = _loaders(g, model, models_dir)
    ds = g.add("LoadImageTextDataSetFromFolder", folder=folder)
    mk = g.add("MakeTrainingDataset", images=out(ds, 0), vae=v, clip=c, texts=out(ds, 1))
    bk = g.add("ResolutionBucket", latents=out(mk, 0), conditioning=out(mk, 1))
    tr = g.add("TrainLoraNode", model=ms["main"], latents=out(bk, 0), positive=out(bk, 1), batch_size=1,
               grad_accumulation_steps=1, steps=int(steps), learning_rate=float(lr), rank=int(rank), optimizer="AdamW",
               loss_function="MSE", seed=int(seed), training_dtype="bf16", lora_dtype="bf16",
               quantized_backward=False, algorithm="LoRA", gradient_checkpointing=True, checkpoint_depth=1,
               offloading=bool(TRAINABLE[model.family.id]["offload"]), existing_lora="[None]", bucket_mode=True,
               bypass_mode=False)
    g.add("SaveLoRA", lora=out(tr, 0), prefix=prefix, steps=out(tr, 2))
    return g.nodes


TRAIN_NODES = ["LoadImageTextDataSetFromFolder", "MakeTrainingDataset", "ResolutionBucket", "TrainLoraNode", "SaveLoRA"]
