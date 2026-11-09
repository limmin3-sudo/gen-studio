"""Builds ComfyUI API-format graphs for each model family.

Graphs are built in code instead of templated JSON so every family shares the
same three shapes: text-to-image, image-to-image and masked inpaint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import ModelSpec


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self._n = 0

    def add(self, class_type: str, **inputs: Any) -> str:
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid


def out(nid: str, idx: int = 0) -> list:
    return [nid, idx]


def _loaders(g: Graph, model: ModelSpec, models_dir: Path | None):
    fam = model.family
    kind = model.loader_kind(models_dir)
    if kind == "unet":
        unet = g.add("UNETLoader", unet_name=model.file.name, weight_dtype="default")
        clip = g.add("CLIPLoader", clip_name=fam.clip, type=fam.clip_type, device="default")
        vae = g.add("VAELoader", vae_name=fam.vae)
        m, c, v = out(unet), out(clip), out(vae)
    else:
        ck = g.add("CheckpointLoaderSimple", ckpt_name=model.file.name)
        m, c, v = out(ck, 0), out(ck, 1), out(ck, 2)
    if fam.shift is not None:
        m = out(g.add("ModelSamplingAuraFlow", model=m, shift=fam.shift))
    if fam.clip_skip:
        c = out(g.add("CLIPSetLastLayer", clip=c, stop_at_clip_layer=-abs(int(fam.clip_skip))))
    return m, c, v


def _conditioning(g: Graph, model: ModelSpec, clip, prompt: str, negative: str):
    fam = model.family
    text = (model.prefix + prompt).strip() if prompt else model.prefix.strip(", ").strip()
    pos = out(g.add("CLIPTextEncode", text=text, clip=clip))
    if fam.negative:
        neg_text = ", ".join(x for x in (negative.strip(), fam.negative_default) if x)
        neg = out(g.add("CLIPTextEncode", text=neg_text, clip=clip))
    else:
        neg = out(g.add("ConditioningZeroOut", conditioning=pos))
    return pos, neg, text


def _sampler(g: Graph, model: ModelSpec, m, pos, neg, latent, seed: int, denoise: float,
             steps: int | None, cfg: float | None):
    fam = model.family
    return out(g.add(
        "KSampler", seed=int(seed), steps=int(steps or fam.steps), cfg=float(cfg if cfg is not None else fam.cfg),
        sampler_name=fam.sampler, scheduler=fam.scheduler, denoise=float(denoise),
        model=m, positive=pos, negative=neg, latent_image=latent,
    ))


def txt2img(model: ModelSpec, prompt: str, negative: str, width: int, height: int, seed: int, *,
            models_dir: Path | None = None, steps: int | None = None, cfg: float | None = None,
            prefix: str = "gen-studio/img") -> tuple[dict, dict]:
    g = Graph()
    m, c, v = _loaders(g, model, models_dir)
    pos, neg, text = _conditioning(g, model, c, prompt, negative)
    latent_cls = "EmptySD3LatentImage" if model.family.latent == "sd3" else "EmptyLatentImage"
    latent = out(g.add(latent_cls, width=int(width), height=int(height), batch_size=1))
    s = _sampler(g, model, m, pos, neg, latent, seed, 1.0, steps, cfg)
    img = out(g.add("VAEDecode", samples=s, vae=v))
    g.add("SaveImage", filename_prefix=prefix, images=img)
    return g.nodes, {"final_prompt": text}


def img2img(model: ModelSpec, image_name: str, prompt: str, negative: str, seed: int, denoise: float, *,
            models_dir: Path | None = None, steps: int | None = None, cfg: float | None = None,
            prefix: str = "gen-studio/edit") -> tuple[dict, dict]:
    g = Graph()
    m, c, v = _loaders(g, model, models_dir)
    pos, neg, text = _conditioning(g, model, c, prompt, negative)
    src = g.add("LoadImage", image=image_name)
    latent = out(g.add("VAEEncode", pixels=out(src), vae=v))
    s = _sampler(g, model, m, pos, neg, latent, seed, denoise, steps, cfg)
    img = out(g.add("VAEDecode", samples=s, vae=v))
    g.add("SaveImage", filename_prefix=prefix, images=img)
    return g.nodes, {"final_prompt": text}


def inpaint(model: ModelSpec, image_name: str, mask_name: str, prompt: str, negative: str, seed: int, *,
            denoise: float = 1.0, grow: int = 12, models_dir: Path | None = None, steps: int | None = None,
            cfg: float | None = None, prefix: str = "gen-studio/edit") -> tuple[dict, dict]:
    """Redraws only the white area of the mask; everything else is pasted back untouched."""
    g = Graph()
    m, c, v = _loaders(g, model, models_dir)
    m = out(g.add("DifferentialDiffusion", model=m))
    pos, neg, text = _conditioning(g, model, c, prompt, negative)
    src = out(g.add("LoadImage", image=image_name))
    mask = out(g.add("LoadImageMask", image=mask_name, channel="red"))
    if grow:
        mask = out(g.add("GrowMask", mask=mask, expand=int(grow), tapered_corners=True))
    latent = out(g.add("VAEEncode", pixels=src, vae=v))
    latent = out(g.add("SetLatentNoiseMask", samples=latent, mask=mask))
    s = _sampler(g, model, m, pos, neg, latent, seed, denoise, steps, cfg)
    img = out(g.add("VAEDecode", samples=s, vae=v))
    merged = out(g.add("ImageCompositeMasked", destination=src, source=img, x=0, y=0, resize_source=False, mask=mask))
    g.add("SaveImage", filename_prefix=prefix, images=merged)
    return g.nodes, {"final_prompt": text}


REQUIRED_NODES = sorted({
    "UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple", "ModelSamplingAuraFlow", "CLIPSetLastLayer",
    "CLIPTextEncode", "ConditioningZeroOut", "EmptySD3LatentImage", "EmptyLatentImage", "KSampler", "VAEDecode",
    "VAEEncode", "SaveImage", "LoadImage", "LoadImageMask", "GrowMask", "SetLatentNoiseMask", "ImageCompositeMasked",
    "DifferentialDiffusion",
})
