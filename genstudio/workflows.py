"""Builds ComfyUI API-format graphs for each model family.

Graphs are built in code instead of templated JSON so every family shares the
same three shapes: text-to-image, image-to-image and masked inpaint.
Family differences (loaders, text encoder node, latent, sampler) come from config/models.yaml.
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


def _patch_model(g: Graph, model: ModelSpec, m, stage: str, user_loras: list[dict] | None = None):
    """LoRAs, sampling shift and cache that sit between the loader and the sampler."""
    fam = model.family
    for lora in [*fam.loras, *(user_loras or [])]:
        if lora.get("stage", "both") in ("both", stage):
            m = out(g.add("LoraLoaderModelOnly", model=m, lora_name=lora["name"],
                          strength_model=float(lora.get("strength", 1.0))))
    if fam.shift is not None:
        node = "ModelSamplingSD3" if fam.shift_node == "sd3" else "ModelSamplingAuraFlow"
        m = out(g.add(node, model=m, shift=fam.shift))
    if fam.encoder == "qwen21":
        m = out(g.add("QwenImage21Cache", model=m, device="auto", dtype="default"))
    return m


def _loaders(g: Graph, model: ModelSpec, models_dir: Path | None, loras: list[dict] | None = None):
    """Returns ({stage: model_ref}, clip, vae). Two-expert models (Wan 2.2) have 'high' and 'low' stages."""
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
        # SDXL-style LoRAs also adjust the text encoder
        for lora in loras or []:
            n = g.add("LoraLoader", model=m, clip=c, lora_name=lora["name"], strength_model=float(lora.get("strength", 1.0)),
                      strength_clip=float(lora.get("strength", 1.0)))
            m, c = out(n, 0), out(n, 1)
        loras = []
    if fam.high_noise:
        high = out(g.add("UNETLoader", unet_name=fam.high_noise, weight_dtype="default"))
        models = {"high": _patch_model(g, model, high, "high", loras), "low": _patch_model(g, model, m, "low", loras)}
    else:
        models = {"main": _patch_model(g, model, m, "main", loras)}
    if fam.clip_skip:
        c = out(g.add("CLIPSetLastLayer", clip=c, stop_at_clip_layer=-abs(int(fam.clip_skip))))
    return models, c, v


def _conditioning(g: Graph, model: ModelSpec, clip, vae, prompt: str, negative: str):
    fam = model.family
    text = (model.prefix + prompt).strip() if prompt else model.prefix.strip(", ").strip()
    neg_text = ", ".join(x for x in (negative.strip(), fam.negative_default) if x)
    if fam.encoder == "qwen21":
        enc = g.add("TextEncodeQwenImage21", clip=clip, prompt=text, negative_prompt=neg_text, vae=vae, resolution=1024)
        return out(enc, 0), out(enc, 1), text
    pos = out(g.add("CLIPTextEncode", text=text, clip=clip))
    if fam.negative:
        neg = out(g.add("CLIPTextEncode", text=neg_text, clip=clip))
    else:
        neg = out(g.add("ConditioningZeroOut", conditioning=pos))
    return pos, neg, text


def _sigmas_from(schedule: str, denoise: float) -> str:
    """Partial denoise on a fixed schedule: start at `denoise` and keep the lower steps."""
    sig = [float(x) for x in schedule.split(",")]
    if denoise >= 0.999:
        return schedule
    tail = [s for s in sig if s < denoise]
    return ", ".join(f"{x:g}" for x in [denoise, *tail])


def _sampler(g: Graph, model: ModelSpec, models: dict, pos, neg, latent, seed: int, denoise: float,
             steps: int | None, cfg: float | None):
    fam = model.family
    cfg_v = float(cfg if cfg is not None else fam.cfg)
    n_steps = int(steps or fam.steps)

    if "high" in models:
        # Wan 2.2: the high-noise expert lays out the picture, the low-noise expert finishes it
        start = min(n_steps - 1, max(0, round(n_steps * (1 - denoise))))
        switch = int(fam.switch_step or n_steps // 2)
        common = dict(noise_seed=int(seed), steps=n_steps, cfg=cfg_v, sampler_name=fam.sampler,
                      scheduler=fam.scheduler, positive=pos, negative=neg)
        if start < switch:
            latent = out(g.add("KSamplerAdvanced", model=models["high"], add_noise="enable", latent_image=latent,
                               start_at_step=start, end_at_step=switch, return_with_leftover_noise="enable", **common))
            return out(g.add("KSamplerAdvanced", model=models["low"], add_noise="disable", latent_image=latent,
                             start_at_step=switch, end_at_step=10000, return_with_leftover_noise="disable", **common))
        return out(g.add("KSamplerAdvanced", model=models["low"], add_noise="enable", latent_image=latent,
                         start_at_step=start, end_at_step=10000, return_with_leftover_noise="disable", **common))

    m = models["main"]
    if fam.sigmas:
        # distilled models ship their own step schedule (e.g. Qwen-Image 2.1 Turbo)
        noise = g.add("RandomNoise", noise_seed=int(seed))
        guider = g.add("CFGGuider", model=m, positive=pos, negative=neg, cfg=cfg_v)
        sampler = g.add("KSamplerSelect", sampler_name=fam.sampler)
        sigmas = g.add("ManualSigmas", sigmas=_sigmas_from(fam.sigmas, denoise))
        return out(g.add("SamplerCustomAdvanced", noise=out(noise), guider=out(guider), sampler=out(sampler),
                         sigmas=out(sigmas), latent_image=latent), 0)
    return out(g.add(
        "KSampler", seed=int(seed), steps=n_steps, cfg=cfg_v,
        sampler_name=fam.sampler, scheduler=fam.scheduler, denoise=float(denoise),
        model=m, positive=pos, negative=neg, latent_image=latent,
    ))


def _empty_latent(g: Graph, model: ModelSpec, width: int, height: int):
    lat = model.family.latent
    if lat == "video":   # a one-frame video is a still image
        return out(g.add("EmptyHunyuanLatentVideo", width=int(width), height=int(height), length=1, batch_size=1))
    cls = {"sd3": "EmptySD3LatentImage"}.get(lat, "EmptyLatentImage")
    return out(g.add(cls, width=int(width), height=int(height), batch_size=1))


def txt2img(model: ModelSpec, prompt: str, negative: str, width: int, height: int, seed: int, *,
            models_dir: Path | None = None, steps: int | None = None, cfg: float | None = None,
            prefix: str = "gen-studio/img", loras: list[dict] | None = None) -> tuple[dict, dict]:
    g = Graph()
    ms, c, v = _loaders(g, model, models_dir, loras)
    pos, neg, text = _conditioning(g, model, c, v, prompt, negative)
    latent = _empty_latent(g, model, width, height)
    s = _sampler(g, model, ms, pos, neg, latent, seed, 1.0, steps, cfg)
    img = out(g.add("VAEDecode", samples=s, vae=v))
    g.add("SaveImage", filename_prefix=prefix, images=img)
    return g.nodes, {"final_prompt": text}


def img2img(model: ModelSpec, image_name: str, prompt: str, negative: str, seed: int, denoise: float, *,
            models_dir: Path | None = None, steps: int | None = None, cfg: float | None = None,
            prefix: str = "gen-studio/edit", loras: list[dict] | None = None) -> tuple[dict, dict]:
    g = Graph()
    ms, c, v = _loaders(g, model, models_dir, loras)
    pos, neg, text = _conditioning(g, model, c, v, prompt, negative)
    src = g.add("LoadImage", image=image_name)
    latent = out(g.add("VAEEncode", pixels=out(src), vae=v))
    s = _sampler(g, model, ms, pos, neg, latent, seed, denoise, steps, cfg)
    img = out(g.add("VAEDecode", samples=s, vae=v))
    g.add("SaveImage", filename_prefix=prefix, images=img)
    return g.nodes, {"final_prompt": text}


def _masked_redraw(g: Graph, model: ModelSpec, ms: dict, c, v, src, mask, prompt: str, negative: str, seed: int,
                   denoise: float, steps: int | None, cfg: float | None, prefix: str) -> str:
    ms = {k: out(g.add("DifferentialDiffusion", model=m)) for k, m in ms.items()}
    pos, neg, text = _conditioning(g, model, c, v, prompt, negative)
    latent = out(g.add("VAEEncode", pixels=src, vae=v))
    latent = out(g.add("SetLatentNoiseMask", samples=latent, mask=mask))
    s = _sampler(g, model, ms, pos, neg, latent, seed, denoise, steps, cfg)
    img = out(g.add("VAEDecode", samples=s, vae=v))
    merged = out(g.add("ImageCompositeMasked", destination=src, source=img, x=0, y=0, resize_source=False, mask=mask))
    g.add("SaveImage", filename_prefix=prefix, images=merged)
    return text


def inpaint(model: ModelSpec, image_name: str, mask_name: str, prompt: str, negative: str, seed: int, *,
            denoise: float = 1.0, grow: int = 12, models_dir: Path | None = None, steps: int | None = None,
            cfg: float | None = None, prefix: str = "gen-studio/edit", loras: list[dict] | None = None) -> tuple[dict, dict]:
    """Redraws only the white area of the mask; everything else is pasted back untouched."""
    g = Graph()
    ms, c, v = _loaders(g, model, models_dir, loras)
    src = out(g.add("LoadImage", image=image_name))
    mask = out(g.add("LoadImageMask", image=mask_name, channel="red"))
    if grow:
        mask = out(g.add("GrowMask", mask=mask, expand=int(grow), tapered_corners=True))
    text = _masked_redraw(g, model, ms, c, v, src, mask, prompt, negative, seed, denoise, steps, cfg, prefix)
    return g.nodes, {"final_prompt": text}


def outpaint(model: ModelSpec, image_name: str, pads: dict, prompt: str, negative: str, seed: int, *,
             feathering: int = 40, models_dir: Path | None = None, steps: int | None = None, cfg: float | None = None,
             prefix: str = "gen-studio/edit", loras: list[dict] | None = None) -> tuple[dict, dict]:
    """Extends the canvas (left/top/right/bottom pixels) and paints the new area to match."""
    g = Graph()
    ms, c, v = _loaders(g, model, models_dir, loras)
    src = out(g.add("LoadImage", image=image_name))
    pad = g.add("ImagePadForOutpaint", image=src, left=int(pads.get("left", 0)), top=int(pads.get("top", 0)),
                right=int(pads.get("right", 0)), bottom=int(pads.get("bottom", 0)), feathering=int(feathering))
    text = _masked_redraw(g, model, ms, c, v, out(pad, 0), out(pad, 1), prompt, negative, seed, 1.0, steps, cfg, prefix)
    return g.nodes, {"final_prompt": text}


def reference(model: ModelSpec, image_names: list[str], prompt: str, negative: str, seed: int, *,
              width: int | None = None, height: int | None = None, models_dir: Path | None = None,
              cfg: float | None = None, prefix: str = "gen-studio/ref", loras: list[dict] | None = None) -> tuple[dict, dict]:
    """Qwen-Image 2.1 reference mode (official Turbo edit template).

    image 1 is the picture being edited (or the main reference); output keeps its size unless width/height
    are given. Up to 10 images: people, outfits, places or styles the prompt refers to as image 1, image 2 ...
    """
    if model.family.encoder != "qwen21":
        raise ValueError("참고 이미지는 Qwen-Image 2.1에서만 쓸 수 있어요")
    if not 1 <= len(image_names) <= 10:
        raise ValueError("참고 이미지는 1~10장이에요")
    g = Graph()
    ms, c, v = _loaders(g, model, models_dir, loras)
    text = (model.prefix + prompt).strip()
    neg_text = ", ".join(x for x in (negative.strip(), model.family.negative_default) if x)
    images = {f"images.image_{i}": out(g.add("LoadImage", image=n)) for i, n in enumerate(image_names, 1)}
    enc = g.add("TextEncodeQwenImage21", clip=c, prompt=text, negative_prompt=neg_text, vae=v, resolution=0, **images)
    latent = _empty_latent(g, model, width, height) if width and height else out(enc, 2)
    s = _sampler(g, model, ms, out(enc, 0), out(enc, 1), latent, seed, 1.0, None, cfg)
    img = out(g.add("VAEDecode", samples=s, vae=v))
    g.add("SaveImage", filename_prefix=prefix, images=img)
    return g.nodes, {"final_prompt": text}


# ------------------------------------------------------------------ segmentation
def describe_objects(image_name: str, vlm_file: str, instruction: str, max_length: int = 600) -> dict:
    """Ask a vision-language model (Qwen3-VL) what is in the picture. Result text lands in a PreviewAny output."""
    g = Graph()
    img = out(g.add("LoadImage", image=image_name))
    clip = out(g.add("CLIPLoader", clip_name=vlm_file, type="stable_diffusion", device="default"))
    txt = g.add("TextGenerateV2", clip=clip, prompt=instruction, **{"images.image1": img}, max_length=max_length,
                sampling_mode="off", thinking=False)
    g.add("PreviewAny", source=out(txt, 0))
    return g.nodes


def rewrite_text(vlm_file: str, instruction: str, max_length: int = 400) -> dict:
    """Text-only generation with Qwen3-VL (used to turn Korean requests into English prompts)."""
    g = Graph()
    clip = out(g.add("CLIPLoader", clip_name=vlm_file, type="stable_diffusion", device="default"))
    txt = g.add("TextGenerateV2", clip=clip, prompt=instruction, max_length=max_length, sampling_mode="off", thinking=False)
    g.add("PreviewAny", source=out(txt, 0))
    return g.nodes


def segment_labels(image_name: str, sam_file: str, labels: list[str], threshold: float = 0.4,
                   prefix: str = "gen-studio/seg") -> tuple[dict, dict[str, int]]:
    """One SAM 3 text query per label. Returns the graph and {SaveImage node id: label index}."""
    g = Graph()
    ck = g.add("CheckpointLoaderSimple", ckpt_name=sam_file)
    img = out(g.add("LoadImage", image=image_name))
    saves: dict[str, int] = {}
    for i, label in enumerate(labels):
        enc = g.add("CLIPTextEncode", text=label, clip=out(ck, 1))
        det = g.add("SAM3_Detect", model=out(ck, 0), image=img, conditioning=out(enc), threshold=float(threshold),
                    refine_iterations=2, individual_masks=True)
        m = g.add("MaskToImage", mask=out(det, 0))
        saves[g.add("SaveImage", filename_prefix=f"{prefix}/{i}", images=out(m))] = i
    return g.nodes, saves


def segment_points(image_name: str, sam_file: str, positive: list[dict], negative: list[dict] | None = None,
                   prefix: str = "gen-studio/seg") -> dict:
    """SAM 3 click selection: positive/negative points in image pixels."""
    import json as _json
    g = Graph()
    ck = g.add("CheckpointLoaderSimple", ckpt_name=sam_file)
    img = out(g.add("LoadImage", image=image_name))
    pos = out(g.add("PrimitiveString", value=_json.dumps(positive)))
    kw = {}
    if negative:
        kw["negative_coords"] = out(g.add("PrimitiveString", value=_json.dumps(negative)))
    det = g.add("SAM3_Detect", model=out(ck, 0), image=img, positive_coords=pos, threshold=0.5,
                refine_iterations=2, individual_masks=False, **kw)
    m = g.add("MaskToImage", mask=out(det, 0))
    g.add("SaveImage", filename_prefix=f"{prefix}/click", images=out(m))
    return g.nodes


REQUIRED_NODES = sorted({
    "UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple", "ModelSamplingAuraFlow", "CLIPSetLastLayer",
    "CLIPTextEncode", "ConditioningZeroOut", "EmptySD3LatentImage", "EmptyLatentImage", "KSampler", "VAEDecode",
    "VAEEncode", "SaveImage", "LoadImage", "LoadImageMask", "GrowMask", "SetLatentNoiseMask", "ImageCompositeMasked",
    "DifferentialDiffusion",
    # Qwen-Image 2.1
    "TextEncodeQwenImage21", "QwenImage21Cache", "RandomNoise", "CFGGuider", "KSamplerSelect", "ManualSigmas",
    "SamplerCustomAdvanced",
    # Wan 2.2
    "ModelSamplingSD3", "LoraLoaderModelOnly", "KSamplerAdvanced", "EmptyHunyuanLatentVideo",
    # canvas extension
    "ImagePadForOutpaint", "LoraLoader",
    # segmentation
    "SAM3_Detect", "MaskToImage", "PrimitiveString", "TextGenerateV2", "PreviewAny",
})
