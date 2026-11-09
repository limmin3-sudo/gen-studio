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


def _apply_control(g: Graph, model: ModelSpec, ms: dict, pos, neg, v, control: dict | None):
    """Pose (or other) control image. Z-Image: Fun ControlNet Union as a model patch (official template);
    Illustrious/SDXL: a regular ControlNet on the prompts."""
    if not control:
        return ms, pos, neg
    img = out(g.add("LoadImage", image=control["image"]))
    strength = float(control.get("strength", 1.0))
    if control["kind"] == "zimage":
        patch = out(g.add("ModelPatchLoader", name=control["file"]))
        ms = {k: out(g.add("QwenImageDiffsynthControlnet", model=m, model_patch=patch, vae=v, image=img,
                           strength=strength)) for k, m in ms.items()}
    else:
        cn = out(g.add("ControlNetLoader", control_net_name=control["file"]))
        ap = g.add("ControlNetApplyAdvanced", positive=pos, negative=neg, control_net=cn, image=img, strength=strength,
                   start_percent=0.0, end_percent=float(control.get("end", 0.9)), vae=v)
        pos, neg = out(ap, 0), out(ap, 1)
    return ms, pos, neg


def pose_map(image_name: str, sdpose_file: str, detector_file: str, prefix: str = "gen-studio/pose") -> dict:
    """Photo -> OpenPose-style skeleton picture for every person (ComfyUI's SDPose multi-person template)."""
    g = Graph()
    img = out(g.add("LoadImage", image=image_name))
    ck = g.add("CheckpointLoaderSimple", ckpt_name=sdpose_file)
    det = out(g.add("UNETLoader", unet_name=detector_file, weight_dtype="default"))
    boxes = out(g.add("RTDETR_detect", model=det, image=img, threshold=0.5, class_name="person", max_detections=10))
    kp = out(g.add("SDPoseKeypointExtractor", model=out(ck, 0), vae=out(ck, 2), image=img, batch_size=16, bboxes=boxes))
    draw = g.add("SDPoseDrawKeypoints", keypoints=kp, draw_body=True, draw_hands=True, draw_face=True, draw_feet=True,
                 stick_width=4, face_point_size=2, score_threshold=0.5, draw_head=True)
    g.add("SaveImage", filename_prefix=prefix, images=out(draw))
    return g.nodes


def txt2img(model: ModelSpec, prompt: str, negative: str, width: int, height: int, seed: int, *,
            models_dir: Path | None = None, steps: int | None = None, cfg: float | None = None,
            prefix: str = "gen-studio/img", loras: list[dict] | None = None,
            control: dict | None = None) -> tuple[dict, dict]:
    g = Graph()
    ms, c, v = _loaders(g, model, models_dir, loras)
    pos, neg, text = _conditioning(g, model, c, v, prompt, negative)
    ms, pos, neg = _apply_control(g, model, ms, pos, neg, v, control)
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
    if model.family.encoder == "qwen21":
        # Qwen sees the whole enlarged canvas (image 1) and where it is empty (image 2), so the new part continues
        # the existing picture instead of being painted as a separate scene.
        instruction = (
            "Image 1 is a photo placed on a larger canvas; the plain gray area is empty. Image 2 shows that empty area in "
            "white. Extend the photo into the empty area so the result looks like one continuous photograph"
            + (f": {prompt}" if prompt else "") + ". Continue the same person, body proportions, clothing, furniture, "
            "floor, background, lighting and style. Do not change anything that is already visible in image 1.")
        mask_img = out(g.add("MaskToImage", mask=out(pad, 1)))
        neg_text = ", ".join(x for x in (negative.strip(), model.family.negative_default) if x)
        enc = g.add("TextEncodeQwenImage21", clip=c, prompt=instruction, negative_prompt=neg_text, vae=v, resolution=0,
                    **{"images.image_1": out(pad, 0), "images.image_2": mask_img})
        s = _sampler(g, model, ms, out(enc, 0), out(enc, 1), out(enc, 2), seed, 1.0, None, cfg)
        img = out(g.add("VAEDecode", samples=s, vae=v))
        merged = out(g.add("ImageCompositeMasked", destination=out(pad, 0), source=img, x=0, y=0, resize_source=True,
                           mask=out(pad, 1)))
        g.add("SaveImage", filename_prefix=prefix, images=merged)
        return g.nodes, {"final_prompt": instruction}
    text = _masked_redraw(g, model, ms, c, v, out(pad, 0), out(pad, 1), prompt, negative, seed, 1.0, steps, cfg, prefix)
    return g.nodes, {"final_prompt": text}


def masked_reference(model: ModelSpec, image_name: str, mask_name: str, instruction: str, negative: str, seed: int, *,
                     grow: int = 12, models_dir: Path | None = None, cfg: float | None = None,
                     prefix: str = "gen-studio/edit", loras: list[dict] | None = None) -> tuple[dict, dict]:
    """Qwen-Image 2.1 edit that sees the whole picture plus the mask (image 2) and only lands inside the mask.

    Keeps the person: the model edits with the original in view, and everything outside the mask is pasted back.
    """
    g = Graph()
    ms, c, v = _loaders(g, model, models_dir, loras)
    src = out(g.add("LoadImage", image=image_name))
    mask_img = out(g.add("LoadImage", image=mask_name))
    mask = out(g.add("LoadImageMask", image=mask_name, channel="red"))
    if grow:
        mask = out(g.add("GrowMask", mask=mask, expand=int(grow), tapered_corners=True))
    neg_text = ", ".join(x for x in (negative.strip(), model.family.negative_default) if x)
    enc = g.add("TextEncodeQwenImage21", clip=c, prompt=instruction, negative_prompt=neg_text, vae=v, resolution=0,
                **{"images.image_1": src, "images.image_2": mask_img})
    s = _sampler(g, model, ms, out(enc, 0), out(enc, 1), out(enc, 2), seed, 1.0, None, cfg)
    img = out(g.add("VAEDecode", samples=s, vae=v))
    merged = out(g.add("ImageCompositeMasked", destination=src, source=img, x=0, y=0, resize_source=True, mask=mask))
    g.add("SaveImage", filename_prefix=prefix, images=merged)
    return g.nodes, {"final_prompt": instruction}


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


def upscale(image_name: str, upscaler: str, scale: int = 2, refine_model: ModelSpec | None = None,
            prompt: str = "", denoise: float = 0.33, seed: int = 0, models_dir: Path | None = None,
            megapixels: float = 1.0, prefix: str = "gen-studio/upscale") -> tuple[dict, dict]:
    """2x / 4x upscale. Same recipe as ComfyUI's "Image Upscale (Z-Image-Turbo)" template: normalise to 1MP,
    Real-ESRGAN x4 then halve (= 2x), and lightly redraw at the new size so skin, hair and fabric get real detail
    instead of smoothing. 4x adds a second Real-ESRGAN step on top (a full redraw at 16MP would not fit in 16GB)."""
    g = Graph()
    img = out(g.add("LoadImage", image=image_name))
    img = out(g.add("ImageScaleToTotalPixels", image=img, upscale_method="lanczos", megapixels=float(megapixels), resolution_steps=1))
    um = out(g.add("UpscaleModelLoader", model_name=upscaler))

    def esrgan_2x(x):
        x = out(g.add("ImageUpscaleWithModel", upscale_model=um, image=x))
        return out(g.add("ImageScaleBy", image=x, upscale_method="lanczos", scale_by=0.5))

    img = esrgan_2x(img)
    text = ""
    if refine_model is not None:
        ms, c, v = _loaders(g, refine_model, models_dir)
        pos, neg, text = _conditioning(g, refine_model, c, v, prompt or "high quality photo, sharp details", "")
        lat = out(g.add("VAEEncode", pixels=img, vae=v))
        fam = refine_model.family
        s = g.add("KSampler", model=ms["main"], seed=int(seed), steps=5 if fam.id == "zimage" else 12,
                  cfg=1.0 if fam.id == "zimage" else max(1.0, fam.cfg - 1), positive=pos, negative=neg, latent_image=lat,
                  sampler_name="dpmpp_2m_sde" if fam.id == "zimage" else fam.sampler,
                  scheduler="beta" if fam.id == "zimage" else fam.scheduler, denoise=float(denoise))
        img = out(g.add("VAEDecode", samples=out(s), vae=v))
    if scale >= 4:
        img = esrgan_2x(img)
    g.add("SaveImage", filename_prefix=prefix, images=img)
    return g.nodes, {"final_prompt": text}


WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
                "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
                "背景人很多，倒着走")


def image_to_video(image_name: str, files: list[str], prompt: str, negative: str, width: int, height: int,
                   frames: int, seed: int, fps: int = 16, prefix: str = "gen-studio/video") -> dict:
    """Wan 2.2 I2V 14B with the 4-step LightX2V LoRAs: ComfyUI's official "Image to Video (Wan 2.2)" template.
    files: high, low, lora high, lora low, text encoder, vae."""
    high, low, l_high, l_low, te, vae = files
    g = Graph()
    clip = out(g.add("CLIPLoader", clip_name=te, type="wan", device="default"))
    v = out(g.add("VAELoader", vae_name=vae))

    def expert(unet, lora):
        m = out(g.add("UNETLoader", unet_name=unet, weight_dtype="default"))
        m = out(g.add("LoraLoaderModelOnly", model=m, lora_name=lora, strength_model=1.0))
        return out(g.add("ModelSamplingSD3", model=m, shift=5.0))

    mh, ml = expert(high, l_high), expert(low, l_low)
    pos = out(g.add("CLIPTextEncode", text=prompt, clip=clip))
    neg = out(g.add("CLIPTextEncode", text=negative or WAN_NEGATIVE, clip=clip))
    start = out(g.add("LoadImage", image=image_name))
    i2v = g.add("WanImageToVideo", positive=pos, negative=neg, vae=v, width=int(width), height=int(height),
                length=int(frames), batch_size=1, start_image=start)
    a = g.add("KSamplerAdvanced", model=mh, add_noise="enable", noise_seed=int(seed), steps=4, cfg=1.0,
              sampler_name="euler", scheduler="simple", positive=out(i2v, 0), negative=out(i2v, 1),
              latent_image=out(i2v, 2), start_at_step=0, end_at_step=2, return_with_leftover_noise="enable")
    b = g.add("KSamplerAdvanced", model=ml, add_noise="disable", noise_seed=int(seed), steps=4, cfg=1.0,
              sampler_name="euler", scheduler="simple", positive=out(i2v, 0), negative=out(i2v, 1),
              latent_image=out(a), start_at_step=2, end_at_step=10000, return_with_leftover_noise="disable")
    frames_img = out(g.add("VAEDecode", samples=out(b), vae=v))
    video = out(g.add("CreateVideo", images=frames_img, fps=float(fps)))
    g.add("SaveVideo", video=video, filename_prefix=prefix, **{"format": "mp4", "format.codec": "auto"})
    return g.nodes


def remove_background(image_name: str, model_file: str, prefix: str = "gen-studio/cutout") -> dict:
    """BiRefNet foreground mask (white = keep). The cut-out itself is composed in Python to keep full quality."""
    g = Graph()
    img = out(g.add("LoadImage", image=image_name))
    bg = out(g.add("LoadBackgroundRemovalModel", bg_removal_name=model_file))
    mask = out(g.add("RemoveBackground", bg_removal_model=bg, image=img))
    g.add("SaveImage", filename_prefix=prefix, images=out(g.add("MaskToImage", mask=mask)))
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
    "ControlNetLoader", "ControlNetApplyAdvanced",
    # upscale
    "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScaleBy", "ImageScaleToTotalPixels",
    # segmentation
    "SAM3_Detect", "MaskToImage", "PrimitiveString", "TextGenerateV2", "PreviewAny",
})
