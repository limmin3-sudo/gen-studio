"""Model registry loaded from config/models.yaml."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import CONFIG_DIR


@dataclass
class FileSpec:
    name: str
    dir: str
    size: int | None = None
    sha256: str | None = None
    url: str | None = None
    civitai_version: int | None = None

    @property
    def source(self) -> str:
        return "civitai" if self.civitai_version else "huggingface"

    def path(self, models_dir: Path) -> Path:
        return models_dir / self.dir / self.name

    def installed(self, models_dir: Path | None) -> bool | None:
        if models_dir is None:
            return None
        p = self.path(models_dir)
        if not p.exists():
            return False
        return self.size is None or p.stat().st_size == self.size


@dataclass
class Family:
    id: str
    name: str
    loader: str = "checkpoint"          # checkpoint | unet
    latent: str = "sdxl"                # sdxl | sd3 | generic
    encoder: str = "clip"               # clip | qwen21 (TextEncodeQwenImage21)
    sigmas: str | None = None           # fixed step schedule for distilled models
    shift_node: str = "auraflow"        # auraflow | sd3 (which ModelSampling node applies `shift`)
    high_noise: str | None = None       # Wan 2.2: high-noise expert file; the model file is the low-noise expert
    switch_step: int | None = None      # step where the high-noise expert hands over to the low-noise one
    loras: list[dict] = field(default_factory=list)   # [{name, stage: high|low|main|both, strength}]
    clip: str | None = None
    clip_type: str | None = None
    vae: str | None = None
    shift: float | None = None
    clip_skip: int | None = None
    steps: int = 28
    cfg: float = 5.0
    sampler: str = "euler"
    scheduler: str = "normal"
    negative: bool = True
    prompt_prefix: str = ""
    negative_default: str = ""
    files: list[FileSpec] = field(default_factory=list)


@dataclass
class ModelSpec:
    id: str
    name: str
    family: Family
    style: str                          # realistic | semi | text
    role: str                           # main | alt
    file: FileSpec
    prompt_prefix: str | None = None
    keywords: list[str] = field(default_factory=list)   # words that pick this model directly
    optional: bool = False                               # not downloaded unless asked for

    @property
    def prefix(self) -> str:
        return self.prompt_prefix if self.prompt_prefix is not None else self.family.prompt_prefix

    def all_files(self) -> list[FileSpec]:
        return [self.file, *self.family.files]

    def installed(self, models_dir: Path | None) -> bool | None:
        states = [f.installed(models_dir) for f in self.all_files()]
        if any(s is None for s in states):
            return None
        return all(states)

    def loader_kind(self, models_dir: Path | None) -> str:
        """unet-style families sometimes ship as full checkpoints; peek at the file to be sure."""
        if self.family.loader != "unet" or models_dir is None:
            return self.family.loader
        p = self.file.path(models_dir)
        return "checkpoint" if p.exists() and is_full_checkpoint(p) else self.family.loader


def is_full_checkpoint(path: Path) -> bool:
    """True when a .safetensors file also carries a VAE and text encoder."""
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > 200_000_000:
                return False
            header = json.loads(fh.read(n))
    except (OSError, ValueError, struct.error):
        return False
    keys = [k for k in header if k != "__metadata__"]
    has_vae = any(k.startswith(("first_stage_model.", "vae.")) for k in keys)
    has_te = any(k.startswith(("conditioner.", "cond_stage_model.", "text_encoders.", "text_encoder.")) for k in keys)
    return has_vae and has_te


class Registry:
    def __init__(self, families: dict[str, Family], models: list[ModelSpec],
                 tools: dict[str, list[FileSpec]] | None = None):
        self.families = families
        self.models = models
        self.tools = tools or {}
        self.by_id = {m.id: m for m in models}

    def tool_ready(self, name: str, models_dir: Path | None) -> bool:
        files = self.tools.get(name, [])
        return bool(files) and all(f.installed(models_dir) is not False for f in files)

    def tool_file(self, name: str) -> str:
        return self.tools[name][0].name

    @classmethod
    def load(cls, path: Path | None = None) -> "Registry":
        path = path or CONFIG_DIR / "models.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        families: dict[str, Family] = {}
        for fid, f in (raw.get("families") or {}).items():
            f = dict(f)
            files = [FileSpec(**x) for x in f.pop("files", []) or []]
            families[fid] = Family(id=fid, files=files, **f)
        models = []
        for m in raw.get("models") or []:
            m = dict(m)
            fam = families[m.pop("family")]
            models.append(ModelSpec(family=fam, file=FileSpec(**m.pop("file")), **m))
        tools = {k: [FileSpec(**f) for f in v or []] for k, v in (raw.get("tools") or {}).items()}
        return cls(families, models, tools)

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self.by_id:
            raise KeyError(f"모델을 찾을 수 없어요: {model_id}")
        return self.by_id[model_id]

    def status(self, models_dir: Path | None) -> list[dict]:
        return [
            {
                "id": m.id,
                "name": m.name,
                "family": m.family.id,
                "family_name": m.family.name,
                "style": m.style,
                "role": m.role,
                "optional": m.optional,
                "installed": m.installed(models_dir),
                "missing": [f.name for f in m.all_files() if f.installed(models_dir) is False],
            }
            for m in self.models
        ]
