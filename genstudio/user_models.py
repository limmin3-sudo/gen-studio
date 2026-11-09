"""Models added from the app by pasting a Civitai link (kept in data/user_models.yaml, next to the built-in list).

Only checkpoints of families the app can run are accepted: Z-Image Turbo (realistic), Illustrious / NoobAI /
SDXL / Pony (semi-real), Qwen-Image 2.1 (realistic, references and edits).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import httpx
import yaml

from . import config as cfg
from .registry import FileSpec, ModelSpec, Registry, is_full_checkpoint

UA = {"User-Agent": "gen-studio/0.1"}
# Civitai base model -> (family, style, folder); anything else is refused
BASES = [
    (re.compile(r"^z\s*image\s*turbo$", re.I), ("zimage", "realistic", "diffusion_models")),
    (re.compile(r"^qwen\s*2\.1$", re.I), ("qwen21", "realistic", "diffusion_models")),
    (re.compile(r"^(illustrious|noobai|sdxl 1\.0|pony)$", re.I), ("illustrious", "semi", "checkpoints")),
]
FORMAT_ORDER = {"fp8": 0, "bf16": 1, "fp16": 2, "int8": 3, "fp32": 4}


def path() -> Path:
    return cfg.DATA_DIR / "user_models.yaml"


def entries() -> list[dict]:
    try:
        return list(yaml.safe_load(path().read_text(encoding="utf-8")) or [])
    except FileNotFoundError:
        return []


def _save(items: list[dict]) -> None:
    path().parent.mkdir(parents=True, exist_ok=True)
    path().write_text(yaml.safe_dump(items, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _spec(reg: Registry, e: dict) -> ModelSpec | None:
    fam = reg.families.get(e.get("family"))
    if not fam:
        return None
    return ModelSpec(id=e["id"], name=e["name"], family=fam, style=e["style"], role=e.get("role", "alt"),
                     file=FileSpec(**e["file"]), best_for=e.get("best_for", ""), prompt_prefix=e.get("prompt_prefix"),
                     also=list(e.get("also") or []))


def merge(reg: Registry) -> None:
    """Add the user's models to a loaded registry (built-in ids win)."""
    for e in entries():
        if e.get("id") in reg.by_id:
            continue
        m = _spec(reg, e)
        if m:
            reg.models.append(m)
            reg.by_id[m.id] = m


def base_info(base: str | None) -> tuple[str, str, str] | None:
    for rx, info in BASES:
        if base and rx.search(base.strip()):
            return info
    return None


def pick_file(files: list[dict], family: str) -> dict | None:
    """The weights file to fetch: model files only, no GGUF; fp8 first for the big unet families (16GB cards)."""
    cands = [f for f in files if f.get("type") in ("Model", "Diffusion Model", "Pruned Model")
             and str(f.get("name", "")).lower().endswith(".safetensors")]
    if not cands:
        return None
    if family == "illustrious":
        return next((f for f in cands if f.get("primary")), cands[0])
    return sorted(cands, key=lambda f: (FORMAT_ORDER.get((f.get("metadata") or {}).get("fp"), 9), not f.get("primary")))[0]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "model"


async def add_from_civitai(reg: Registry, url: str, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    from .loras import parse_civitai_url
    mid, vid = parse_civitai_url(url)
    if not (mid or vid):
        raise ValueError("Civitai 모델 주소를 붙여넣어 주세요 (civitai.com 또는 civitai.red/models/...)")
    async with httpx.AsyncClient(headers=UA, timeout=20, transport=transport) as c:
        if vid:
            r = await c.get(f"https://civitai.com/api/v1/model-versions/{vid}")
            if r.status_code != 200:
                raise ValueError("Civitai에서 이 버전을 찾지 못했어요")
            v = r.json()
            model_name = (v.get("model") or {}).get("name") or v.get("name")
            mtype = (v.get("model") or {}).get("type")
        else:
            r = await c.get(f"https://civitai.com/api/v1/models/{mid}")
            if r.status_code != 200:
                raise ValueError("Civitai에서 이 모델을 찾지 못했어요")
            m = r.json()
            vs = m.get("modelVersions") or []
            v = next((x for x in vs if base_info(x.get("baseModel"))), vs[0] if vs else None)
            if not v:
                raise ValueError("받을 수 있는 버전이 없어요")
            model_name, mtype = m.get("name"), m.get("type")
    if mtype and mtype != "Checkpoint":
        raise ValueError(f"체크포인트가 아니에요 ({mtype}). LoRA는 LoRA 탭에서 추가하세요.")
    info = base_info(v.get("baseModel"))
    if not info:
        raise ValueError(f"'{v.get('baseModel')}' 계열은 아직 못 써요. Z-Image Turbo, Illustrious(SDXL·Pony), Qwen 2.1만 돼요. "
                         "주소 끝에 ?modelVersionId=… 로 맞는 버전을 고르세요.")
    family, style, folder = info
    f = pick_file(v.get("files") or [], family)
    if not f:
        raise ValueError("받을 수 있는 .safetensors 파일이 없어요 (GGUF 등은 아직 못 써요)")
    if family != "illustrious" and f.get("type") == "Model" and (f.get("sizeKB") or 0) > 9_000_000:
        folder = "checkpoints"        # big "Model" files usually carry the text encoder and VAE too
    name = f"{model_name} · {v.get('name')}".strip(" ·")
    entry = {
        "id": f"user-{_slug(model_name or '')}-{v['id']}", "name": name, "family": family, "style": style,
        "also": ["text"] if family == "qwen21" else [], "role": "alt",
        "best_for": f"Civitai에서 추가 ({v.get('baseModel')})",
        "file": {"name": f["name"], "dir": folder, "civitai_version": v["id"],
                 "size": int((f.get("sizeKB") or 0) * 1024) or None,
                 "sha256": (f.get("hashes") or {}).get("SHA256")},
        "civitai_url": f"https://civitai.red/models/{v.get('modelId')}?modelVersionId={v['id']}",
    }
    if family == "qwen21" and re.search(r"\bbase\b", v.get("name") or "", re.I):
        entry["note"] = "Base 버전은 Turbo보다 스텝이 많이 필요해서, 지금 설정(8스텝)으로는 흐리게 나올 수 있어요."
    items = [e for e in entries() if e.get("id") != entry["id"]]
    items.append(entry)
    _save(items)
    if entry["id"] not in reg.by_id:
        m = _spec(reg, entry)
        reg.models.append(m)
        reg.by_id[m.id] = m
    return entry


def remove(reg: Registry, model_id: str) -> bool:
    items = entries()
    keep = [e for e in items if e.get("id") != model_id]
    if len(keep) == len(items):
        return False
    _save(keep)
    m = reg.by_id.pop(model_id, None)
    if m in reg.models:
        reg.models.remove(m)
    return True


def fix_locations(reg: Registry, models_dir: Path | None) -> None:
    """A full checkpoint saved under diffusion_models is moved to checkpoints so ComfyUI's loader finds it."""
    if not models_dir:
        return
    items, changed = entries(), False
    for e in items:
        f = e.get("file") or {}
        p = models_dir / f.get("dir", "") / f.get("name", "")
        if f.get("dir") == "diffusion_models" and p.exists() and is_full_checkpoint(p):
            dest = models_dir / "checkpoints" / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), dest)
            f["dir"] = "checkpoints"
            changed = True
            m = reg.by_id.get(e["id"])
            if m:
                m.file.dir = "checkpoints"
    if changed:
        _save(items)
