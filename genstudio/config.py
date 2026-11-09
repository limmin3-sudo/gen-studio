"""Settings, paths and locally stored secrets."""
from __future__ import annotations

import json
import os
import secrets as _secrets
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
SECRETS_FILE = DATA_DIR / "secrets.json"


def _resolve(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(p)))
    return path if path.is_absolute() else (ROOT / path)


@dataclass
class Settings:
    comfy_url: str = "http://127.0.0.1:8188"
    comfy_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 7860
    remote_access: bool = False
    output_dir: Path = field(default_factory=lambda: ROOT / "outputs")
    vram_gb: float = 16
    default_style: str = "realistic"
    translate: str = "auto"      # auto: translate Korean for models that need English | always | never

    @property
    def models_dir(self) -> Path | None:
        return self.comfy_path / "models" if self.comfy_path else None


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_DIR / "settings.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw or {}
    comfy = raw.get("comfyui", {}) or {}
    server = raw.get("server", {}) or {}
    s = Settings(
        comfy_url=str(comfy.get("url", Settings.comfy_url)).rstrip("/"),
        comfy_path=_resolve(comfy.get("path")),
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 7860)),
        remote_access=bool(server.get("remote_access", False)),
        output_dir=_resolve(raw.get("output_dir", "outputs")) or ROOT / "outputs",
        vram_gb=float((raw.get("gpu") or {}).get("vram_gb", 16)),
        default_style=str((raw.get("generation") or {}).get("default_style", "realistic")),
        translate=str((raw.get("generation") or {}).get("translate", "auto")),
    )
    # environment overrides (handy for tests and one-off runs)
    env = os.environ
    if env.get("GENSTUDIO_COMFY_URL"):
        s.comfy_url = env["GENSTUDIO_COMFY_URL"].rstrip("/")
    if env.get("GENSTUDIO_COMFY_PATH"):
        s.comfy_path = _resolve(env["GENSTUDIO_COMFY_PATH"])
    if env.get("GENSTUDIO_OUTPUT_DIR"):
        s.output_dir = _resolve(env["GENSTUDIO_OUTPUT_DIR"]) or s.output_dir
    if env.get("GENSTUDIO_PORT"):
        s.port = int(env["GENSTUDIO_PORT"])
    if env.get("GENSTUDIO_REMOTE"):
        s.remote_access = env["GENSTUDIO_REMOTE"].lower() in ("1", "true", "yes")
    return s


# ---------------------------------------------------------------- secrets
# Stored only on this PC in data/secrets.json (git-ignored).

def read_secrets() -> dict:
    try:
        return json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_secret(key: str, value: str) -> None:
    data = read_secrets()
    data[key] = value
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_secret(key: str, env: str | None = None) -> str | None:
    if env and os.environ.get(env):
        return os.environ[env]
    return read_secrets().get(key) or None


def access_token() -> str:
    """Token required for any request that does not come from this PC."""
    tok = get_secret("access_token", "GENSTUDIO_TOKEN")
    if not tok:
        tok = _secrets.token_urlsafe(32)
        write_secret("access_token", tok)
    return tok
