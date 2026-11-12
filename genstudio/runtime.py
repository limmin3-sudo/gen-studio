"""What the desktop app manages besides the web server: the ComfyUI process, Tailscale remote access, model
downloads with progress, Codex registration, updates, and a few remembered preferences.

Everything here is used from the settings screen; the old .bat files keep working on their own.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from . import config as cfg

REPO = "limmin3-sudo/gen-studio"
WIN = sys.platform == "win32"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if WIN else 0


def data_dir() -> Path:
    return cfg.DATA_DIR


def log_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ preferences
def prefs() -> dict:
    try:
        return json.loads((data_dir() / "app.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def save_prefs(**changes) -> dict:
    p = {**prefs(), **changes}
    data_dir().mkdir(parents=True, exist_ok=True)
    (data_dir() / "app.json").write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ------------------------------------------------------------------ ComfyUI process
def comfy_alive(url: str) -> bool:
    try:
        return httpx.get(url + "/system_stats", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


class ComfyProcess:
    """Starts the portable ComfyUI without a console window; its output goes to data/logs/comfyui.log."""

    def __init__(self, settings: cfg.Settings):
        self.s = settings
        self.proc: subprocess.Popen | None = None
        self.state = "unknown"           # running | starting | stopped | missing | updating | error
        self.message = ""
        self._lock = threading.Lock()

    @property
    def root(self) -> Path | None:
        return self.s.comfy_path.parent if self.s.comfy_path else None

    @property
    def installed(self) -> bool:
        r = self.root
        return bool(r and (r / "python_embeded" / "python.exe").exists() and (self.s.comfy_path / "main.py").exists())

    @property
    def managed(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def alive(self) -> bool:
        return comfy_alive(self.s.comfy_url)

    def start(self, wait: float = 180) -> bool:
        with self._lock:
            if self.alive():
                self.state = "running"
                return True
            if not self.installed:
                self.state, self.message = "missing", "ComfyUI가 설치돼 있지 않아요. install.bat을 실행하세요."
                return False
            port = self.s.comfy_url.rsplit(":", 1)[-1].strip("/") or "8188"
            cmd = [str(self.root / "python_embeded" / "python.exe"), "-s", str(self.s.comfy_path / "main.py"),
                   "--windows-standalone-build", "--disable-auto-launch", "--listen", "127.0.0.1", "--port", port]
            log = open(log_dir() / "comfyui.log", "a", encoding="utf-8", errors="replace")
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} ComfyUI 시작 =====\n")
            log.flush()
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            self.proc = subprocess.Popen(cmd, cwd=str(self.root), stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, creationflags=NO_WINDOW, env=env)
            self.state, self.message = "starting", "ComfyUI 켜는 중"
        end = time.monotonic() + wait
        while time.monotonic() < end:
            if self.alive():
                self.state, self.message = "running", ""
                return True
            if self.proc and self.proc.poll() is not None:
                self.state, self.message = "error", "ComfyUI가 바로 꺼졌어요. 진단 정보의 ComfyUI 로그를 확인하세요."
                return False
            time.sleep(1)
        self.state, self.message = "error", "ComfyUI가 제때 켜지지 않았어요."
        return False

    def stop(self) -> None:
        with self._lock:
            p, self.proc = self.proc, None
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    p.kill()
            self.state = "stopped"

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def update(self) -> tuple[bool, str]:
        """Runs ComfyUI's own update script (needs ComfyUI stopped), then starts it again."""
        bat = self.root / "update" / "update_comfyui.bat" if self.root else None
        if not bat or not bat.exists():
            return False, "ComfyUI 업데이트 파일을 찾지 못했어요."
        self.stop()
        self.state, self.message = "updating", "ComfyUI 업데이트 중"
        with open(log_dir() / "comfyui-update.log", "w", encoding="utf-8", errors="replace") as log:
            try:
                r = subprocess.run(["cmd", "/c", str(bat)], cwd=str(bat.parent), input="\n\n\n", text=True,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW, timeout=1800)
                ok = r.returncode == 0
            except (OSError, subprocess.SubprocessError) as e:
                log.write(str(e))
                ok = False
        started = self.start()
        return ok and started, "ComfyUI를 업데이트했어요." if ok else "업데이트 중 문제가 있었어요. 로그를 확인하세요."


# ------------------------------------------------------------------ Tailscale
def tailscale_exe() -> str | None:
    exe = shutil.which("tailscale") or (r"C:\Program Files\Tailscale\tailscale.exe" if WIN else None)
    return exe if exe and Path(exe).exists() else None


def tailscale_name() -> str | None:
    exe = tailscale_exe()
    if not exe:
        return None
    try:
        st = json.loads(subprocess.run([exe, "status", "--json"], capture_output=True, encoding="utf-8",
                                       errors="replace", timeout=8, creationflags=NO_WINDOW).stdout)
        return st["Self"]["DNSName"].rstrip(".") or None
    except Exception:  # noqa: BLE001 - not logged in, not running ...
        return None


def tailscale_serve(port: int, on: bool) -> tuple[bool, str]:
    exe = tailscale_exe()
    if not exe:
        return False, "Tailscale이 설치돼 있지 않아요 (docs/remote.md)."
    args = [exe, "serve", "--bg", str(port)] if on else [exe, "serve", "reset"]
    try:
        r = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", timeout=60,
                           creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    msg = (r.stdout + r.stderr).strip()
    return r.returncode == 0, msg


# ------------------------------------------------------------------ versions and updates
def local_version() -> dict:
    try:
        return json.loads((data_dir() / "version.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


async def latest_version() -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=10, headers={"Accept": "application/vnd.github+json",
                                                          "User-Agent": "gen-studio", "Cache-Control": "no-cache"}) as c:
            r = await c.get(f"https://api.github.com/repos/{REPO}/commits/main")
            if r.status_code != 200:
                return None
            j = r.json()
            return {"commit": j["sha"], "date": j["commit"]["committer"]["date"],
                    "message": j["commit"]["message"].splitlines()[0]}
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def pythonw() -> str:
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return str(w if WIN and w.exists() else exe)


def spawn_updater(port: int) -> None:
    """The updater replaces files while the app is closed, then opens the app again."""
    flags = (getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if WIN else 0
    subprocess.Popen([pythonw(), str(cfg.ROOT / "scripts" / "updater.py"), "--port", str(port), "--relaunch"],
                     cwd=str(cfg.ROOT), creationflags=flags | NO_WINDOW, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------ model downloads
class Downloads:
    """Missing model files fetched in the background, with progress for the settings screen."""

    def __init__(self, settings: cfg.Settings, registry):
        self.s, self.reg = settings, registry
        self.state: dict = {"running": False}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def missing(self) -> list[dict]:
        from .downloader import plan
        if not self.s.models_dir:
            return []
        return [{"label": label, "file": f.name, "dir": f.dir, "size": f.size or 0, "civitai": bool(f.civitai_version)}
                for label, f, ok in plan(self.reg, self.s.models_dir) if not ok]

    def start(self, tools: list[str] | None = None) -> dict:
        """Fetch what's missing; with `tools`, just those (e.g. the optional LTX video files)."""
        if self.state.get("running"):
            return self.state
        self._tools = tools
        if not self.s.models_dir:
            raise ValueError("config/settings.yaml 의 comfyui path가 비어 있어요")
        self._stop.clear()
        self.state = {"running": True, "index": 0, "count": 0, "file": "", "done": 0, "total": 0, "speed": 0,
                      "failed": [], "error": None, "finished": False}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self.state

    def cancel(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        from .downloader import UA, civitai_key, download, plan
        st = self.state
        try:
            todo = [f for _l, f, ok in plan(self.reg, self.s.models_dir, tools=getattr(self, "_tools", None)) if not ok]
            key = civitai_key(interactive=False)
            if any(f.civitai_version for f in todo) and not key:
                st["error"] = "Civitai 모델을 받으려면 API 키가 필요해요. 아래에 넣어 주세요."
                todo = [f for f in todo if not f.civitai_version]
            st["count"] = len(todo)
            with httpx.Client(headers={"User-Agent": UA}) as client:
                for i, f in enumerate(todo, 1):
                    if self._stop.is_set():
                        break
                    st.update(index=i, file=f.name, done=0, total=f.size or 0, speed=0)

                    def prog(done, total, speed):
                        if self._stop.is_set():
                            raise InterruptedError
                        st.update(done=done, total=total or st["total"], speed=speed)
                    try:
                        download(f, f.path(self.s.models_dir), client, key, on_progress=prog)
                    except InterruptedError:
                        break
                    except Exception as e:  # noqa: BLE001 - keep going with the next file
                        st["failed"].append(f"{f.name}: {e}")
        finally:
            st.update(running=False, finished=True)
            try:
                from . import user_models
                user_models.fix_locations(self.reg, self.s.models_dir)
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ unused model files
MODEL_DIRS = ("diffusion_models", "checkpoints", "text_encoders", "vae", "upscale_models", "controlnet",
              "model_patches", "background_removal", "unet", "clip")


def unused_model_files(registry, models_dir: Path | None, extra_keep: set[str] | None = None) -> list[dict]:
    """Model files in ComfyUI/models that no listed model or tool uses (old lineup leftovers, half downloads).
    LoRAs are left to the LoRA tab."""
    if not models_dir or not models_dir.exists():
        return []
    keep = {f"{f.dir}/{f.name}" for m in registry.models for f in m.all_files()}
    keep |= {f"{f.dir}/{f.name}" for files in registry.tools.values() for f in files}
    keep |= extra_keep or set()
    out = []
    for d in MODEL_DIRS:
        folder = models_dir / d
        if not folder.is_dir():
            continue
        for p in folder.rglob("*"):
            if not p.is_file() or p.name.startswith(".") or p.suffix.lower() in (".txt", ".md", ".json", ".yaml"):
                continue
            rel = p.relative_to(models_dir).as_posix()
            if rel in keep or p.name.startswith("put_"):
                continue
            out.append({"path": rel, "size": p.stat().st_size, "partial": p.name.endswith(".part")})
    return sorted(out, key=lambda x: -x["size"])


def delete_model_files(models_dir: Path, paths: list[str], allowed: list[dict]) -> int:
    ok = {a["path"] for a in allowed}
    n = 0
    for rel in paths:
        if rel not in ok:
            continue
        p = (models_dir / rel).resolve()
        if models_dir.resolve() in p.parents and p.is_file():
            p.unlink()
            n += 1
    return n


# ------------------------------------------------------------------ copies ComfyUI keeps of our pictures
def comfy_copy_dirs(comfy_path: Path | None) -> list[Path]:
    """ComfyUI saves every result again under output/gen-studio and keeps uploaded sources under input/gen-studio
    (plus input/gen-studio-train for LoRA training pictures). Only these folders are ours."""
    if not comfy_path:
        return []
    dirs = [comfy_path / "output" / "gen-studio", comfy_path / "input" / "gen-studio"]
    inp = comfy_path / "input"
    if inp.is_dir():
        dirs += sorted(d for d in inp.iterdir() if d.is_dir() and d.name.startswith("gen-studio-"))
    return [d for d in dirs if d.is_dir()]


def comfy_copies_bytes(comfy_path: Path | None) -> int:
    return sum(f.stat().st_size for d in comfy_copy_dirs(comfy_path) for f in d.rglob("*") if f.is_file())


def clear_comfy_copies(comfy_path: Path | None) -> int:
    freed = comfy_copies_bytes(comfy_path)
    for d in comfy_copy_dirs(comfy_path):
        shutil.rmtree(d, ignore_errors=True)
    return freed
