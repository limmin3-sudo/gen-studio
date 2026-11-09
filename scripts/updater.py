"""Updates Gen Studio while the app is closed, then opens it again.

Started by the app's "업데이트" button (genstudio.runtime.spawn_updater). Steps:
  1. wait until the app has closed its port
  2. new code: `git pull` in a git checkout, otherwise download the main branch zip and copy it over
     (models, ComfyUI, your images and settings are never touched)
  3. install new Python packages
  4. refresh the desktop shortcut, remember the installed version
  5. start the app again
Everything is logged to data/logs/update.log.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "limmin3-sudo/gen-studio"
KEEP = {".venv", "data", "outputs", "ComfyUI_windows_portable", "tools", ".git"}
WIN = sys.platform == "win32"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if WIN else 0
LOG = ROOT / "data" / "logs" / "update.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def run(cmd: list[str], **kw) -> int:
    log("$ " + " ".join(cmd))
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        return subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
                              **kw).returncode


def port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def latest_commit() -> str | None:
    import httpx
    try:
        r = httpx.get(f"https://api.github.com/repos/{REPO}/commits/main", timeout=15,
                      headers={"User-Agent": "gen-studio-updater"})
        return r.json()["sha"] if r.status_code == 200 else None
    except Exception:  # noqa: BLE001
        return None


def update_code() -> None:
    if (ROOT / ".git").is_dir() and shutil.which("git"):
        if run(["git", "pull", "--ff-only"]) != 0:
            raise RuntimeError("git pull 실패")
        return
    import httpx
    log("main.zip 받는 중")
    r = httpx.get(f"https://github.com/{REPO}/archive/refs/heads/main.zip", follow_redirects=True, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        prefix = z.namelist()[0].split("/")[0] + "/"
        n = 0
        for info in z.infolist():
            rel = info.filename[len(prefix):]
            if not rel or rel.split("/")[0] in KEEP:
                continue
            dest = ROOT / rel
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))
            n += 1
    log(f"파일 {n}개 바꿈")


def install_packages() -> None:
    py = ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / ("uv.exe" if WIN else "uv"))
    if Path(uv).exists():
        code = run([uv, "pip", "install", "--python", str(py), "-r", "requirements.txt"])
    else:
        code = run([str(py), "-m", "pip", "install", "-r", "requirements.txt"])
    if code != 0:
        raise RuntimeError("패키지 설치 실패")


def refresh_shortcut() -> None:
    ps1 = ROOT / "scripts" / "make_shortcut.ps1"
    if WIN and ps1.exists():
        run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)])


def relaunch() -> None:
    exe = ROOT / ".venv" / ("Scripts/pythonw.exe" if WIN else "bin/python")
    flags = (getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if WIN else 0
    subprocess.Popen([str(exe), "-m", "genstudio.desktop", "--after-update"], cwd=str(ROOT), creationflags=flags,
                     close_fds=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--relaunch", action="store_true")
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"===== 업데이트 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n", encoding="utf-8")
    for _ in range(60):                 # the app closes itself right after starting us
        if not port_open(args.port):
            break
        time.sleep(0.5)
    time.sleep(1.5)                     # let ComfyUI release its files too
    status = {"updated_at": time.time()}
    try:
        update_code()
        install_packages()
        refresh_shortcut()
        status["commit"] = latest_commit()
        log("업데이트 완료")
    except Exception as e:  # noqa: BLE001 - always reopen the app, with the error shown there
        status["last_error"] = str(e)
        log(f"실패: {e}")
    prev = {}
    vf = ROOT / "data" / "version.json"
    try:
        prev = json.loads(vf.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        pass
    if "last_error" in status:
        status = {**prev, "last_error": status["last_error"], "updated_at": status["updated_at"]}
    vf.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.relaunch:
        relaunch()
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
