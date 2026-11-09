"""Start Gen Studio (and ComfyUI if it is not running yet):  python -m genstudio"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

import httpx
import uvicorn

from .config import Settings, access_token, load_settings


def comfy_alive(url: str) -> bool:
    try:
        return httpx.get(url + "/system_stats", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_comfy(s: Settings) -> None:
    """Starts the portable ComfyUI in its own window when it is installed but not running."""
    if comfy_alive(s.comfy_url):
        print("  ComfyUI 실행 중 ✓")
        return
    root = s.comfy_path.parent if s.comfy_path else None
    py = root / "python_embeded" / "python.exe" if root else None
    if not (py and py.exists() and (s.comfy_path / "main.py").exists()):
        print("  ComfyUI를 찾지 못했어요. install.bat을 먼저 실행하거나 ComfyUI를 직접 켜 주세요.")
        return
    port = s.comfy_url.rsplit(":", 1)[-1].strip("/") or "8188"
    cmd = [str(py), "-s", str(s.comfy_path / "main.py"), "--windows-standalone-build",
           "--disable-auto-launch", "--listen", "127.0.0.1", "--port", port]
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    print("  ComfyUI 시작 중… (새 창이 열립니다. 그 창은 닫지 마세요)")
    subprocess.Popen(cmd, cwd=str(root), creationflags=flags)
    for i in range(180):
        if comfy_alive(s.comfy_url):
            print("  ComfyUI 준비 완료 ✓")
            return
        time.sleep(1)
    print("  ComfyUI가 아직 준비되지 않았어요. ComfyUI 창의 메시지를 확인하세요. (Gen Studio는 먼저 켭니다)")


def tailscale_url(port: int, enable: bool) -> str | None:
    """Publishes Gen Studio on the private Tailscale network (https://<pc>.<tailnet>.ts.net)."""
    exe = shutil.which("tailscale") or r"C:\Program Files\Tailscale\tailscale.exe"
    try:
        # Tailscale prints UTF-8; Korean Windows would otherwise decode it as cp949 and fail
        st = json.loads(subprocess.run([exe, "status", "--json"], capture_output=True, encoding="utf-8",
                                       errors="replace", timeout=8).stdout)
        name = st["Self"]["DNSName"].rstrip(".")
    except Exception:
        return None
    if enable:
        r = subprocess.run([exe, "serve", "--bg", str(port)], capture_output=True, encoding="utf-8", errors="replace",
                           timeout=60)
        msg = (r.stdout + r.stderr).strip()
        if r.returncode != 0 or "http" in msg.lower() and "enable" in msg.lower():
            # first use: Tailscale asks to switch on Serve/HTTPS for the tailnet and prints a link
            print("\n  Tailscale 메시지:")
            for line in msg.splitlines():
                print("    " + line)
            print("  위에 링크가 있으면 브라우저로 열어 승인한 뒤 start-remote.bat을 다시 실행하세요.")
    return f"https://{name}"


def already_running(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=2)
        return r.status_code == 200 and "version" in r.json()
    except (httpx.HTTPError, ValueError):
        return False


def port_busy(port: int) -> bool:
    import socket
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="genstudio")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-comfy", action="store_true", help="ComfyUI 자동 실행 안 함")
    ap.add_argument("--port", type=int)
    ap.add_argument("--remote", action="store_true", help="노트북 등 다른 기기에서 접속 허용 (Tailscale)")
    args = ap.parse_args()

    s = load_settings()
    if args.port:
        s.port = args.port
    if args.remote:
        s.remote_access = True

    print("\n  Gen Studio")
    if already_running(s.port):
        print(f"\n  Gen Studio가 이미 켜져 있어요. 브라우저로 엽니다: http://127.0.0.1:{s.port}")
        print("  원격 접속(start-remote)으로 다시 켜려면, 켜져 있는 Gen Studio 창을 먼저 닫으세요.\n")
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{s.port}")
        return
    if port_busy(s.port):
        print(f"\n  {s.port}번 포트를 다른 프로그램이 쓰고 있어요. 그 프로그램을 끄거나,")
        print("  config/settings.yaml 의 server.port 를 다른 번호(예: 7870)로 바꾼 뒤 다시 실행하세요.\n")
        sys.exit(1)
    if not args.no_comfy:
        ensure_comfy(s)
    local = f"http://127.0.0.1:{s.port}"
    print(f"\n  이 PC에서 열기:     {local}")
    print(f"  Codex MCP 주소:     {local}/mcp")
    if s.remote_access:
        tok = access_token()
        url = tailscale_url(s.port, enable=True)
        if url:
            print(f"\n  노트북에서 열기:    {url}/?token={tok}")
            print(f"  노트북 Codex MCP:   {url}/mcp   (환경변수 GEN_STUDIO_TOKEN={tok[:6]}… 필요)")
        else:
            print("\n  Tailscale을 찾지 못했어요. docs/remote.md를 보고 설치하세요.")
        print("  접속 토큰은 data/secrets.json 에 있어요. 다른 사람에게 보여주지 마세요.")
    print("\n  이 창을 닫으면 Gen Studio가 꺼집니다.\n")

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(local)), daemon=True).start()

    from .app import create_app
    try:
        uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="warning")
    except OSError as e:
        print(f"  포트 {s.port}을(를) 열 수 없어요: {e}\n  이미 Gen Studio가 켜져 있는지 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
