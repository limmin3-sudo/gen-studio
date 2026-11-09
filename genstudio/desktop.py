"""Gen Studio as a desktop app:  pythonw -m genstudio.desktop  (what the desktop shortcut runs)

One window (WebView2, the engine Windows already has), no console windows: ComfyUI and the web server run in
the background and log to data/logs. A tray icon keeps work going when the window is closed during a job.
Opening the shortcut again just brings the existing window forward.
"""
from __future__ import annotations

import argparse
import ctypes
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import httpx

from . import runtime
from .config import ROOT, load_settings

ICON = ROOT / "web" / "icon.ico"
TITLE = "Gen Studio"


def _quiet_streams() -> None:
    """pythonw has no console: send prints and tracebacks to data/logs/app.log instead of nowhere."""
    log = open(runtime.log_dir() / "app.log", "a", encoding="utf-8", errors="replace", buffering=1)
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} Gen Studio 시작 =====\n")
    if sys.stdout is None or not sys.stdout.isatty():
        sys.stdout = log
    if sys.stderr is None or not sys.stderr.isatty():
        sys.stderr = log


def _message(text: str) -> None:
    if runtime.WIN:
        ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x40)
    else:
        print(text)


def _running(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=2)
        return r.status_code == 200 and "version" in r.json()
    except (httpx.HTTPError, ValueError):
        return False


def _port_busy(port: int) -> bool:
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _set_window_icon() -> None:
    """WebView2 windows show the python icon in the taskbar; swap in ours (best effort)."""
    if not (runtime.WIN and ICON.exists()):
        return
    u = ctypes.windll.user32
    for _ in range(50):
        hwnd = u.FindWindowW(None, TITLE)
        if hwnd:
            for size, which in ((16, 0), (32, 1)):
                h = u.LoadImageW(None, str(ICON), 1, size, size, 0x10)
                if h:
                    u.SendMessageW(hwnd, 0x80, which, h)
            return
        time.sleep(0.2)


class Desktop:
    """What the API and the tray can ask the app to do."""

    def __init__(self, settings, comfy: runtime.ComfyProcess):
        self.s, self.comfy = settings, comfy
        self.window = None
        self.tray = None
        self.server = None
        self.engine = None
        self._told_tray = False
        self._quitting = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.s.port}/"

    # -------------------------------------------------------------- window
    def show(self) -> None:
        if self.window is None:
            _edge_app(self.url)
            return
        try:
            self.window.show()
            self.window.restore()
        except Exception:  # noqa: BLE001
            pass

    def busy(self) -> bool:
        e = self.engine
        return bool(e and (e._running or any(j.status == "queued" for j in e.jobs.values())))

    def on_closing(self) -> bool:
        """Closing the window during a job keeps it running in the tray; otherwise the app quits."""
        if self._quitting:
            return True
        if self.tray and self.busy():
            self.window.hide()
            if not self._told_tray:
                self._told_tray = True
                try:
                    self.tray.notify("작업이 끝날 때까지 트레이에서 계속 돌아가요. 완전히 끄려면 트레이 아이콘 → 종료.", TITLE)
                except Exception:  # noqa: BLE001
                    pass
            return False
        threading.Thread(target=self.quit, daemon=True).start()
        return False

    def quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        if self.server:
            self.server.should_exit = True
        self.comfy.stop()
        if self.tray:
            try:
                self.tray.stop()
            except Exception:  # noqa: BLE001
                pass
        if self.window:
            try:
                self.window.destroy()
            except Exception:  # noqa: BLE001
                pass

    def restart_for_update(self) -> None:
        runtime.spawn_updater(self.s.port)
        threading.Thread(target=lambda: (time.sleep(0.8), self.quit()), daemon=True).start()

    # -------------------------------------------------------------- tray
    def start_tray(self) -> None:
        try:
            import pystray
            from PIL import Image
        except ImportError:
            return
        img = Image.open(ICON) if ICON.exists() else Image.new("RGB", (64, 64), (77, 120, 245))

        def remote_checked(_item):
            return bool(self.s.remote_access)

        def toggle_remote(_icon, _item):
            from .app import set_remote
            set_remote(self.s, not self.s.remote_access)

        menu = pystray.Menu(
            pystray.MenuItem("열기", lambda: self.show(), default=True),
            pystray.MenuItem("원격 접속 (폰·노트북)", toggle_remote, checked=remote_checked),
            pystray.MenuItem("로그 폴더 열기", lambda: _open_folder(runtime.log_dir())),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("종료", lambda: threading.Thread(target=self.quit, daemon=True).start()),
        )
        self.tray = pystray.Icon("gen-studio", img, TITLE, menu)
        self.tray.run_detached()


def _open_folder(p: Path) -> None:
    if runtime.WIN:
        import os
        os.startfile(str(p))  # noqa: S606
    else:
        subprocess.Popen(["xdg-open", str(p)])


def _edge_app(url: str) -> None:
    """No WebView2 in Python: Edge's app mode still gives a window without tabs or address bar."""
    for exe in (shutil.which("msedge"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if exe and Path(exe).exists():
            subprocess.Popen([exe, f"--app={url}", "--window-size=1440,920"])
            return
    webbrowser.open(url)


def main() -> None:
    ap = argparse.ArgumentParser(prog="genstudio.desktop")
    ap.add_argument("--port", type=int)
    ap.add_argument("--after-update", action="store_true")
    args = ap.parse_args()
    _quiet_streams()
    if runtime.WIN:
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GenStudio.App")
        except Exception:  # noqa: BLE001
            pass

    s = load_settings()
    if args.port:
        s.port = args.port
    s.remote_access = bool(runtime.prefs().get("remote_access", s.remote_access))

    if _running(s.port):                    # second launch: bring the first window forward
        try:
            httpx.post(f"http://127.0.0.1:{s.port}/api/app/show", timeout=3)
        except httpx.HTTPError:
            pass
        return
    if _port_busy(s.port):
        _message(f"{s.port}번 포트를 다른 프로그램이 쓰고 있어요.\n그 프로그램을 끄거나 config/settings.yaml의 "
                 "server.port를 바꾼 뒤 다시 실행하세요.")
        return

    comfy = runtime.ComfyProcess(s)
    desk = Desktop(s, comfy)
    threading.Thread(target=comfy.start, daemon=True).start()

    import uvicorn
    from .app import create_app, set_remote
    app = create_app(s)
    app.state.gs.extra.update(desktop=desk, comfy_proc=comfy)
    desk.engine = app.state.gs.engine
    server = uvicorn.Server(uvicorn.Config(app, host=s.host, port=s.port, log_level="warning", log_config=None))
    desk.server = server
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    if s.remote_access:
        threading.Thread(target=set_remote, args=(s, True), daemon=True).start()

    desk.start_tray()
    try:
        import webview
    except ImportError:
        webview = None
    if webview is None:
        _edge_app(desk.url)
        while not desk._quitting and server.started and not server.should_exit:
            time.sleep(1)
        desk.quit()
        return

    storage = runtime.data_dir() / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    desk.window = webview.create_window(TITLE, desk.url, width=1440, height=920, min_size=(900, 600),
                                        background_color="#F6F5F2", text_select=True)
    desk.window.events.closing += desk.on_closing
    threading.Thread(target=_set_window_icon, daemon=True).start()
    webview.start(private_mode=False, storage_path=str(storage))
    desk.quit()


if __name__ == "__main__":
    main()
