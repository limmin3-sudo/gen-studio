"""One text block that explains the state of the install, for pasting into a chat when something breaks.

Contains versions, GPU/RAM, missing models and nodes, recent failed jobs with tracebacks, and the end of the
ComfyUI log. Secrets (access token, Civitai key) are masked before anything leaves this module.
"""
from __future__ import annotations

import collections
import logging
import platform
import re
import sys
import time
import traceback

from . import __version__

_RING: collections.deque = collections.deque(maxlen=200)


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if record.exc_info:
                msg += "\n" + "".join(traceback.format_exception(*record.exc_info))[-3000:]
            _RING.append(f"{time.strftime('%H:%M:%S', time.localtime(record.created))} {record.levelname} {msg}")
        except Exception:
            pass


def install() -> None:
    """Keep the last warnings and errors of Gen Studio (and the web server) in memory."""
    for name in ("genstudio", "uvicorn.error"):
        lg = logging.getLogger(name)
        if not any(isinstance(h, _RingHandler) for h in lg.handlers):
            h = _RingHandler(level=logging.WARNING)
            h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
            lg.addHandler(h)
            if lg.level == logging.NOTSET or lg.level > logging.WARNING:
                lg.setLevel(logging.WARNING)


def mask(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "***")
    return re.sub(r"(token=|Bearer\s+)[A-Za-z0-9._~+/=-]{6,}", r"\1***", text)


def _section(title: str, lines: list[str]) -> str:
    return f"## {title}\n" + ("\n".join(lines) if lines else "(없음)")


async def build(state, secrets: list[str]) -> str:
    from . import sysinfo
    from .training import TRAIN_NODES
    from .workflows import REQUIRED_NODES
    s, engine, comfy = state.settings, state.engine, state.comfy
    stats = await comfy.ping()
    out = [f"# Gen Studio 진단 ({time.strftime('%Y-%m-%d %H:%M:%S')})",
           _section("버전", [f"Gen Studio {__version__}", f"OS {platform.platform()}",
                            f"Python {sys.version.split()[0]}"])]

    snap = await sysinfo.snapshot(stats)
    g = snap.get("gpu") or {}
    out.append(_section("장치", [
        f"GPU {g.get('name', '?')} · 사용률 {g.get('util', '?')}% · VRAM {g.get('vram_used', '?')}/{g.get('vram_total', '?')}GB"
        f" · 온도 {g.get('temp', '?')}", f"RAM {snap.get('ram_used', '?')}/{snap.get('ram_total', '?')}GB · CPU {snap.get('cpu', '?')}%"]))

    lines = [f"주소 {s.comfy_url} · {'연결됨' if stats else '연결 안 됨'}", f"폴더 {s.comfy_path or '(설정 없음)'}"]
    missing, optional = [], {}
    if stats:
        sysd = stats.get("system") or {}
        lines.append(" · ".join(f"{k} {sysd[k]}" for k in ("comfyui_version", "pytorch_version", "python_version")
                                if sysd.get(k)) or "버전 정보 없음")
        try:
            info = await comfy.object_info()
            missing = [n for n in REQUIRED_NODES if n not in info]
            optional = {k: [n for n in v if n not in info] for k, v in {
                "배경 제거": ["LoadBackgroundRemovalModel", "RemoveBackground"], "LoRA 학습": TRAIN_NODES}.items()}
        except Exception as e:  # noqa: BLE001 - diagnostics must never fail
            lines.append(f"노드 목록을 읽지 못함: {e}")
    lines.append(f"없는 필수 노드: {', '.join(missing) or '없음'}")
    lines += [f"{k}용 노드 없음: {', '.join(v)} (update_comfyui.bat 필요)" for k, v in optional.items() if v]
    out.append(_section("ComfyUI", lines))

    reg = engine.registry
    models = [f"{'✓' if m.installed(s.models_dir) is not False else '✗'} {m.name} ({m.id})" for m in reg.models]
    models += [f"{'✓' if reg.tool_ready(t, s.models_dir) else '✗'} 도구: {t}" for t in reg.tools]
    out.append(_section("모델", models))

    out.append(_section("설정", [f"원격 접속 {'켜짐' if s.remote_access else '꺼짐'} · 번역 {s.translate} · 기본 스타일 "
                                f"{s.default_style}", f"저장 폴더 {s.output_dir}"]))

    run = engine.jobs.get(engine._running) if engine._running else None
    waiting = sum(1 for j in engine.jobs.values() if j.status == "queued")
    out.append(_section("작업", [f"진행 중: {run.kind} {run.decision.get('action', '')} {run.message} "
                                f"{round(run.progress * 100)}%" if run else "진행 중: 없음", f"대기 {waiting}개"]))

    failed = sorted((j for j in engine.jobs.values() if j.status == "error"), key=lambda j: j.created_at)[-8:]
    out.append(_section("최근 실패한 작업", [
        f"- {time.strftime('%H:%M:%S', time.localtime(j.created_at))} {j.kind} · "
        f"{j.decision.get('action') or j.decision.get('model_name') or ''} · {j.decision.get('model_name', '')}\n"
        f"  요청: {str(j.request.get('prompt') or j.request.get('name') or '')[:120]}\n  오류: {(j.error or '')[:600]}"
        for j in failed]))

    out.append(_section("Gen Studio 로그 (경고·오류)", list(_RING)[-40:]))

    comfy_log = []
    if stats:
        try:
            r = await comfy._req("GET", "/internal/logs")
            if r.status_code == 200:
                comfy_log = str(r.json()).splitlines()[-60:]
        except Exception:  # noqa: BLE001
            pass
    out.append(_section("ComfyUI 로그 (마지막 60줄)", comfy_log))
    return mask("\n\n".join(out), secrets)
