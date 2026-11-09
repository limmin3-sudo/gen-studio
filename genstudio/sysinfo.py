"""Live machine numbers for the status bar: GPU load and VRAM (nvidia-smi), CPU and RAM (psutil).

Every source is optional: without nvidia-smi the VRAM comes from ComfyUI's /system_stats, without psutil the RAM
does too, and anything unknown is simply left out.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time

try:
    import psutil
except ImportError:          # status bar still works with ComfyUI's numbers
    psutil = None

GB = 1024 ** 3
_cache: dict = {"t": 0.0, "gpu": None}


def _nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        line = r.stdout.strip().splitlines()[0]
        name, util, used, total, temp = [x.strip() for x in line.split(",")]
        num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None
        return {"name": name, "util": num(util),
                "vram_used": round(num(used) / 1024, 2) if num(used) is not None else None,
                "vram_total": round(num(total) / 1024, 2) if num(total) is not None else None,
                "temp": num(temp)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


async def gpu() -> dict | None:
    if time.monotonic() - _cache["t"] > 1.5:
        _cache["gpu"] = await asyncio.to_thread(_nvidia_smi)
        _cache["t"] = time.monotonic()
    return _cache["gpu"]


async def snapshot(comfy_stats: dict | None) -> dict:
    out: dict = {"gpu": await gpu()}
    dev = ((comfy_stats or {}).get("devices") or [{}])[0]
    if not out["gpu"] and dev.get("vram_total"):
        out["gpu"] = {"name": str(dev.get("name", "GPU")).split(":")[-1].strip(), "util": None,
                      "vram_used": round((dev["vram_total"] - dev.get("vram_free", 0)) / GB, 2),
                      "vram_total": round(dev["vram_total"] / GB, 2), "temp": None}
    if psutil:
        vm = psutil.virtual_memory()
        out["ram_used"], out["ram_total"] = round((vm.total - vm.available) / GB, 1), round(vm.total / GB, 1)
        out["cpu"] = psutil.cpu_percent(interval=None)
    else:
        sysd = (comfy_stats or {}).get("system") or {}
        if sysd.get("ram_total"):
            out["ram_total"] = round(sysd["ram_total"] / GB, 1)
            out["ram_used"] = round((sysd["ram_total"] - sysd.get("ram_free", 0)) / GB, 1)
    return out
