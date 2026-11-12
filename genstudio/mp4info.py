"""Frame count, frame rate, length and sound of an MP4/MOV from its header boxes, without decoding it.

ComfyUI's SaveVideo writes ordinary MP4s; the numbers here are what a player (or ffprobe) reports, so the
library shows the length of the file that was actually saved rather than the length that was asked for.
"""
from __future__ import annotations

import struct
from pathlib import Path

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}


def _boxes(data: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", data[i:i + 8])
        head = 8
        if size == 1:
            if i + 16 > end:
                return
            size = struct.unpack(">Q", data[i + 8:i + 16])[0]
            head = 16
        elif size == 0:
            size = end - i
        if size < head or i + size > end:
            return
        yield kind, i + head, i + size
        i += size


def _track(data: bytes, start: int, end: int) -> dict:
    t: dict = {}
    for kind, a, b in _boxes(data, start, end):
        if kind in _CONTAINERS:
            t.update(_track(data, a, b))
        elif kind == b"hdlr" and b - a >= 12:
            t["handler"] = data[a + 8:a + 12]
        elif kind == b"mdhd" and b - a >= 24:
            if data[a] == 1:
                t["timescale"] = struct.unpack(">I", data[a + 20:a + 24])[0]
            else:
                t["timescale"] = struct.unpack(">I", data[a + 12:a + 16])[0]
        elif kind == b"stts" and b - a >= 8:
            n = struct.unpack(">I", data[a + 4:a + 8])[0]
            frames = ticks = 0
            for k in range(n):
                p = a + 8 + 8 * k
                if p + 8 > b:
                    break
                count, delta = struct.unpack(">II", data[p:p + 8])
                frames += count
                ticks += count * delta
            t["samples"], t["ticks"] = frames, ticks
    return t


def probe(data: bytes) -> dict | None:
    """{frames, fps, duration, audio, audio_duration} or None when this is not an MP4/MOV we can read."""
    moov = next(((a, b) for kind, a, b in _boxes(data, 0, len(data)) if kind == b"moov"), None)
    if not moov:
        return None
    video = audio = None
    for kind, a, b in _boxes(data, *moov):
        if kind != b"trak":
            continue
        t = _track(data, a, b)
        if t.get("handler") == b"vide" and video is None:
            video = t
        elif t.get("handler") == b"soun" and audio is None:
            audio = t
    if not video or not video.get("samples") or not video.get("ticks") or not video.get("timescale"):
        return None
    duration = video["ticks"] / video["timescale"]
    out = {"frames": video["samples"], "fps": round(video["samples"] / duration, 3),
           "duration": round(duration, 3), "audio": bool(audio and audio.get("samples"))}
    if out["audio"] and audio.get("timescale"):
        out["audio_duration"] = round(audio["ticks"] / audio["timescale"], 3)
    return out


def probe_file(path: Path) -> dict | None:
    try:
        return probe(Path(path).read_bytes())
    except OSError:
        return None
