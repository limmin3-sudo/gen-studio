"""Joining clips made at different frame rates, and reading the real length of a saved MP4."""
import struct

from genstudio import workflows
from genstudio.mp4info import probe


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _trak(handler: bytes, timescale: int, samples: int, delta: int) -> bytes:
    hdlr = _box(b"hdlr", b"\0" * 8 + handler + b"\0" * 12)
    mdhd = _box(b"mdhd", b"\0" * 12 + struct.pack(">II", timescale, samples * delta) + b"\0" * 4)
    stts = _box(b"stts", b"\0" * 4 + struct.pack(">III", 1, samples, delta))
    return _box(b"trak", _box(b"mdia", mdhd + hdlr + _box(b"minf", _box(b"stbl", stts))))


def _mp4(frames: int, fps: int, audio_secs: float | None = None) -> bytes:
    traks = _trak(b"vide", fps * 512, frames, 512)
    if audio_secs:
        traks += _trak(b"soun", 48000, round(audio_secs * 48000 / 1024), 1024)
    return _box(b"ftyp", b"isom\0\0\0\0isom") + _box(b"moov", _box(b"mvhd", b"\0" * 100) + traks) + _box(b"mdat", b"x")


def test_probe_reads_frames_fps_and_sound():
    assert probe(_mp4(97, 32)) == {"frames": 97, "fps": 32.0, "duration": 3.031, "audio": False}
    i = probe(_mp4(97, 24, 4.05))
    assert (i["frames"], i["fps"], i["duration"], i["audio"]) == (97, 24.0, 4.042, True)
    assert probe(b"\x00\x00\x00\x18ftypmp42fake-video") is None


def _frames(graph: dict) -> int:
    """Frames taken piece by piece out of a clip (not the poster frame)."""
    return sum(n["inputs"]["length"] for n in graph.values() if n["class_type"] == "ImageFromBatch"
               and graph[n["inputs"]["image"][0]]["class_type"] == "GetVideoComponents")


def test_retime_keeps_each_clip_length():
    idx = workflows.retime_indices(97, 24, 32)
    assert len(idx) == 129 and idx[:6] == [0, 0, 1, 2, 3, 3] and idx[-1] == 96
    assert workflows.retime_indices(97, 32, 32) == list(range(97))
    down = workflows.retime_indices(97, 32, 24)
    assert len(down) == 73 and down[:4] == [0, 1, 2, 4]


def test_join_mixed_fps_both_orders():
    wan = {"name": "wan.mp4", "frames": 97, "fps": 32.0, "audio": False}
    ltx = {"name": "ltx.mp4", "frames": 97, "fps": 24.0, "audio": True}
    want = 97 / 32 + 97 / 24
    for clips in ([wan, ltx], [ltx, wan]):
        g, plan = workflows.join_videos(clips, audio=True)
        assert plan["fps"] == 32.0 and plan["frames"] == 97 + 129
        assert abs(plan["duration"] - want) <= 1 / 32                          # within one output frame
        create = next(n for n in g.values() if n["class_type"] == "CreateVideo")["inputs"]
        assert create["fps"] == 32.0 and "audio" in create
        assert _frames(g) == 129                                                # the 24 fps clip, retimed
        # each clip's sound part is exactly as long as its picture part
        parts = sorted(n["inputs"]["duration"] for n in g.values() if n["class_type"] == "EmptyAudio")
        assert parts == sorted([97 / 32, 129 / 32])
        trims = [n["inputs"]["duration"] for n in g.values() if n["class_type"] == "TrimAudioDuration"]
        assert trims == [129 / 32]                                              # only the LTX clip has sound


def test_join_same_fps_uses_frames_as_they_are():
    a = {"name": "a.mp4", "frames": 81, "fps": 16.0, "audio": False}
    g, plan = workflows.join_videos([a, dict(a, name="b.mp4")], audio=False)
    assert plan == {"fps": 16.0, "frames": 162, "duration": 10.125}
    kinds = [n["class_type"] for n in g.values()]
    assert kinds.count("ImageFromBatch") == 1 and "EmptyAudio" not in kinds and "AudioConcat" not in kinds


def test_join_many_pieces_batches_in_groups():
    long = {"name": "l.mp4", "frames": 481, "fps": 24.0, "audio": False}       # 20 s at 24 fps -> 32 fps
    g, plan = workflows.join_videos([{"name": "s.mp4", "frames": 33, "fps": 32.0, "audio": False}, long], audio=False)
    assert plan["frames"] == 33 + 641
    for n in g.values():
        if n["class_type"] == "BatchImagesNode":
            assert len(n["inputs"]) <= 50
