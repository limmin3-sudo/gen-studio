"""The real model lineup (config/models.yaml): uncensored Qwen 2.1 as the realistic main, a Z-Image helper,
Illustrij for semi-real, DaSiWa for video. Other tests use the frozen copy in tests/fixtures."""
import io

import pytest
from PIL import Image

from genstudio.config import CONFIG_DIR
from genstudio.registry import FileSpec, Registry
from test_api import wait_job


@pytest.fixture
def real(settings, fake, monkeypatch, tmp_path):
    monkeypatch.setattr("genstudio.registry.CONFIG_DIR", CONFIG_DIR)
    monkeypatch.setattr("genstudio.config.SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr("genstudio.config.DATA_DIR", tmp_path)
    from fastapi.testclient import TestClient
    from genstudio.app import create_app
    with TestClient(create_app(settings, comfy_transport=fake.transport())) as c:
        yield c


def test_lineup_file():
    reg = Registry.load(CONFIG_DIR / "models.yaml")
    assert {m.id for m in reg.models} == {"redcraft-qw21", "pornmaster-zit", "illustrij"}
    assert "wan22" not in reg.families
    files = [f for m in reg.models for f in m.all_files()]
    files += [f for name, fs in reg.tools.items() if name not in reg.optional_tools for f in fs]
    assert all(f.sha256 for f in files)
    # optional (gated) downloads: listed, but never part of the default download
    assert not reg.optional_tools and "video_ltx" not in reg.tools          # LTX was removed (2026-11-12)
    assert not any("lightx2v" in f.name for f in reg.tools["video_i2v"])
    assert FileSpec("a", "b", size=1000, sha256="x", civitai_version=1).size_ok(1000 + 3000)
    assert not FileSpec("a", "b", size=1000, sha256="x", url="u").size_ok(1001)


def test_routing_and_helpers(real, fake):
    r = lambda p, **kw: real.post("/api/route", json={"prompt": p, **kw}).json()
    assert r("카페에 앉은 여자")["model_id"] == "redcraft-qw21"
    assert r("반실사 카페 여자")["model_id"] == "illustrij"
    assert r('간판에 "OPEN" 이라고 쓴 가게 포스터')["model_id"] == "redcraft-qw21"
    assert r("극실사 여자")["model_id"] == "redcraft-qw21"          # no Wan image model any more

    img = wait_job(real, real.post("/api/generate", json={"prompt": "full body photo of a woman", "aspect": "tall"}).json()["id"])["result_ids"][0]
    assert real.get(f"/api/images/{img}").json()["model_id"] == "redcraft-qw21"
    d = real.post(f"/api/images/{img}/detail", json={"targets": ["face"]}).json()
    assert d["decision"]["model_id"] == "pornmaster-zit"
    u = real.post(f"/api/images/{img}/upscale", json={"scale": 2}).json()
    assert u["decision"]["model_id"] == "pornmaster-zit"


def test_pose_uses_controlnet_helper_and_video_has_no_extra_loras(real, fake):
    im = Image.new("RGB", (900, 1600), (40, 40, 40))
    buf = io.BytesIO(); im.save(buf, "PNG")
    photo = real.post("/api/import", files={"file": ("p.png", buf.getvalue(), "image/png")}).json()["id"]
    plain = real.post("/api/route", json={"prompt": "photo of a woman", "pose": photo}).json()
    assert plain["model_id"] == "pornmaster-zit" and plain["pose"]["kind"] == "zimage"
    with_ref = real.post("/api/route", json={"prompt": "image 1 woman", "pose": photo, "references": [photo]}).json()
    assert with_ref["model_id"] == "redcraft-qw21" and with_ref["pose"]["kind"] == "qwen"

    v = wait_job(real, real.post(f"/api/images/{photo}/video", json={"seconds": 3}).json()["id"])
    assert v["status"] == "done", v["error"]
    g = list(fake.prompts.values())[-1]
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in g.values())
    unets = sorted(n["inputs"]["unet_name"] for n in g.values() if n["class_type"] == "UNETLoader")
    assert unets[0].startswith("DaSiWa_") and "HIGH" in unets[0] and "LOW" in unets[1]


def test_wan_video_is_720p_and_smoothed(real, fake):
    reg = Registry.load(CONFIG_DIR / "models.yaml")
    assert reg.tools["frame_interp"][0].name == "film_net_fp16.safetensors"
    img = real.post("/api/import", files={"file": ("p.png", _png(900, 1600), "image/png")}).json()["id"]
    v = real.post(f"/api/images/{img}/video", json={"prompt": "she smiles"}).json()
    d = v["decision"]
    assert d["smooth"] is True and d["fps"] == 32 and abs(d["width"] * d["height"] - 1280 * 720) < 40000
    v = wait_job(real, v["id"])
    assert v["status"] == "done", v["error"]
    g = list(fake.prompts.values())[-1]
    fi = next(n for n in g.values() if n["class_type"] == "FrameInterpolate")["inputs"]
    assert fi["multiplier"] == 2
    assert next(n for n in g.values() if n["class_type"] == "CreateVideo")["inputs"]["fps"] == 32.0
    # without smoothing: plain 16 fps
    v2 = wait_job(real, real.post(f"/api/images/{img}/video", json={"prompt": "x", "smooth": False}).json()["id"])
    g = list(fake.prompts.values())[-1]
    assert not any(n["class_type"] == "FrameInterpolate" for n in g.values())
    assert v2["decision"]["fps"] == 16


def _png(w, h):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (50, 50, 50)).save(b, "PNG")
    return b.getvalue()


def test_wan_keyframes_and_knobs(real, fake):
    """First + last shot (WanFirstLastFrameToVideo); with a middle shot two halves are made and joined."""
    a, b, c = (real.post("/api/import", files={"file": (f"{n}.png", _png(900, 1600), "image/png")}).json()["id"]
               for n in "abc")
    assert real.post(f"/api/images/{a}/video", json={"mid_image_id": b}).status_code == 400   # middle needs an end
    v = wait_job(real, real.post(f"/api/images/{a}/video", json={
        "end_image_id": c, "steps": 8, "smooth_multiplier": 3, "width": 480, "negative": "blurry"}).json()["id"])
    assert v["status"] == "done", v["error"]
    d = v["decision"]
    assert d["width"] == 480 and d["height"] % 16 == 0 and d["height"] > 800 and d["fps"] == 48 and "처음·끝" in d["action"]
    g = list(fake.prompts.values())[-1]
    flf = [n["inputs"] for n in g.values() if n["class_type"] == "WanFirstLastFrameToVideo"]
    assert len(flf) == 1 and flf[0]["length"] == 81 and "end_image" in flf[0]
    ks = [n["inputs"] for n in g.values() if n["class_type"] == "KSamplerAdvanced"]
    assert [(k["steps"], k["start_at_step"], k["end_at_step"]) for k in ks] == [(8, 0, 4), (8, 4, 10000)]
    assert next(n for n in g.values() if n["class_type"] == "FrameInterpolate")["inputs"]["multiplier"] == 3
    rec = real.get(f"/api/images/{v['result_ids'][0]}").json()
    assert rec["frames"] == 241 and rec["keyframes"] == [c]

    v = wait_job(real, real.post(f"/api/images/{a}/video", json={"end_image_id": c, "mid_image_id": b,
                                                                 "seconds": 10, "smooth": False}).json()["id"])
    assert v["status"] == "done", v["error"]
    assert v["decision"]["frames"] == 161
    g = list(fake.prompts.values())[-1]
    flf = [n["inputs"] for n in g.values() if n["class_type"] == "WanFirstLastFrameToVideo"]
    assert [f["length"] for f in flf] == [81, 81]
    assert any(n["class_type"] == "BatchImagesNode" for n in g.values())
    # regenerate keeps the keyframes
    r = wait_job(real, real.post(f"/api/images/{v['result_ids'][0]}/regenerate").json()["id"])
    assert r["status"] == "done" and r["request"]["mid_image_id"] == b

    # last frame out as a picture, then the two clips joined
    clip1, clip2 = v["result_ids"][0], r["result_ids"][0]
    f = wait_job(real, real.post(f"/api/videos/{clip1}/frame", json={"at": "last"}).json()["id"])
    assert f["status"] == "done", f["error"]
    g = list(fake.prompts.values())[-1]
    assert next(n for n in g.values() if n["class_type"] == "ImageFromBatch")["inputs"]["batch_index"] == 160
    assert next(n for n in g.values() if n["class_type"] == "LoadVideo")["inputs"]["file"].startswith("gs-clip-")
    frame = real.get(f"/api/images/{f['result_ids'][0]}").json()
    assert frame.get("media") != "video" and frame["parent_id"] == clip1
    assert real.post(f"/api/videos/{a}/frame", json={}).status_code == 400                     # not a clip
    j = wait_job(real, real.post("/api/videos/join", json={"video_ids": [clip1, clip2]}).json()["id"])
    assert j["status"] == "done", j["error"]
    g = list(fake.prompts.values())[-1]
    assert sum(n["class_type"] == "LoadVideo" for n in g.values()) == 2
    assert not any(n["class_type"] == "AudioConcat" for n in g.values())                      # Wan clips are silent
    joined = real.get(f"/api/images/{j['result_ids'][0]}").json()
    assert joined["media"] == "video" and joined["joined"] == [clip1, clip2] and joined["duration"] == 20.125
