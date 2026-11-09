"""LoRA library: detection, Civitai download, applying to generations."""
import hashlib
import json
import struct

import httpx
import pytest

from genstudio.loras import apply_triggers, family_from_base, guess_family_from_file, parse_civitai_url
from test_api import wait_job
from test_mcp import rpc
from test_phase2 import last_graph, nodes


def lora_bytes(keys, meta=None):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}
    if meta:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0\0"


ZIMAGE_LORA = lora_bytes(["diffusion_model.layers.0.attention.qkv.lora_A.weight"])
SDXL_LORA = lora_bytes(["lora_unet_input_blocks_4_1_proj_in.lora_down.weight", "lora_te1_text_model.lora_up.weight"])


def test_family_detection(tmp_path):
    assert family_from_base("ZImageTurbo") == "zimage" and family_from_base("Illustrious") == "illustrious"
    assert family_from_base("Wan Video 2.2 T2V-A14B") == "wan22" and family_from_base("Qwen 2.1") == "qwen21"
    p = tmp_path / "a.safetensors"
    p.write_bytes(SDXL_LORA)
    assert guess_family_from_file(p)[0] == "illustrious"
    p.write_bytes(ZIMAGE_LORA)
    assert guess_family_from_file(p)[0] == "zimage"
    p.write_bytes(lora_bytes(["x"], {"ss_base_model_version": "sdxl_base_v1-0"}))
    assert guess_family_from_file(p)[0] == "illustrious"
    assert parse_civitai_url("https://civitai.com/models/1662740/lenovo?modelVersionId=2452071") == (1662740, 2452071)
    assert apply_triggers("a woman", [{"triggers": ["l3n0v0"]}, {"triggers": ["a woman"]}]) == "l3n0v0, a woman"


@pytest.fixture
def lora_env(client, tmp_path):
    eng = client.app.state.gs.engine
    eng.loras.dir = tmp_path / "loras"
    sha = hashlib.sha256(ZIMAGE_LORA).hexdigest().upper()

    def civitai(req: httpx.Request):
        u = str(req.url)
        if f"by-hash/{sha}" in u or "model-versions/2452071" in u:
            return httpx.Response(200, json={"id": 2452071, "modelId": 1662740, "baseModel": "ZImageTurbo",
                                             "trainedWords": ["l3n0v0"], "model": {"name": "Lenovo UltraReal", "type": "LORA"},
                                             "files": [{"name": "lenovo_z.safetensors", "primary": True, "sizeKB": 1,
                                                        "downloadUrl": "https://civitai.com/api/download/models/2452071"}],
                                             "images": [{"url": "https://image.civitai.com/x.jpeg", "type": "image"}]})
        if "/api/download/models/2452071" in u:
            return httpx.Response(200, content=ZIMAGE_LORA)
        return httpx.Response(404, json={})
    eng.loras._transport = httpx.MockTransport(civitai)
    return eng


def test_upload_identifies_and_generation_uses_lora(client, fake, lora_env):
    r = client.post("/api/loras/upload", files={"file": ("lenovo_z.safetensors", ZIMAGE_LORA, "application/octet-stream")}).json()
    assert r["family"] == "zimage" and r["triggers"] == ["l3n0v0"] and r["name"] == "Lenovo UltraReal"
    r2 = client.post("/api/loras/upload", files={"file": ("anime_style.safetensors", SDXL_LORA, "application/octet-stream")}).json()
    assert r2["family"] == "illustrious" and "SDXL" in r2["note"]

    # a semi-real prompt with a Z-Image LoRA switches to the Z-Image model
    j = client.post("/api/generate", json={"prompt": "반실사 여자", "loras": [{"name": "lenovo", "strength": 0.7}]}).json()
    assert j["decision"]["model_id"] == "cyberrealistic-zit" and "LoRA" in j["decision"]["reasons"][-1]
    j = wait_job(client, j["id"])
    g = last_graph(fake)
    lo = nodes(g, "LoraLoaderModelOnly")[0]
    assert lo["lora_name"] == "lenovo_z.safetensors" and lo["strength_model"] == 0.7
    assert nodes(g, "CLIPTextEncode")[0]["text"].startswith("l3n0v0, ")
    rec = client.get(f"/api/images/{j['result_ids'][0]}").json()
    assert rec["loras"] == [{"name": "Lenovo UltraReal", "strength": 0.7}]

    # SDXL LoRA goes through LoraLoader (model + text encoder)
    j = wait_job(client, client.post("/api/generate", json={"prompt": "x", "loras": [{"name": "anime_style"}]}).json()["id"])
    assert nodes(last_graph(fake), "LoraLoader")[0]["strength_clip"] == 0.8
    # mixing families or forcing the wrong model is refused
    assert client.post("/api/generate", json={"prompt": "x", "loras": [{"name": "lenovo"}, {"name": "anime_style"}]}).status_code == 400
    assert client.post("/api/generate", json={"prompt": "x", "model": "wan22-14b", "loras": [{"name": "lenovo"}]}).status_code == 400
    assert client.post("/api/generate", json={"prompt": "x", "loras": [{"name": "nope"}]}).status_code == 404


def test_civitai_download_and_mcp(client, lora_env):
    did = client.post("/api/loras/civitai", json={"url": "https://civitai.com/models/1662740?modelVersionId=2452071"}).json()["download_id"]
    import time
    for _ in range(50):
        d = next(x for x in client.get("/api/loras").json()["downloads"] if x["id"] == did)
        if d["status"] != "running":
            break
        time.sleep(0.05)
    assert d["status"] == "done", d
    assert (lora_env.loras.dir / "lenovo_z.safetensors").exists()
    assert client.post("/api/loras/civitai", json={"url": "https://example.com"}).status_code == 400
    lst = json.loads(rpc(client, "tools/call", {"name": "list_loras", "arguments": {}})["content"][0]["text"])["loras"]
    assert lst[0]["name"] == "Lenovo UltraReal" and lst[0]["family"] == "zimage"
    # edits accept LoRAs too
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo"}).json()["id"])
    e = client.post("/api/edit", json={"image_id": j["result_ids"][0], "mode": "variation", "prompt": "sunset",
                                       "loras": [{"name": "Lenovo UltraReal", "strength": 0.5}]}).json()
    assert wait_job(client, e["id"])["status"] == "done"
    # rescanning the folder registers stray files
    (lora_env.loras.dir / "stray.safetensors").write_bytes(SDXL_LORA)
    added = client.post("/api/loras/scan").json()["added"]
    assert [a["file"] for a in added] == ["stray.safetensors"]
    assert client.patch("/api/loras/stray", json={"triggers": ["str4y"], "strength": 0.6}).json()["triggers"] == ["str4y"]
