"""References, characters, canvas expansion and export."""
import base64
import io
import json

import pytest
from PIL import Image

from genstudio.jobs import outpaint_pads
from test_api import wait_job
from test_mcp import rpc


def make_png(w=512, h=768, color=(200, 180, 160)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def imported(client, w=512, h=768, name="director.png"):
    return client.post("/api/import", files={"file": (name, make_png(w, h), "image/png")}).json()["id"]


def last_graph(fake):
    return list(fake.prompts.values())[-1]


def nodes(g, cls):
    return [n["inputs"] for n in g.values() if n["class_type"] == cls]


def test_outpaint_pads_math():
    p = outpaint_pads(512, 512, "3:4", "top")            # taller, keep picture at the top
    assert p == {"left": 0, "top": 0, "right": 0, "bottom": 688 - 512}
    p = outpaint_pads(768, 1024, "16:9", "center")        # wider, split both sides
    assert p["left"] + p["right"] == 1824 - 768 and abs(p["left"] - p["right"]) <= 16
    assert (768 + p["left"] + p["right"]) % 16 == 0
    with pytest.raises(ValueError):
        outpaint_pads(512, 512, "1:1")
    with pytest.raises(ValueError):
        outpaint_pads(512, 512, "7:3")


def test_character_generation_uses_qwen_references(client, fake):
    ref = imported(client)
    c = client.post("/api/characters", json={"name": "Flight Director", "ref_ids": [ref],
                                             "description": "silver hair, white officer shirt with epaulets"}).json()
    assert c["id"] == "flight-director"
    assert client.post("/api/characters", json={"name": "Flight Director", "ref_ids": [ref]}).status_code == 400

    j = client.post("/api/generate", json={"prompt": "explaining the cargo bay", "character": "Flight Director",
                                           "aspect": "portrait"}).json()
    assert j["decision"]["model_id"] == "qwen-image-21-turbo" and j["decision"]["ref_ids"] == [ref]
    j = wait_job(client, j["id"])
    assert j["status"] == "done", j["error"]
    g = last_graph(fake)
    enc = nodes(g, "TextEncodeQwenImage21")[0]
    assert "images.image_1" in enc and enc["resolution"] == 0
    assert "Flight Director" in enc["prompt"] and "silver hair" in enc["prompt"]
    assert nodes(g, "EmptyLatentImage")[0]["width"] == 832          # requested size, not the reference size
    rec = client.get(f"/api/images/{j['result_ids'][0]}").json()
    assert rec["character"] == "Flight Director"

    assert client.post("/api/generate", json={"prompt": "x", "character": "nobody"}).status_code == 404
    # regenerate keeps the character
    r = wait_job(client, client.post(f"/api/images/{rec['id']}/regenerate").json()["id"])
    assert client.get(f"/api/images/{r['result_ids'][0]}").json()["character"] == "Flight Director"


def test_instruction_edit_with_extra_reference(client, fake):
    src = imported(client, 640, 960)
    outfit = imported(client, 512, 512, "jacket.png")
    j = client.post("/api/edit", json={"image_id": src, "mode": "reference", "prompt": "dress her in the jacket from image 2",
                                       "references": [outfit]}).json()
    j = wait_job(client, j["id"])
    assert j["status"] == "done", j["error"]
    g = last_graph(fake)
    enc = nodes(g, "TextEncodeQwenImage21")[0]
    assert {"images.image_1", "images.image_2"} <= set(enc)
    assert "EmptyLatentImage" not in [n["class_type"] for n in g.values()]   # output keeps the source size
    rec = client.get(f"/api/images/{j['result_ids'][0]}").json()
    assert rec["parent_id"] == src and rec["label"].startswith("참고 편집")
    assert client.post("/api/edit", json={"image_id": src, "mode": "reference"}).status_code == 400


def test_outpaint_edit(client, fake):
    src = imported(client, 512, 512)
    j = wait_job(client, client.post("/api/edit", json={"image_id": src, "mode": "outpaint", "aspect": "3:4",
                                                         "anchor": "top", "prompt": "show her down to the thighs"}).json()["id"])
    assert j["status"] == "done", j["error"]
    pad = nodes(last_graph(fake), "ImagePadForOutpaint")[0]
    assert (pad["top"], pad["bottom"]) == (0, 176)
    assert j["decision"]["pads"]["bottom"] == 176


def test_export(client, tmp_path):
    src = imported(client, 800, 1200)
    r = client.post(f"/api/images/{src}/export", json={"path": str(tmp_path / "assets" / "partner"), "format": "webp",
                                                       "max_side": 600}).json()
    assert r["path"].endswith("partner.webp") and (r["width"], r["height"]) == (400, 600)
    with Image.open(r["path"]) as im:
        assert im.format == "WEBP"
    again = client.post(f"/api/images/{src}/export", json={"path": r["path"]})
    assert again.status_code == 400
    r2 = client.post(f"/api/images/{src}/export", json={"path": r["path"], "overwrite": True, "width": 300}).json()
    assert (r2["width"], r2["height"]) == (300, 450)


def test_mcp_character_flow_and_export(client, tmp_path):
    p = tmp_path / "director.png"
    p.write_bytes(make_png())
    res = rpc(client, "tools/call", {"name": "create_character", "arguments": {"name": "Director", "image_paths": [str(p)]}})
    assert json.loads(res["content"][0]["text"])["id"] == "director"
    res = rpc(client, "tools/call", {"name": "generate_image", "arguments": {
        "prompt": "standing at the hangar door", "character": "Director", "wait_seconds": 10}})
    info = json.loads(res["content"][0]["text"])
    assert info["status"] == "done" and info["decision"]["character"] == "Director"
    img = info["images"][0]["image_id"]
    res = rpc(client, "tools/call", {"name": "edit_image", "arguments": {
        "image_id": img, "expand_aspect": "9:16", "expand_anchor": "top", "wait_seconds": 10, "include_image": False}})
    assert json.loads(res["content"][0]["text"])["status"] == "done"
    out = tmp_path / "out" / "director-tutorial.webp"
    res = rpc(client, "tools/call", {"name": "export_image", "arguments": {"image_id": img, "path": str(out), "max_side": 512}})
    assert out.exists() and json.loads(res["content"][0]["text"])["format"] == "webp"


def test_character_loose_names_and_aliases(tmp_path):
    from genstudio.characters import Characters
    cs = Characters(tmp_path / "c.json")
    cs.save("플라이트 디렉터", ["a"], aliases=["director"])
    cs.save("밀수 브로커", ["b"])
    assert cs.get("디렉터")["name"] == "플라이트 디렉터"
    assert cs.get("Director")["name"] == "플라이트 디렉터"
    assert cs.get("플라이트디렉터")["name"] == "플라이트 디렉터"
    assert cs.get("브로커")["name"] == "밀수 브로커"
    assert cs.get("항해사") is None
    assert Characters(tmp_path / "c.json").get("director")["aliases"] == ["director"]   # persisted


def test_recommendations_per_task(client):
    r = client.post("/api/recommend", json={"task": "generate", "prompt": '네온 간판 "OPEN"'}).json()
    assert r["recommended"] == "qwen-image-21-turbo" and r["models"][0]["recommended"]
    assert all(m["best_for"] for m in r["models"])
    r = client.post("/api/recommend", json={"task": "instruction"}).json()
    assert [m["id"] for m in r["models"] if m["compatible"]] == ["qwen-image-21-turbo"]
    j = wait_job(client, client.post("/api/generate", json={"prompt": "반실사 여자"}).json()["id"])
    r = client.post("/api/recommend", json={"task": "inpaint", "image_id": j["result_ids"][0]}).json()
    assert r["recommended"] == "realism-illustrious" and "그림체" in r["reason"]
