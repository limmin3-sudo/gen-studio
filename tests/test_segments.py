"""Automatic segments, click selection and segment-based edits."""
import io
import json
import zipfile

from PIL import Image

from genstudio.segments import parse_labels
from test_api import wait_job
from test_mcp import rpc
from test_phase2 import imported, last_graph, nodes


def segment(client, img, **body):
    j = client.post(f"/api/images/{img}/segment", json=body).json()
    j = wait_job(client, j["id"])
    assert j["status"] == "done", j["error"]
    return client.get(f"/api/images/{img}/segments").json()


def test_parse_labels_is_forgiving():
    txt = 'Sure!\n```json\n[{"label": "Blonde Hair!", "ko": "금발", "parent": "Woman"}, "hat", {"label": "hat"}, 7]\n```'
    assert parse_labels(txt) == [{"label": "blonde hair", "ko": "금발", "parent": "woman"},
                                 {"label": "hat", "ko": "", "parent": None}]
    assert parse_labels("no json here") == []


def test_auto_segmentation_with_vlm_labels(client, fake):
    img = imported(client, 512, 768)
    segs = segment(client, img)
    labels = [s["label"] for s in segs]
    assert labels == ["woman", "blonde hair", "white collared shirt", "epaulets", "epaulets", "background"]
    ep = [s for s in segs if s["label"] == "epaulets"]
    assert [s["instance"] for s in ep] == [1, 2] and all(s["group"] == "epaulets" for s in ep)
    assert segs[1]["ko"] == "금발 머리" and segs[1]["parent"] == "woman"
    # the describe graph used Qwen3-VL as a text generator
    g0 = list(fake.prompts.values())[-2]
    gen = nodes(g0, "TextGenerateV2")[0]
    assert gen["sampling_mode"] == "off" and "images.image1" in gen
    assert nodes(g0, "CLIPLoader")[0]["clip_name"] == "qwen3vl_8b_int8_convrot.safetensors"
    det = nodes(last_graph(fake), "SAM3_Detect")
    assert len(det) == 6 and all(d["individual_masks"] for d in det)

    # cutouts and masks
    r = client.get(f"/api/images/{img}/segments/{segs[1]['id']}/cutout")
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.mode == "RGBA" and abs(im.size[0] - 102) <= 2 and abs(im.size[1] - 115) <= 2
    assert client.get(f"/api/images/{img}/segments/epaulets/mask").status_code == 200
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/images/{img}/segments.zip").content))
    assert len(z.namelist()) == 6 and any("epaulets-2" in n for n in z.namelist())


def test_labels_points_and_fallback(client, fake):
    img = imported(client, 400, 400)
    segs = segment(client, img, labels=["Epaulets"])
    assert [s["label"] for s in segs] == ["epaulets", "epaulets"]
    segs = segment(client, img, points=[{"x": 200, "y": 200}])
    assert segs[-1]["source"] == "click" and segs[-1]["ko"] == "선택한 부분"
    # a re-scan keeps clicked parts
    segs = segment(client, img, labels=["woman"])
    assert {s["source"] for s in segs} == {"click", "labels"}

    client.app.state.gs.engine.vlm_file = lambda: None      # no Qwen installed -> built-in word list
    img2 = imported(client, 300, 600)
    segs = segment(client, img2)
    assert [s["label"] for s in segs] == ["person", "background"]


def test_edit_by_segment_and_delete_by_number(client, fake):
    img = imported(client, 512, 768)
    segment(client, img)
    j = wait_job(client, client.post("/api/edit", json={"image_id": img, "segments": ["epaulets"],
                                                         "prompt": "gold star pins"}).json()["id"])
    assert j["status"] == "done", j["error"]
    assert nodes(last_graph(fake), "LoadImageMask")
    uploaded_mask = next(v for k, v in fake.uploads.items() if "-mask-" in k)
    with Image.open(io.BytesIO(uploaded_mask)) as m:
        g = m.convert("L")
        assert g.getpixel((int(512 * .40), int(768 * .315))) == 255      # left epaulet
        assert g.getpixel((int(512 * .60), int(768 * .315))) == 255      # right epaulet
        assert g.getpixel((256, 600)) == 0
    # delete the shirt (number 3 on the overview) and let the model fill it
    j = wait_job(client, client.post("/api/edit", json={"image_id": img, "segments": ["3"], "fill": "auto"}).json()["id"])
    assert j["status"] == "done" and j["decision"]["action"] == "모델이 알아서 채우기"
    assert client.post("/api/edit", json={"image_id": img, "segments": ["unicorn"], "prompt": "x"}).status_code == 404


def test_mcp_segment_then_edit(client):
    img = imported(client, 512, 768)
    res = rpc(client, "tools/call", {"name": "segment_image", "arguments": {"image_id": img, "wait_seconds": 10}})
    info = json.loads(res["content"][0]["text"])
    assert [s["number"] for s in info["segments"]] == [1, 2, 3, 4, 5, 6]
    assert any(c["type"] == "image" for c in res["content"])
    res = rpc(client, "tools/call", {"name": "edit_image", "arguments": {
        "image_id": img, "segments": ["white collared shirt"], "prompt": "red leather jacket", "wait_seconds": 10,
        "include_image": False}})
    assert json.loads(res["content"][0]["text"])["status"] == "done"
    # second call reuses stored segments without a new job
    res = rpc(client, "tools/call", {"name": "segment_image", "arguments": {"image_id": img}})
    assert len(json.loads(res["content"][0]["text"])["segments"]) == 6
