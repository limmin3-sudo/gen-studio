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
    assert {s["label"] for s in segs} == {"person", "face", "background"}


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


def test_mcp_apps_panel(client):
    init = rpc(client, "initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "t", "version": "1"},
                                      "capabilities": {"extensions": {"io.modelcontextprotocol/ui": {
                                          "mimeTypes": ["text/html;profile=mcp-app"]}}}})
    assert init["serverInfo"]["name"] == "gen-studio"
    tools = {t["name"]: t for t in rpc(client, "tools/list")["tools"]}
    assert tools["open_editor"]["_meta"]["ui"]["resourceUri"] == "ui://gen-studio/panel.html"
    assert tools["panel_state"]["_meta"]["ui"]["visibility"] == ["app"]
    res = rpc(client, "resources/read", {"uri": "ui://gen-studio/panel.html"})
    c = res["contents"][0]
    assert c["mimeType"] == "text/html;profile=mcp-app" and "ui/initialize" in c["text"]

    img = imported(client, 512, 768)
    # before segments: open_editor still answers with a preview and the editor link
    res = rpc(client, "tools/call", {"name": "open_editor", "arguments": {"image_id": img}})
    info = json.loads(res["content"][0]["text"])
    assert info["editor_url"].endswith(f"#edit={img}") and any(x["type"] == "image" for x in res["content"])
    rpc(client, "tools/call", {"name": "segment_image", "arguments": {"image_id": img, "wait_seconds": 10}})
    st = json.loads(rpc(client, "tools/call", {"name": "panel_state", "arguments": {"image_id": img}})["content"][0]["text"])
    assert st["image"].startswith("data:image/jpeg;base64,") and len(st["segments"]) == 6
    assert st["segments"][0]["mask"].startswith("data:image/png;base64,") and st["segments"][0]["thumb"]
    assert st["versions"][0]["id"] == img


def test_masked_edits_keep_the_person(client, fake):
    from genstudio.jobs import qwen_mask_instruction
    # Qwen source: instruction edit with the picture and the mask, pasted back inside the mask only
    j = wait_job(client, client.post("/api/generate", json={"prompt": '포스터 "SALE"'}).json()["id"])
    qimg = j["result_ids"][0]
    segment(client, qimg)
    e = wait_job(client, client.post("/api/edit", json={"image_id": qimg, "segments": ["woman"],
                                                         "prompt": "허벅지를 더 탄탄하게"}).json()["id"])
    assert e["status"] == "done", e["error"]
    g = [x for x in fake.prompts.values() if nodes(x, "TextEncodeQwenImage21")][-1]
    enc = nodes(g, "TextEncodeQwenImage21")[0]
    # the Korean request reaches Qwen as a plain English instruction (fake Qwen3-VL answers with a fixed sentence)
    assert {"images.image_1", "images.image_2"} <= set(enc) and "A woman sitting by a window" in enc["prompt"]
    assert "허벅지" not in enc["prompt"]
    assert "the woman" in enc["prompt"] and "Keep the same person" in enc["prompt"]
    assert "SetLatentNoiseMask" not in [n["class_type"] for n in g.values()]
    assert nodes(g, "ImageCompositeMasked")[0]["resize_source"] is True
    # the fake hands the picture back unchanged: the area is redrawn once with the helper model
    assert e["result"]["fallback"] and nodes(last_graph(fake), "SetLatentNoiseMask")
    assert client.get(f"/api/images/{e['result_ids'][0]}").json()["model_id"] == "cyberrealistic-zit"
    assert "Remove the epaulets" in qwen_mask_instruction("", "clear", ["epaulets"])

    # other models: a modification keeps the original (denoise 0.6, original prompt + change)
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo of a woman in a cafe"}).json()["id"])
    zimg = j["result_ids"][0]
    segment(client, zimg)
    wait_job(client, client.post("/api/edit", json={"image_id": zimg, "segments": ["woman"], "prompt": "toned thighs, slightly firmer"}).json()["id"])
    g = last_graph(fake)
    assert nodes(g, "KSampler")[0]["denoise"] == 0.6
    assert nodes(g, "CLIPTextEncode")[0]["text"].startswith("photo of a woman in a cafe, toned thighs")
    # a new thing in place of the old one is still drawn fresh
    wait_job(client, client.post("/api/edit", json={"image_id": zimg, "segments": ["white collared shirt"], "prompt": "red leather jacket"}).json()["id"])
    assert nodes(last_graph(fake), "KSampler")[0]["denoise"] == 1.0


def test_bigger_requests_widen_the_mask(client, fake):
    from genstudio.jobs import mask_grow
    assert mask_grow("볼륨 키워줘", 1024, 1820) == 81 and mask_grow("빨간색으로", 1024, 1024) == 12
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo of a woman"}).json()["id"])
    img = j["result_ids"][0]
    segment(client, img)
    wait_job(client, client.post("/api/edit", json={"image_id": img, "segments": ["white collared shirt"], "prompt": "가슴 볼륨 키워줘"}).json()["id"])
    assert nodes(last_graph(fake), "GrowMask")[0]["expand"] == 81


def test_library_restarts_with_segment_files(client, settings):
    from genstudio.library import Library
    img = imported(client, 300, 400)
    segment(client, img)
    lib = Library(settings.output_dir)          # what happens on the next start
    assert lib.get(img) is not None and all(isinstance(r, dict) for r in lib.list())


def test_find_by_korean_word(client, fake):
    from genstudio.segments import korean_to_label
    assert korean_to_label("스타킹") == "stockings" and korean_to_label("검은 스타킹") is None
    img = imported(client, 512, 768)
    segs = segment(client, img, labels=["견장"])
    assert [s["label"] for s in segs] == ["epaulets", "epaulets"] and segs[0]["ko"] == "견장"


def test_left_and_right_parts_are_merged():
    txt = json.dumps([{"label": "left arm", "ko": "왼팔"}, {"label": "right arm", "ko": "오른팔"},
                      {"label": "left leg", "ko": "왼쪽 다리"}, {"label": "right foot", "ko": "오른발"},
                      {"label": "left foot", "ko": "왼발"}, {"label": "blonde hair", "ko": "금발"}])
    assert [(x["label"], x["ko"]) for x in parse_labels(txt)] == [
        ("arms", "팔"), ("legs", "다리"), ("feet", "발"), ("blonde hair", "금발")]
