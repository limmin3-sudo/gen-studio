import base64
import io
import time

from PIL import Image


def wait_job(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "canceled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def mask_png(w, h, box):
    im = Image.new("L", (w, h), 0)
    im.paste(255, box)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_status_and_models(client):
    s = client.get("/api/status").json()
    assert s["comfyui"]["connected"] is True
    assert "RTX 4080" in s["comfyui"]["devices"][0]
    models = client.get("/api/models").json()
    assert {m["id"] for m in models} == {"cyberrealistic-zit", "realism-illustrious", "qwen-image-21-turbo", "wan22-14b"}


def test_route_preview(client):
    d = client.post("/api/route", json={"prompt": "반실사 세로 전신"}).json()
    assert d["style"] == "semi" and d["aspect"] == "portrait" and (d["width"], d["height"]) == (832, 1216)


def test_generate_edit_and_versions(client, fake):
    job = client.post("/api/generate", json={"prompt": "photo of a woman in a white shirt", "aspect": "portrait"}).json()
    assert job["decision"]["model_id"] == "cyberrealistic-zit"
    job = wait_job(client, job["id"])
    assert job["status"] == "done", job["error"]
    img_id = job["result_ids"][0]
    rec = client.get(f"/api/images/{img_id}").json()
    assert (rec["width"], rec["height"]) == (832, 1216)
    assert rec["model_id"] == "cyberrealistic-zit" and rec["label"] == "원본"
    assert client.get(f"/api/images/{img_id}/file").headers["content-type"] == "image/png"
    assert client.get(f"/api/images/{img_id}/thumb").status_code == 200

    # inpaint with a prompt
    e = client.post("/api/edit", json={"image_id": img_id, "prompt": "red ribbon", "fill": "prompt",
                                       "mask_png": mask_png(832, 1216, (100, 100, 300, 300))}).json()
    e = wait_job(client, e["id"])
    assert e["status"] == "done", e["error"]
    graph = list(fake.prompts.values())[-1]
    assert any(n["class_type"] == "ImageCompositeMasked" for n in graph.values())
    assert any(n["inputs"].get("text") == "red ribbon" for n in graph.values() if n["class_type"] == "CLIPTextEncode")

    # auto fill reuses the original prompt
    e2 = client.post("/api/edit", json={"image_id": e["result_ids"][0], "fill": "auto",
                                        "mask_png": mask_png(832, 1216, (0, 0, 50, 50))}).json()
    e2 = wait_job(client, e2["id"])
    graph = list(fake.prompts.values())[-1]
    assert any("white shirt" in n["inputs"].get("text", "") for n in graph.values() if n["class_type"] == "CLIPTextEncode")
    assert e2["decision"]["action"] == "모델이 알아서 채우기"

    versions = client.get(f"/api/images/{img_id}").json()["versions"]
    assert [v["label"] for v in versions] == ["원본", "부분 다시 그리기: red ribbon", "모델이 알아서 채우기"]
    assert {v["root_id"] for v in versions} == {img_id}


def test_switching_family_frees_vram(client, fake):
    j1 = client.post("/api/generate", json={"prompt": "photo"}).json()
    wait_job(client, j1["id"])
    j2 = client.post("/api/generate", json={"prompt": "반실사"}).json()
    wait_job(client, j2["id"])
    assert fake.freed == 1


def test_edit_validation(client):
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo"}).json()["id"])
    img = j["result_ids"][0]
    r = client.post("/api/edit", json={"image_id": img, "prompt": "x", "mode": "inpaint"})
    assert r.status_code == 400 and "마스크" in r.json()["detail"]
    r = client.post("/api/edit", json={"image_id": img, "fill": "prompt", "mask_png": mask_png(1024, 1024, (0, 0, 9, 9))})
    assert r.status_code == 400
    assert client.post("/api/edit", json={"image_id": "nope", "mode": "variation"}).status_code == 404


def test_import_resizes_and_reads_metadata(client):
    from PIL import PngImagePlugin
    im = Image.new("RGB", (3000, 2000), (10, 20, 30))
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a ship at sea, masterpiece\nNegative prompt: blurry\nSteps: 20")
    buf = io.BytesIO()
    im.save(buf, "PNG", pnginfo=info)
    rec = client.post("/api/import", files={"file": ("ship.png", buf.getvalue(), "image/png")}).json()
    assert rec["width"] % 16 == 0 and rec["height"] % 16 == 0
    assert rec["width"] * rec["height"] <= 2_100_000
    assert rec["embedded"]["tool"] == "A1111/Forge"
    assert rec["prompt"] == "a ship at sea, masterpiece"


def test_comfy_error_is_reported(settings, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from fake_comfy import FakeComfy
    from genstudio.app import create_app
    monkeypatch.setattr("genstudio.config.SECRETS_FILE", tmp_path / "secrets.json")
    app = create_app(settings, comfy_transport=FakeComfy(fail_with="CUDA out of memory").transport())
    with TestClient(app) as c:
        j = wait_job(c, c.post("/api/generate", json={"prompt": "x"}).json()["id"])
        assert j["status"] == "error" and "out of memory" in j["error"]


def test_remote_requests_need_token(settings, fake, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from genstudio.app import create_app
    monkeypatch.setattr("genstudio.config.SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr("genstudio.config.DATA_DIR", tmp_path)
    monkeypatch.setenv("GENSTUDIO_TOKEN", "s3cret")
    app = create_app(settings, comfy_transport=fake.transport())
    with TestClient(app, client=("100.64.0.5", 5000)) as c:
        assert c.get("/api/status").status_code == 403          # remote access off
        settings.remote_access = True
        assert c.get("/api/status").status_code == 401          # no token
        assert c.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/api/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_regenerate_generate_and_edit(client, fake):
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo of a ship", "aspect": "landscape"}).json()["id"])
    img = j["result_ids"][0]
    r = wait_job(client, client.post(f"/api/images/{img}/regenerate").json()["id"])
    rec = client.get(f"/api/images/{r['result_ids'][0]}").json()
    assert rec["prompt"] == "photo of a ship" and (rec["width"], rec["height"]) == (1216, 832)

    e = wait_job(client, client.post("/api/edit", json={"image_id": img, "prompt": "flag", "mask_png": mask_png(1216, 832, (0, 0, 64, 64))}).json()["id"])
    edited = client.get(f"/api/images/{e['result_ids'][0]}").json()
    assert edited["mask_file"].endswith(".mask.png")
    r2 = wait_job(client, client.post(f"/api/images/{edited['id']}/regenerate").json()["id"])
    again = client.get(f"/api/images/{r2['result_ids'][0]}").json()
    assert again["parent_id"] == img and again["label"] == edited["label"]

    imp = client.post("/api/import", files={"file": ("a.png", io.BytesIO(base64.b64decode(mask_png(64, 64, (0, 0, 8, 8)))), "image/png")}).json()
    assert client.post(f"/api/images/{imp['id']}/regenerate").status_code == 400


def test_proxied_and_cross_site_requests(client):
    # Tailscale Serve forwards from 127.0.0.1 but with a ts.net Host and proxy headers: treat as remote
    r = client.get("/api/status", headers={"host": "home-pc.tail1234.ts.net", "x-forwarded-for": "100.64.0.9"})
    assert r.status_code == 403
    # another website trying to post into the local app
    r = client.post("/api/generate", json={"prompt": "x"}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    # the app's own page posting is fine
    r = client.post("/api/route", json={"prompt": "x"}, headers={"origin": "http://testserver"})
    assert r.status_code == 200


def test_token_link_sets_cookie_and_redirects_relative(settings, fake, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from genstudio.app import create_app
    monkeypatch.setattr("genstudio.config.SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setenv("GENSTUDIO_TOKEN", "s3cret")
    settings.remote_access = True
    app = create_app(settings, comfy_transport=fake.transport())
    with TestClient(app) as c:
        h = {"host": "home-pc.tail1234.ts.net", "x-forwarded-for": "100.64.0.9", "x-forwarded-proto": "https"}
        r = c.get("/?token=s3cret", headers=h, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/"
        assert "gs_token=s3cret" in r.headers["set-cookie"] and "Secure" in r.headers["set-cookie"]
        assert c.get("/api/status", headers={**h, "cookie": "gs_token=s3cret"}).status_code == 200


def test_deleting_versions(client):
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo"}).json()["id"])
    root = j["result_ids"][0]
    ids = [root]
    for p in ("a", "b"):
        e = wait_job(client, client.post("/api/edit", json={"image_id": ids[-1], "mode": "variation", "prompt": p}).json()["id"])
        ids.append(e["result_ids"][0])
    # removing the first version keeps the others together as one set
    assert client.delete(f"/api/images/{root}").json()["deleted"]
    vs = client.get(f"/api/images/{ids[1]}").json()["versions"]
    assert [v["id"] for v in vs] == ids[1:] and vs[0]["root_id"] == ids[1]
    assert any(r["id"] == ids[1] for r in client.get("/api/images?roots_only=true").json())
    # gallery X removes the whole set
    assert client.delete(f"/api/images/{ids[2]}?all_versions=true").json()["deleted"] == 2
    assert client.get(f"/api/images/{ids[1]}").status_code == 404


def test_one_reference_restyled_with_style_model(client, fake):
    im = Image.new("RGB", (1000, 1500), (90, 80, 70))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    ref = client.post("/api/import", files={"file": ("photo.png", buf.getvalue(), "image/png")}).json()["id"]
    d = client.post("/api/route", json={"prompt": "반실사로 만들어봐", "style": "semi", "references": [ref]}).json()
    assert d["model_id"] == "realism-illustrious" and d["restyle"] == ref
    assert d["width"] * d["height"] <= 1_150_000 and abs(d["width"] / d["height"] - 1000 / 1500) < 0.03
    j = wait_job(client, client.post("/api/generate", json={"prompt": "반실사로 만들어봐", "references": [ref]}).json()["id"])
    assert j["status"] == "done", j["error"]
    graph = list(fake.prompts.values())[-1]
    kinds = {n["class_type"] for n in graph.values()}
    assert "VAEEncode" in kinds and "TextEncodeQwenImage21" not in kinds
    ks = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert ks["inputs"]["denoise"] == 0.55
    text = next(n["inputs"]["text"] for n in graph.values() if n["class_type"] == "CLIPTextEncode")
    assert "1girl" in text and "반실사" not in text           # described in tags, not the Korean request
    rec = client.get(f"/api/images/{j['result_ids'][0]}").json()
    assert rec["label"] == "반실사 변환" and rec["ref_ids"] == [ref]
    # regenerating keeps the same path
    r = wait_job(client, client.post(f"/api/images/{rec['id']}/regenerate").json()["id"])
    assert client.get(f"/api/images/{r['result_ids'][0]}").json()["label"] == "반실사 변환"
    # a plain reference request still goes to Qwen
    d2 = client.post("/api/route", json={"prompt": "image 1 여자가 해변에 서 있는 사진", "references": [ref]}).json()
    assert d2["model_id"] == "qwen-image-21-turbo" and "restyle" not in d2


def test_remove_background(client, fake, tmp_path):
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo of a woman", "aspect": "portrait"}).json()["id"])
    img = j["result_ids"][0]
    c = wait_job(client, client.post(f"/api/images/{img}/remove-background", json={}).json()["id"])
    assert c["status"] == "done", c["error"]
    rec = client.get(f"/api/images/{c['result_ids'][0]}").json()
    assert rec["parent_id"] == img and rec["label"] == "배경 제거 (투명)" and rec["root_id"] == img
    data = client.get(f"/api/images/{rec['id']}/file").content
    with Image.open(io.BytesIO(data)) as im:
        assert im.mode == "RGBA" and im.getpixel((5, 5))[3] == 0 and im.getpixel((416, 600))[3] == 255
    graph = list(fake.prompts.values())[-1]
    assert any(n["class_type"] == "RemoveBackground" for n in graph.values())
    # white background, and regenerate repeats it
    w = wait_job(client, client.post(f"/api/images/{img}/remove-background", json={"background": "white"}).json()["id"])
    with Image.open(io.BytesIO(client.get(f"/api/images/{w['result_ids'][0]}/file").content)) as im:
        assert im.mode == "RGB" and im.getpixel((5, 5)) == (255, 255, 255)
    r = wait_job(client, client.post(f"/api/images/{rec['id']}/regenerate").json()["id"])
    assert client.get(f"/api/images/{r['result_ids'][0]}").json()["label"] == "배경 제거 (투명)"
    assert client.post(f"/api/images/{img}/remove-background", json={"background": "plaid"}).status_code == 400
