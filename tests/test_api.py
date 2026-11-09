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
