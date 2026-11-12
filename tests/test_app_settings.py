"""Settings screen API: local only, remote toggle, Codex registration, model list, QR."""
from genstudio import app_routes, runtime


def test_settings_api_is_local_only(client):
    assert client.get("/api/app/info").status_code == 200
    r = client.get("/api/app/info", headers={"host": "home-pc.tail1234.ts.net", "x-forwarded-for": "100.64.0.9"})
    assert r.status_code == 403


def test_info_remote_codex_and_models(client, settings, monkeypatch, tmp_path):
    monkeypatch.setattr("genstudio.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(app_routes, "CODEX_CONFIG", tmp_path / ".codex" / "config.toml")
    monkeypatch.setattr(runtime, "tailscale_name", lambda: "home-pc.tail1234.ts.net")
    monkeypatch.setattr(runtime, "tailscale_exe", lambda: "tailscale")
    calls = []
    monkeypatch.setattr(runtime, "tailscale_serve", lambda port, on: (calls.append((port, on)), (True, ""))[1])
    info = client.get("/api/app/info").json()
    assert info["remote"]["on"] is False and info["codex"]["connected"] is False
    assert info["comfy"]["alive"] is True and isinstance(info["models"]["missing"], list)

    r = client.post("/api/app/remote", json={"on": True}).json()
    assert r["on"] is True and r["link"].startswith("https://home-pc.tail1234.ts.net/?token=") and calls == [(7860, True)]
    assert runtime.prefs()["remote_access"] is True and settings.remote_access is True
    qr = client.get("/api/app/remote/qr")
    assert qr.status_code == 200 and qr.headers["content-type"] == "image/png"
    client.post("/api/app/remote", json={"on": False})
    assert settings.remote_access is False and calls[-1] == (7860, False)
    assert client.get("/api/app/remote/qr").status_code == 404

    c = client.post("/api/app/codex").json()
    assert c["connected"] and "[mcp_servers.gen_studio]" in (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert client.get("/api/app/info").json()["codex"]["connected"] is True


def test_missing_model_list_and_civitai_key(client, settings, monkeypatch, tmp_path):
    monkeypatch.setattr("genstudio.config.SECRETS_FILE", tmp_path / "secrets.json")
    settings.comfy_path = tmp_path / "ComfyUI"
    missing = client.get("/api/app/info").json()["models"]["missing"]
    names = {m["file"] for m in missing}
    assert "birefnet.safetensors" in names and "RealESRGAN_x4plus.safetensors" in names
    assert len(names) == len(missing)                       # each file once
    assert client.post("/api/app/civitai-key", json={"key": "short"}).status_code == 400
    assert client.post("/api/app/civitai-key", json={"key": "abcdef0123456789"}).json()["ok"]


def test_unused_model_files(client, settings, tmp_path):
    settings.comfy_path = tmp_path / "ComfyUI"
    md = settings.comfy_path / "models"
    (md / "diffusion_models").mkdir(parents=True)
    (md / "loras").mkdir()
    (md / "diffusion_models" / "old_model.safetensors").write_bytes(b"x" * 100)
    (md / "diffusion_models" / "half.safetensors.part").write_bytes(b"x" * 10)
    (md / "loras" / "mine.safetensors").write_bytes(b"x")                       # LoRAs are not touched here
    keep = md / "vae" / "ae.safetensors"; keep.parent.mkdir(); keep.write_bytes(b"x")   # used by Z-Image
    r = client.get("/api/app/models/unused").json()
    assert [f["path"] for f in r["files"]] == ["diffusion_models/old_model.safetensors", "diffusion_models/half.safetensors.part"]
    assert r["bytes"] == 110
    d = client.post("/api/app/models/unused/delete", json={"paths": ["diffusion_models/old_model.safetensors", "vae/ae.safetensors", "../../etc/passwd"]}).json()
    assert d["deleted"] == 1 and keep.exists() and not (md / "diffusion_models" / "old_model.safetensors").exists()


def test_add_model_from_civitai_link(client, settings, monkeypatch, tmp_path):
    import httpx
    monkeypatch.setattr("genstudio.config.DATA_DIR", tmp_path)

    def civitai(req: httpx.Request):
        u = str(req.url)
        if u.endswith("/model-versions/2728617"):
            return httpx.Response(200, json={"id": 2728617, "modelId": 1025051, "name": "v21", "baseModel": "Illustrious",
                                             "model": {"name": "Illustrij", "type": "Checkpoint"},
                                             "files": [{"name": "Illustrijv21.safetensors", "type": "Model", "primary": True,
                                                        "sizeKB": 7245902.25, "hashes": {"SHA256": "AB"}, "metadata": {"fp": "fp16"}}]})
        if u.endswith("/model-versions/9"):
            return httpx.Response(200, json={"id": 9, "modelId": 8, "name": "v1", "baseModel": "Flux.1 D",
                                             "model": {"name": "X", "type": "Checkpoint"}, "files": []})
        if u.endswith("/model-versions/7"):
            return httpx.Response(200, json={"id": 7, "modelId": 6, "name": "Turbo", "baseModel": "Qwen 2.1",
                                             "model": {"name": "QQ", "type": "Checkpoint"},
                                             "files": [{"name": "q.gguf", "type": "Model", "metadata": {}},
                                                       {"name": "q_bf16.safetensors", "type": "Diffusion Model", "sizeKB": 13e6, "metadata": {"fp": "bf16"}},
                                                       {"name": "q_fp8.safetensors", "type": "Diffusion Model", "sizeKB": 7e6, "metadata": {"fp": "fp8"}, "primary": False}]})
        return httpx.Response(404, json={})
    client.app.state.gs.extra["civitai_transport"] = httpx.MockTransport(civitai)

    r = client.post("/api/app/models/add", json={"url": "https://civitai.red/models/1025051/illustrij?modelVersionId=2728617"}).json()
    m = r["model"]
    assert m["family"] == "illustrious" and m["style"] == "semi" and m["file"]["dir"] == "checkpoints"
    assert any(x["id"] == m["id"] for x in client.get("/api/models").json())
    assert client.post("/api/route", json={"prompt": "x", "model": m["id"]}).json()["model_id"] == m["id"]
    q = client.post("/api/app/models/add", json={"url": "https://civitai.com/models/6?modelVersionId=7"}).json()["model"]
    assert q["family"] == "qwen21" and q["file"]["name"] == "q_fp8.safetensors" and "text" in q["also"]
    bad = client.post("/api/app/models/add", json={"url": "https://civitai.com/models/8?modelVersionId=9"})
    assert bad.status_code == 400 and "Flux" in bad.json()["detail"]
    assert [x["id"] for x in client.get("/api/app/models/added").json()["models"]] == [m["id"], q["id"]]
    assert client.delete(f"/api/app/models/added/{m['id']}").json()["removed"]
    assert all(x["id"] != m["id"] for x in client.get("/api/models").json())


def test_clear_all_pictures(client, settings, fake, tmp_path):
    from test_api import mask_png, wait_job
    settings.comfy_path = tmp_path / "ComfyUI"
    out = settings.comfy_path / "output" / "gen-studio" / "img"; out.mkdir(parents=True)
    (out / "x_00001_.png").write_bytes(b"x" * 50)
    inp = settings.comfy_path / "input" / "gen-studio"; inp.mkdir(parents=True)
    (inp / "src.png").write_bytes(b"y" * 30)
    other = settings.comfy_path / "output" / "my_own"; other.mkdir()
    (other / "keep.png").write_bytes(b"z")                                   # not ours: stays
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo"}).json()["id"])
    img = j["result_ids"][0]
    wait_job(client, client.post("/api/edit", json={"image_id": img, "mode": "variation", "prompt": "x"}).json()["id"])
    client.post("/api/characters", json={"name": "A", "ref_ids": [img]})
    lib = client.get("/api/app/library").json()
    assert lib["count"] == 2 and lib["comfy_bytes"] == 80 and lib["characters"] == 1
    assert client.post("/api/app/library/clear", json={}).status_code == 400          # needs the confirmation
    r = client.post("/api/app/library/clear", json={"confirm": "모두 지우기"}).json()
    assert r["deleted"] == 2 and r["characters"] == 1 and r["bytes"] >= 80
    assert client.get("/api/images").json() == [] and client.get("/api/characters").json() == []
    assert not (settings.comfy_path / "output" / "gen-studio").exists() and not inp.exists()
    assert (other / "keep.png").exists()
    assert not [f for f in settings.output_dir.rglob("*") if f.is_file() and f.suffix in (".png", ".jpg", ".mp4")]
    # works again afterwards
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo"}).json()["id"])
    assert j["status"] == "done" and client.get(f"/api/images/{j['result_ids'][0]}/thumb").status_code == 200


def test_ltx_video_for_codex(client, settings, fake, monkeypatch, tmp_path):
    """Optional download; once the files are there Codex's create_video makes an mp4 with sound + a poster."""
    from test_api import wait_job
    from genstudio.config import CONFIG_DIR
    from genstudio.registry import Registry
    monkeypatch.setattr("genstudio.config.SECRETS_FILE", tmp_path / "secrets.json")
    reg = Registry.load(CONFIG_DIR / "models.yaml")
    eng = client.app.state.gs.engine
    eng.registry.tools["video_ltx"] = reg.tools["video_ltx"]
    eng.registry.optional_tools = reg.optional_tools
    settings.comfy_path = tmp_path / "ComfyUI"
    r = client.get("/api/app/ltx").json()
    assert r["ready"] is False and r["hf_token"] is False and len(r["files"]) == 5
    assert client.post("/api/app/hf-token", json={"token": "nope"}).status_code == 400
    assert client.post("/api/app/hf-token", json={"token": "hf_" + "x" * 30}).json()["ok"]
    assert client.get("/api/app/ltx").json()["hf_token"] is True
    # not part of the normal "모두 받기" list
    assert not any("ltx-2.5" in m["file"] for m in client.get("/api/app/info").json()["models"]["missing"])
    assert client.post("/api/ltx", json={"prompt": "x"}).status_code == 400            # files not there yet
    for f in reg.tools["video_ltx"]:
        f.path(settings.models_dir).parent.mkdir(parents=True, exist_ok=True)
        f.path(settings.models_dir).write_bytes(b"x")
    assert client.get("/api/app/ltx").json()["ready"] is True

    j = wait_job(client, client.post("/api/ltx", json={"prompt": 'A rainy neon street. She says "let us go".',
                                                        "seconds": 4, "orientation": "portrait"}).json()["id"])
    assert j["status"] == "done", j["error"]
    assert (j["decision"]["width"], j["decision"]["height"]) == (704, 1280) and j["decision"]["frames"] == 97
    g = list(fake.prompts.values())[-1]
    kinds = [n["class_type"] for n in g.values()]
    assert kinds.count("SamplerCustomAdvanced") == 2 and "LTXVLatentUpsampler" in kinds and "LTXVAudioVAEDecode" in kinds
    assert "LTXVImgToVideoInplace" not in kinds                                           # text only
    empty = next(n for n in g.values() if n["class_type"] == "EmptyLTXVLatentVideo")["inputs"]
    assert (empty["width"], empty["height"], empty["length"]) == (352, 640, 97)
    rec = client.get(f"/api/images/{j['result_ids'][0]}").json()
    assert rec["media"] == "video" and rec["audio"] is True and rec["label"] == "LTX 영상 4초" and rec["parent_id"] is None
    assert client.get(f"/api/images/{rec['id']}/thumb").status_code == 200

    # from a picture: it is put back in both passes; regenerate repeats the same request
    img = wait_job(client, client.post("/api/generate", json={"prompt": "photo", "aspect": "landscape"}).json()["id"])["result_ids"][0]
    j2 = wait_job(client, client.post("/api/ltx", json={"prompt": "she waves", "image_id": img}).json()["id"])
    g = list(fake.prompts.values())[-1]
    assert [n["inputs"]["strength"] for n in g.values() if n["class_type"] == "LTXVImgToVideoInplace"] == [0.7, 1.0]
    r2 = wait_job(client, client.post(f"/api/images/{j2['result_ids'][0]}/regenerate").json()["id"])
    assert r2["kind"] == "ltx" and r2["status"] == "done"


def test_ltx_keyframes_and_knobs(client, settings, fake, tmp_path):
    """Start / middle / end / any second: ComfyUI's First & Last Frame recipe (one pass, LTXVAddGuide per picture)."""
    from test_api import wait_job
    from genstudio.config import CONFIG_DIR
    from genstudio.registry import Registry
    reg = Registry.load(CONFIG_DIR / "models.yaml")
    eng = client.app.state.gs.engine
    eng.registry.tools["video_ltx"] = reg.tools["video_ltx"]
    settings.comfy_path = tmp_path / "ComfyUI"
    for f in reg.tools["video_ltx"]:
        f.path(settings.models_dir).parent.mkdir(parents=True, exist_ok=True)
        f.path(settings.models_dir).write_bytes(b"x")
    ids = [wait_job(client, client.post("/api/generate", json={"prompt": "photo", "aspect": "landscape"}).json()["id"])
           ["result_ids"][0] for _ in range(4)]
    j = wait_job(client, client.post("/api/ltx", json={
        "prompt": "she walks to the window", "image_id": ids[0], "mid_image_id": ids[1], "end_image_id": ids[2],
        "keyframes": [{"image_id": ids[3], "at": 1.5}], "seconds": 6, "fps": 30, "image_strength": 0.5,
        "audio": False, "width": 1000}).json()["id"])
    assert j["status"] == "done", j["error"]
    d = j["decision"]
    assert d["mode"] == "keyframes" and d["fps"] == 30 and d["frames"] == 181 and d["width"] == 1024
    assert [(x["image_id"], x["frame"]) for x in d["guides"]] == [(ids[0], 0), (ids[3], 45), (ids[1], 90), (ids[2], -1)]
    g = list(fake.prompts.values())[-1]
    kinds = [n["class_type"] for n in g.values()]
    assert kinds.count("SamplerCustomAdvanced") == 1 and "LTXVCropGuides" in kinds and "LTXVLatentUpsampler" not in kinds
    assert "LTXVAudioVAEDecode" not in kinds and "audio" not in next(n for n in g.values() if n["class_type"] == "CreateVideo")["inputs"]
    guides = [n["inputs"] for n in g.values() if n["class_type"] == "LTXVAddGuide"]
    assert [x["frame_idx"] for x in guides] == [0, 45, 90, -1] and {x["strength"] for x in guides} == {0.5}
    assert next(n for n in g.values() if n["class_type"] == "LTXVConditioning")["inputs"]["frame_rate"] == 30.0
    rec = client.get(f"/api/images/{j['result_ids'][0]}").json()
    assert rec["audio"] is False and rec["fps"] == 30 and rec["duration"] == 6.0
    r = wait_job(client, client.post(f"/api/images/{rec['id']}/regenerate").json()["id"])
    assert r["status"] == "done" and r["decision"]["mode"] == "keyframes"
    # only an end picture still works (text start)
    j = wait_job(client, client.post("/api/ltx", json={"prompt": "x", "end_image_id": ids[2]}).json()["id"])
    assert j["status"] == "done" and j["decision"]["guides"] == [{"image_id": ids[2], "frame": -1, "strength": 0.7}]
    assert client.post("/api/ltx", json={"prompt": "x", "end_image_id": "nope"}).status_code == 404


def test_ltx_memory_guard_and_keyframe_strength(client, settings, fake):
    """1080p 10 s at 24 fps ran out of 16GB on the home PC: refused up front with what would fit."""
    from test_api import wait_job
    from genstudio.config import CONFIG_DIR
    from genstudio.jobs import ltx_tokens
    from genstudio.registry import Registry
    reg = Registry.load(CONFIG_DIR / "models.yaml")
    client.app.state.gs.engine.registry.tools["video_ltx"] = reg.tools["video_ltx"]
    assert ltx_tokens(1920, 1088, 6 * 24 + 1) == 38760 and ltx_tokens(1920, 1088, 10 * 24 + 1) == 63240
    img = wait_job(client, client.post("/api/generate", json={"prompt": "photo", "aspect": "landscape"}).json()["id"])["result_ids"][0]
    r = client.post("/api/ltx", json={"prompt": "x", "image_id": img, "end_image_id": img, "width": 1920, "height": 1080,
                                      "seconds": 10})
    assert r.status_code == 400 and "12fps" in r.json()["detail"] and "allow_risky" in r.json()["detail"]
    ok = client.post("/api/ltx", json={"prompt": "x", "image_id": img, "end_image_id": img, "width": 1920,
                                       "height": 1080, "seconds": 6})
    assert ok.status_code == 200 and ok.json()["decision"]["height"] == 1088
    risky = client.post("/api/ltx", json={"prompt": "x", "image_id": img, "width": 1920, "height": 1080, "seconds": 10,
                                          "allow_risky": True})
    assert risky.status_code == 200
    # per-keyframe strength
    j = wait_job(client, client.post("/api/ltx", json={"prompt": "x", "image_id": img, "end_image_id": img,
                                                        "keyframes": [{"image_id": img, "at": 2, "strength": 0.45}],
                                                        "seconds": 4}).json()["id"])
    assert j["status"] == "done", j["error"]
    g = list(fake.prompts.values())[-1]
    assert [n["inputs"]["strength"] for n in g.values() if n["class_type"] == "LTXVAddGuide"] == [0.7, 0.45, 0.7]


def test_comfy_memory_error_is_explained(client, fake):
    from test_api import wait_job
    fake.fail_with = "aimdo memory compile error"
    j = wait_job(client, client.post("/api/generate", json={"prompt": "photo"}).json()["id"])
    fake.fail_with = None
    assert j["status"] == "error" and "그래픽 메모리가 부족해요" in j["error"] and "aimdo" in j["error"]
