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
