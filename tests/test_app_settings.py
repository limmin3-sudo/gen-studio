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
