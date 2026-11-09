import json

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
           "MCP-Protocol-Version": "2025-06-18"}


def rpc(client, method, params=None, id_=1):
    r = client.post("/mcp", headers=HEADERS, json={"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body
    return body["result"]


def test_mcp_initialize_and_list_tools(client):
    res = rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "test", "version": "1"}})
    assert res["serverInfo"]["name"] == "gen-studio"
    tools = {t["name"] for t in rpc(client, "tools/list")["tools"]}
    assert tools == {"generate_image", "edit_image", "get_job_status", "get_image", "list_images", "list_models"}


def test_mcp_generate_returns_image_then_edit(client):
    res = rpc(client, "tools/call", {"name": "generate_image",
                                     "arguments": {"prompt": "photo of a cat", "wait_seconds": 10}})
    kinds = [c["type"] for c in res["content"]]
    assert "image" in kinds, res
    info = json.loads(next(c["text"] for c in res["content"] if c["type"] == "text"))
    assert info["status"] == "done"
    image_id = info["images"][0]["image_id"]

    res = rpc(client, "tools/call", {"name": "edit_image",
                                     "arguments": {"image_id": image_id, "prompt": "make it night", "denoise": 0.5,
                                                   "wait_seconds": 10, "include_image": False}})
    info = json.loads(res["content"][0]["text"])
    assert info["status"] == "done"
    assert info["images"][0]["parent_id"] == image_id

    res = rpc(client, "tools/call", {"name": "list_models", "arguments": {}})
    assert "cyberrealistic-zit" in json.dumps(res)


def test_mcp_path_has_no_redirect(client):
    r = client.post("/mcp", headers=HEADERS, follow_redirects=False,
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert r.status_code == 200, (r.status_code, r.headers.get("location"))
