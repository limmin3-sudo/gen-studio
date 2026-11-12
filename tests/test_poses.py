"""Skeleton editor: poses made by moving joints, joints read from photos, Codex's make_pose / read_pose."""
import io

from PIL import Image

from genstudio import poses
from test_api import wait_job
from test_lineup import real  # noqa: F401 - fixture: the real model list (SDPose is in it)
from test_mcp import rpc


def _png(w, h):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (90, 90, 90)).save(b, "PNG")
    return b.getvalue()


def test_render_draws_like_comfyui():
    p = poses.preset("stand", 768, 1152)
    im = Image.open(io.BytesIO(poses.render([p], 768, 1152))).convert("RGB")
    assert im.size == (768, 1152) and im.getpixel((5, 5)) == (0, 0, 0)
    nx, ny = p[1]                                                   # neck joint: colour 1 of the DWPose palette
    assert im.getpixel((round(nx * 768), round(ny * 1152))) == poses.COLORS[1]
    # mirror swaps sides; a hidden joint is not drawn
    m = poses.mirror(p)
    assert m[2][0] == round(1 - p[5][0], 4) and m[0] == [round(1 - p[0][0], 4), p[0][1]]
    p2 = [None if i == 4 else j for i, j in enumerate(p)]
    im2 = Image.open(io.BytesIO(poses.render([p2], 768, 1152))).convert("RGB")
    wx, wy = p[4]
    assert im2.getpixel((round(wx * 768), round(wy * 1152))) == (0, 0, 0)


def test_preset_keeps_body_shape_on_other_canvas():
    a = poses.preset("stand", 768, 1152)
    b = poses.preset("stand", 1216, 832)                            # landscape: narrower in 0..1, same pixels
    shoulder = lambda p, w, h: ((p[5][0] - p[2][0]) * w, (p[10][1] - p[0][1]) * h)
    sa, sb = shoulder(a, 768, 1152), shoulder(b, 1216, 832)
    assert abs(sa[0] / sa[1] - sb[0] / sb[1]) < 0.01


def test_named_joints_and_sdpose():
    p = poses.from_named({"left_wrist": [0.9, 0.1], "오른쪽 발목": None})
    assert p[7] == [0.9, 0.1] and p[10] is None and p[1] == poses.preset()[1]
    try:
        poses.from_named({"tail": [0, 0]})
        raise AssertionError
    except ValueError as e:
        assert "tail" in str(e)
    j = poses.from_sdpose('[{"canvas_width": 100, "canvas_height": 200, "people": [{"pose_keypoints_2d": '
                          + str([v for i in range(18) for v in (10, 20, 0.9 if i else 0.1)]) + '}]}]')
    assert j["people"][0][0] is None and j["people"][0][1] == [0.1, 0.1]
    assert poses.from_sdpose("not json") is None


def test_editor_api(client, fake):
    meta = client.get("/api/poses").json()
    assert len(meta["joints"]) == 18 and {p["id"] for p in meta["presets"]} >= {"stand", "sit"} and meta["saved"] == []
    people = [poses.preset("stand", 832, 1216), poses.mirror(poses.preset("walk", 832, 1216))]
    rec = client.post("/api/poses", json={"people": people, "width": 832, "height": 1216, "name": "둘이 걷기"}).json()
    assert rec["kind"] == "pose" and (rec["width"], rec["height"]) == (832, 1216) and len(rec["keypoints"]["people"]) == 2
    assert [s["name"] for s in client.get("/api/poses").json()["saved"]] == ["둘이 걷기"]
    assert client.post("/api/poses", json={"people": [[[0, 0]] * 3]}).status_code == 400
    # drawing with it uses the map as it is (no SDPose pass), sized like the pose
    j = wait_job(client, client.post("/api/generate", json={"prompt": "two women walking", "pose": rec["id"]}).json()["id"])
    assert j["status"] == "done", j["error"]
    assert not any(n["class_type"] == "SDPoseKeypointExtractor" for g in fake.prompts.values() for n in g.values())


def test_read_pose_from_photo(real, fake):
    photo = real.post("/api/import", files={"file": ("p.png", _png(512, 768), "image/png")}).json()["id"]
    j = wait_job(real, real.post(f"/api/images/{photo}/read-pose").json()["id"])
    assert j["status"] == "done", j["error"]
    pose = real.get(f"/api/images/{j['result_ids'][0]}").json()
    people = pose["keypoints"]["people"]
    assert len(people) == 1 and people[0][1] == [0.5, 0.22] and people[0][17] is None     # low score -> hidden
    g = list(fake.prompts.values())[-1]
    assert any(n["class_type"] == "PreviewAny" for n in g.values())
    # asked again: the kept pose map is reused, no second SDPose run
    n = len(fake.prompts)
    j2 = wait_job(real, real.post(f"/api/images/{photo}/read-pose").json()["id"])
    assert j2["result_ids"] == j["result_ids"] and len(fake.prompts) == n
    # a hand-made pose can be read back as well; a still made from it is not a pose
    assert wait_job(real, real.post(f"/api/images/{pose['id']}/read-pose").json()["id"])["status"] == "done"
    # change the pose of the photo with an edited skeleton
    edited = [[None if j is None else list(j) for j in people[0]]]
    edited[0][4] = [0.3, 0.05]
    mine = real.post("/api/poses", json={"people": edited, "width": 512, "height": 768, "parent_id": photo}).json()
    r = wait_job(real, real.post(f"/api/images/{photo}/repose", json={"pose": mine["id"]}).json()["id"])
    assert r["status"] == "done", r["error"]


def test_codex_make_and_read_pose(real, fake):
    import json
    res = rpc(real, "tools/call", {"name": "make_pose", "arguments": {
        "people": [{"left_wrist": [0.8, 0.05], "left_elbow": [0.7, 0.15]}, {"preset": "sit"}],
        "width": 1216, "height": 832, "name": "손들기"}})
    assert [c["type"] for c in res["content"]] == ["text", "image"]
    info = json.loads(res["content"][0]["text"])
    assert len(info["people"]) == 2 and info["people"][0]["left_wrist"] == [0.8, 0.05]
    rec = real.get(f"/api/images/{info['pose_id']}").json()
    assert rec["pose_name"] == "손들기" and (rec["width"], rec["height"]) == (1216, 832)
    # move one joint of that pose
    res = rpc(real, "tools/call", {"name": "make_pose", "arguments": {
        "people": [{"nose": [0.52, 0.2]}], "start_pose_id": info["pose_id"], "width": 1216, "height": 832}})
    moved = json.loads(res["content"][0]["text"])["people"][0]
    assert moved["nose"] == [0.52, 0.2] and moved["left_wrist"] == [0.8, 0.05]
    # read a photo
    photo = real.post("/api/import", files={"file": ("p.png", _png(512, 768), "image/png")}).json()["id"]
    res = rpc(real, "tools/call", {"name": "read_pose", "arguments": {"image_id": photo}})
    got = json.loads(res["content"][0]["text"])
    assert got["people"][0]["neck"] == [0.5, 0.22] and "left_ear" not in got["people"][0]
