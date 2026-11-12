"""Skeleton editor: poses made by moving joints, joints read from photos, Codex's make_pose / read_pose /
pose_video."""
import io
import json

from PIL import Image

from genstudio import poses
from test_api import wait_job
from test_lineup import real  # noqa: F401 - fixture: the real model list (SDPose, Wan, LTX)
from test_mcp import rpc


def _png(w, h):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (90, 90, 90)).save(b, "PNG")
    return b.getvalue()


def _px(png, x, y, w, h):
    return Image.open(io.BytesIO(png)).convert("RGB").getpixel((round(x * w), round(y * h)))


def test_render_draws_like_comfyui():
    p = poses.preset("stand", 768, 1152)
    png = poses.render([p], 768, 1152)
    assert Image.open(io.BytesIO(png)).size == (768, 1152) and _px(png, 0.005, 0.005, 768, 1152) == (0, 0, 0)
    assert _px(png, *p[1], 768, 1152) == poses.COLORS[1]                  # neck: colour 1 of the DWPose palette
    m = poses.mirror(p)
    assert m[2][0] == round(1 - p[5][0], 4) and m[0] == [round(1 - p[0][0], 4), p[0][1]]
    p2 = [None if i == 4 else j for i, j in enumerate(p)]                 # a hidden joint is not drawn
    assert _px(poses.render([p2], 768, 1152), *p[4], 768, 1152) == (0, 0, 0)


def test_hands_face_feet_follow_the_body():
    b3 = poses.preset3d("stand", 768, 1152)
    person = poses.compose([b3], 768, 1152, hands={"right": "open", "left": "fist"})[0]
    assert len(person["right_hand"]) == 21 and len(person["left_hand"]) == 21
    assert len(person["face"]) == 68 and len(person["feet"]) == 6
    assert person["right_hand"][0] == person["body"][4]                   # the hand starts at the wrist
    # fingers point away from the elbow (down, arms hanging), a fist is shorter than an open hand
    assert person["right_hand"][12][1] > person["body"][4][1]
    span = lambda h: max(p[1] for p in h) - h[0][1]
    assert span(person["left_hand"]) < span(person["right_hand"])
    # nose tip of the face = the nose joint; feet below the ankles
    assert person["face"][30] == [round(person["body"][0][0], 4), round(person["body"][0][1], 4)]
    assert all(f[1] > person["body"][10][1] - 0.01 for f in person["feet"][3:])
    png = poses.render([person], 768, 1152)
    assert _px(png, *person["face"][36], 768, 1152) == (255, 255, 255)   # face: white dots
    assert _px(png, *person["right_hand"][8], 768, 1152) == (0, 0, 255)  # hand: blue dots
    # move the arm: the hand comes along
    b3[4] = [0.3, 0.3, 0.0]
    moved = poses.compose([b3], 768, 1152, hands={"right": "open"})[0]
    assert moved["right_hand"][0] == [0.3, 0.3]
    # turning the camera: the body narrows, the far eye/ear go behind the head
    side = poses.compose([poses.preset3d("stand", 768, 1152)], 768, 1152, yaw=70)[0]["body"]
    front = person["body"]
    assert abs(side[2][0] - side[5][0]) < abs(front[2][0] - front[5][0]) * 0.5
    assert side[15] is None or side[14] is None
    # from above: things toward the camera (knees of a sitting person) move down the picture
    sit = poses.preset3d("sit", 768, 1152)
    top = poses.compose([sit], 768, 1152, pitch=40)[0]["body"]
    flat = poses.compose([sit], 768, 1152)[0]["body"]
    assert top[9][1] - top[8][1] > flat[9][1] - flat[8][1]


def test_hand_local_round_trip_and_copy_side():
    body = poses.preset("stand", 768, 1152)
    pts = poses.hand_points("peace", body, "left", 768, 1152)
    local = poses.hand_local(pts, body, "left", 768, 1152)
    again = poses.hand_points(local, body, "left", 768, 1152)
    assert all(abs(a[0] - b[0]) < 0.002 and abs(a[1] - b[1]) < 0.002 for a, b in zip(pts, again))
    b3 = poses.preset3d("stand", 768, 1152)
    b3[7] = [0.8, 0.1, -0.05]                                               # left wrist up and out
    c = poses.copy_side(b3, "left")
    assert c[4][1] == 0.1 and abs((c[4][0] - c[2][0]) + (b3[7][0] - b3[5][0])) < 1e-3 and c[4][2] == -0.05


def test_preset_keeps_body_shape_on_other_canvas():
    a = poses.preset("stand", 768, 1152)
    b = poses.preset("stand", 1216, 832)
    shape = lambda p, w, h: ((p[5][0] - p[2][0]) * w) / ((p[10][1] - p[0][1]) * h)
    assert abs(shape(a, 768, 1152) - shape(b, 1216, 832)) < 0.01


def test_named_joints_and_sdpose():
    p = poses.from_named({"left_wrist": [0.9, 0.1], "오른쪽 발목": None})
    assert p[7] == [0.9, 0.1, 0.0] and p[10] is None and p[1][:2] == poses.preset()[1]
    try:
        poses.from_named({"tail": [0, 0]})
        raise AssertionError
    except ValueError as e:
        assert "tail" in str(e)
    flat = [v for i in range(18) for v in (10, 20, 0.9 if i else 0.1)]
    hand = [v for i in range(21) for v in (30, 40, 0.9)]
    j = poses.from_sdpose(json.dumps([{"canvas_width": 100, "canvas_height": 200, "people": [
        {"pose_keypoints_2d": flat, "hand_right_keypoints_2d": hand, "hand_left_keypoints_2d": [0] * 63}]}]))
    person = j["people"][0]
    assert person["body"][0] is None and person["body"][1] == [0.1, 0.1]
    assert person["right_hand"][0] == [0.3, 0.2] and person["left_hand"] is None
    assert poses.from_sdpose("not json") is None
    # older saves (a plain list of joints) still load
    assert poses.clean([poses.preset()])[0]["body"][1] == poses.preset()[1]


def test_editor_api(client, fake):
    meta = client.get("/api/poses").json()
    assert len(meta["joints"]) == 18 and {p["id"] for p in meta["presets"]} >= {"stand", "sit"} and meta["saved"] == []
    assert {h["id"] for h in meta["hands"]} >= {"open", "fist", "peace"} and len(meta["face"]) == 68
    people = poses.compose([poses.preset3d("stand", 832, 1216), poses.mirror(poses.preset3d("walk", 832, 1216))],
                           832, 1216, hands={"right": "open", "left": "relaxed"})
    rec = client.post("/api/poses", json={"people": people, "width": 832, "height": 1216, "name": "둘이 걷기",
                                          "edit": {"camera": {"yaw": 0, "pitch": 0}}}).json()
    assert rec["kind"] == "pose" and (rec["width"], rec["height"]) == (832, 1216) and len(rec["keypoints"]["people"]) == 2
    assert rec["keypoints"]["edit"]["camera"]["yaw"] == 0
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
    body = pose["keypoints"]["people"][0]["body"]
    assert body[1] == [0.5, 0.22] and body[17] is None                                   # low score -> hidden
    assert any(n["class_type"] == "PreviewAny" for n in list(fake.prompts.values())[-1].values())
    n = len(fake.prompts)                                       # asked again: the kept pose map is reused
    j2 = wait_job(real, real.post(f"/api/images/{photo}/read-pose").json()["id"])
    assert j2["result_ids"] == j["result_ids"] and len(fake.prompts) == n
    assert wait_job(real, real.post(f"/api/images/{pose['id']}/read-pose").json()["id"])["status"] == "done"
    edited = [None if p is None else list(p) for p in body]
    edited[4] = [0.3, 0.05]
    mine = real.post("/api/poses", json={"people": [edited], "width": 512, "height": 768, "parent_id": photo}).json()
    r = wait_job(real, real.post(f"/api/images/{photo}/repose", json={"pose": mine["id"]}).json()["id"])
    assert r["status"] == "done", r["error"]


def test_codex_make_and_read_pose(real, fake):
    res = rpc(real, "tools/call", {"name": "make_pose", "arguments": {
        "people": [{"left_wrist": [0.8, 0.05], "left_elbow": [0.7, 0.15], "right_hand": "fist", "left_hand": "peace"},
                   {"preset": "sit", "copy_side": "left"}],
        "width": 1216, "height": 832, "name": "손들기", "camera_yaw": 20, "camera_pitch": 10}})
    assert [c["type"] for c in res["content"]] == ["text", "image"]
    info = json.loads(res["content"][0]["text"])
    assert len(info["people"]) == 2 and info["people"][0]["left_wrist"][:2] == [0.8, 0.05]
    assert info["people"][0]["left_hand"] == "peace" and info["camera"] == {"yaw": 20.0, "pitch": 10.0}
    rec = real.get(f"/api/images/{info['pose_id']}").json()
    drawn = rec["keypoints"]["people"][0]
    assert rec["pose_name"] == "손들기" and (rec["width"], rec["height"]) == (1216, 832)
    assert drawn["left_hand"] and drawn["face"] and drawn["feet"]
    # move one joint of that pose: everything else (hands, camera depth) is kept
    res = rpc(real, "tools/call", {"name": "make_pose", "arguments": {
        "people": [{"nose": [0.52, 0.2]}], "start_pose_id": info["pose_id"], "width": 1216, "height": 832}})
    moved = json.loads(res["content"][0]["text"])["people"][0]
    assert moved["nose"][:2] == [0.52, 0.2] and moved["left_wrist"][:2] == [0.8, 0.05] and moved["left_hand"] == "peace"
    bad = real.post("/mcp", headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
                                     "MCP-Protocol-Version": "2025-06-18"},
                    json={"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                          "params": {"name": "make_pose", "arguments": {"people": [{"right_hand": "claw"}]}}}).json()
    assert "claw" in json.dumps(bad)
    # read a photo
    photo = real.post("/api/import", files={"file": ("p.png", _png(512, 768), "image/png")}).json()["id"]
    res = rpc(real, "tools/call", {"name": "read_pose", "arguments": {"image_id": photo}})
    got = json.loads(res["content"][0]["text"])
    assert got["people"][0]["neck"] == [0.5, 0.22] and "left_ear" not in got["people"][0]


def test_pose_video_for_codex(real, fake):
    """Poses -> stills of the same person (Qwen 자세 바꾸기) -> one Wan clip through them (start/middle/end)."""
    photo = real.post("/api/import", files={"file": ("p.png", _png(720, 1280), "image/png")}).json()["id"]
    mk = lambda **kw: json.loads(rpc(real, "tools/call", {"name": "make_pose", "arguments": kw})["content"][0]["text"])["pose_id"]
    up, sit = mk(people=[{"preset": "arms_up"}], width=720, height=1280), mk(people=[{"preset": "sit"}], width=720, height=1280)
    res = rpc(real, "tools/call", {"name": "pose_video", "arguments": {
        "image_id": photo, "poses": [{"pose_id": up, "at": 2}, {"pose_id": sit, "at": -1}], "prompt": "she dances",
        "seconds": 5, "wait_seconds": 30}})
    out = json.loads(res["content"][0]["text"])
    assert out["status"] == "done", out
    assert out["video"]["media"] == "video" and len(out["pose_stills"]) == 2
    g = list(fake.prompts.values())[-1]
    assert sum(n["class_type"] == "WanFirstLastFrameToVideo" for n in g.values()) == 2      # photo -> arms up -> sitting
    reposes = [p for p in fake.prompts.values() if any(n["class_type"] == "TextEncodeQwenImage21" for n in p.values())]
    assert len(reposes) >= 2
    too_many = real.post("/api/pose-video", json={"image_id": photo,
                                                  "poses": [{"pose_id": up, "at": 1}, {"pose_id": sit, "at": 2},
                                                            {"pose_id": up, "at": -1}]})
    assert too_many.status_code == 400
