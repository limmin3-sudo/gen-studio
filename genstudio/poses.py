"""Body poses as joints you can move: the skeleton editor, Codex's make_pose, and pose maps read from photos.

A pose is a canvas size plus people; each person is 18 joints in OpenPose order, each [x, y] in 0..1 of the
canvas (or None when the joint is not there). It is drawn the way ComfyUI's SDPose "Draw Keypoints" node draws
bodies (DWPose colours and limb order, filled ellipse limbs, round joints on black), so a pose made by hand
goes through the same ControlNet / Qwen paths as one read from a photo.
"""
from __future__ import annotations

import io
import json
import math

from PIL import Image, ImageDraw

JOINTS = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
          "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
          "right_eye", "left_eye", "right_ear", "left_ear"]
JOINT_KO = ["코", "목", "오른쪽 어깨", "오른쪽 팔꿈치", "오른쪽 손목", "왼쪽 어깨", "왼쪽 팔꿈치", "왼쪽 손목",
            "오른쪽 엉덩이", "오른쪽 무릎", "오른쪽 발목", "왼쪽 엉덩이", "왼쪽 무릎", "왼쪽 발목",
            "오른쪽 눈", "왼쪽 눈", "오른쪽 귀", "왼쪽 귀"]
# ComfyUI comfy_extras/pose/keypoint_draw.py (DWPose): body limbs, then head edges; 1-indexed
LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
         [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
# The person's right side is on the picture's left (facing the camera).
MIRROR = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]

# Shapes for a 2:3 portrait canvas, facing the camera.
PRESETS: dict[str, tuple[str, list[list[float]]]] = {
    "stand": ("서기", [[.5, .14], [.5, .22], [.41, .23], [.38, .36], [.37, .48], [.59, .23], [.62, .36], [.63, .48],
                      [.45, .5], [.45, .68], [.45, .87], [.55, .5], [.55, .68], [.55, .87],
                      [.485, .125], [.515, .125], [.465, .135], [.535, .135]]),
    "arms_up": ("만세", [[.5, .2], [.5, .28], [.41, .29], [.37, .19], [.35, .08], [.59, .29], [.63, .19], [.65, .08],
                        [.45, .56], [.45, .73], [.45, .9], [.55, .56], [.55, .73], [.55, .9],
                        [.485, .185], [.515, .185], [.465, .195], [.535, .195]]),
    "walk": ("걷기", [[.5, .14], [.5, .22], [.41, .23], [.37, .35], [.38, .46], [.59, .23], [.62, .35], [.6, .46],
                     [.45, .5], [.42, .67], [.4, .85], [.55, .5], [.58, .66], [.62, .84],
                     [.485, .125], [.515, .125], [.465, .135], [.535, .135]]),
    "sit": ("앉기", [[.5, .26], [.5, .34], [.41, .35], [.39, .47], [.43, .57], [.59, .35], [.61, .47], [.57, .57],
                    [.45, .6], [.42, .66], [.42, .84], [.55, .6], [.58, .66], [.58, .84],
                    [.485, .245], [.515, .245], [.465, .255], [.535, .255]]),
    "hands_hips": ("허리에 손", [[.5, .14], [.5, .22], [.41, .23], [.33, .33], [.43, .45], [.59, .23], [.67, .33],
                                 [.57, .45], [.45, .5], [.44, .68], [.43, .87], [.55, .5], [.56, .68], [.57, .87],
                                 [.485, .125], [.515, .125], [.465, .135], [.535, .135]]),
}


def preset(name: str = "stand", width: int = 768, height: int = 1152) -> list:
    """A preset person fitted to this canvas (kept the same body shape whatever the canvas shape)."""
    pts = PRESETS.get(name, PRESETS["stand"])[1]
    # presets are drawn on 2:3; keep their pixel shape on another canvas
    s = min(width / 2, height / 3)
    return [[round(.5 + (x - .5) * 2 * s / width, 4), round(.5 + (y - .5) * 3 * s / height, 4)] for x, y in pts]


def clean(people, max_people: int = 10) -> list[list]:
    """Validate people: lists of 18 joints, each [x, y] in 0..1 or None."""
    if not isinstance(people, list) or not people:
        raise ValueError("사람이 한 명 이상 있어야 해요")
    out = []
    for p in people[:max_people]:
        if not isinstance(p, list) or len(p) != 18:
            raise ValueError("사람마다 관절 18개가 필요해요")
        joints = []
        for j in p:
            if j is None:
                joints.append(None)
                continue
            x, y = float(j[0]), float(j[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("관절 위치가 잘못됐어요")
            joints.append([round(min(1.5, max(-0.5, x)), 4), round(min(1.5, max(-0.5, y)), 4)])
        out.append(joints)
    return out


def from_named(person: dict, base: list | None = None) -> list:
    """{"left_wrist": [x, y], ...} (any subset, Korean names too) on top of base (default: standing)."""
    joints = [list(j) if j else None for j in (base or preset())]
    for k, v in (person or {}).items():
        key = str(k).strip().lower().replace(" ", "_").replace("-", "_")
        i = JOINTS.index(key) if key in JOINTS else JOINT_KO.index(str(k).strip()) if str(k).strip() in JOINT_KO else -1
        if i < 0:
            raise ValueError(f"모르는 관절 이름: {k} (쓸 수 있는 이름: {', '.join(JOINTS)})")
        joints[i] = None if v is None else [float(v[0]), float(v[1])]
    return joints


def named(person: list) -> dict:
    return {JOINTS[i]: j for i, j in enumerate(person) if j is not None}


def mirror(person: list) -> list:
    return [None if person[MIRROR[i]] is None else [round(1 - person[MIRROR[i]][0], 4), person[MIRROR[i]][1]]
            for i in range(18)]


def from_sdpose(text: str, min_score: float = 0.3) -> dict | None:
    """ComfyUI's SDPose keypoints (PreviewAny JSON) -> {"people": [...]} in 0..1, or None."""
    try:
        frames = json.loads(text)
    except (TypeError, ValueError):
        return None
    if isinstance(frames, dict):
        frames = [frames]
    if not frames or not isinstance(frames[0], dict):
        return None
    f = frames[0]
    w, h = float(f.get("canvas_width") or 0), float(f.get("canvas_height") or 0)
    if w <= 0 or h <= 0:
        return None
    people = []
    for p in f.get("people") or []:
        flat = p.get("pose_keypoints_2d") or []
        if len(flat) < 54:
            continue
        joints = [[round(flat[3 * i] / w, 4), round(flat[3 * i + 1] / h, 4)] if flat[3 * i + 2] >= min_score else None
                  for i in range(18)]
        if sum(j is not None for j in joints) >= 4:
            people.append(joints)
    return {"people": people} if people else None


def render(people: list, width: int, height: int) -> bytes:
    """The skeleton picture (PNG) the image models read."""
    im = Image.new("RGB", (width, height), (0, 0, 0))
    d = ImageDraw.Draw(im)
    stick = max(3, round(4 * max(width, height) / 1024))
    for person in people:
        pts = [None if j is None else (j[0] * width, j[1] * height) for j in person]
        for i, (a, b) in enumerate(LIMBS):
            p, q = pts[a - 1], pts[b - 1]
            if p is None or q is None:
                continue
            length = math.dist(p, q)
            if length < 1:
                continue
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            ang = math.atan2(p[1] - q[1], p[0] - q[0])
            ca, sa = math.cos(ang), math.sin(ang)
            poly = [(mx + length / 2 * math.cos(t) * ca - stick * math.sin(t) * sa,
                     my + length / 2 * math.cos(t) * sa + stick * math.sin(t) * ca)
                    for t in (k * math.pi / 180 for k in range(0, 360, 4))]
            d.polygon(poly, fill=COLORS[i % len(COLORS)])
        for i, p in enumerate(pts):
            if p is not None and 0 <= p[0] < width and 0 <= p[1] < height:
                d.ellipse((p[0] - stick, p[1] - stick, p[0] + stick, p[1] + stick), fill=COLORS[i % len(COLORS)])
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def canvas_size(aspect_w: float, aspect_h: float, area: int = 1024 * 1024) -> tuple[int, int]:
    k = (area / (aspect_w * aspect_h)) ** 0.5
    return max(256, round(aspect_w * k / 64) * 64), max(256, round(aspect_h * k / 64) * 64)
