"""Body poses as joints you can move: the skeleton editor, Codex's make_pose, and pose maps read from photos.

A pose is a canvas size plus people. Each person has 18 body joints in OpenPose order ([x, y] in 0..1 of the
canvas, or None when the joint is not there) and optionally hands (21 points each), a face (68 points) and feet
(6 points), the same layout ComfyUI's SDPose nodes use. It is drawn the way ComfyUI's SDPose "Draw Keypoints"
node draws (DWPose colours and limb order, filled ellipse limbs, rainbow hand lines with blue dots, white face
dots), so a pose made by hand goes through the same ControlNet / Qwen paths as one read from a photo.

Bodies can also carry depth (z, in canvas heights, + = away from the camera) so the camera can be turned:
yaw (left/right, degrees) and pitch (from above, degrees). Hands follow the forearm, the face follows the eyes
and nose, and the feet follow the ankles, so turning the camera or moving an arm keeps them in place.
"""
from __future__ import annotations

import colorsys
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
HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
HAND_COLORS = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / len(HAND_EDGES), 1.0, 1.0))
               for i in range(len(HAND_EDGES))]
# The person's right side is on the picture's left (facing the camera).
MIRROR = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
ARMS = {"right": (2, 3, 4), "left": (5, 6, 7)}
LEGS = {"right": (8, 9, 10), "left": (11, 12, 13)}

# Shapes for a 2:3 portrait canvas, facing the camera: (name, [x, y] per joint, depth per joint)
_HEAD_Z = [-.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -.02, -.02, .02, .02]


def _z(**kw) -> list[float]:
    z = list(_HEAD_Z)
    for k, v in kw.items():
        z[JOINTS.index(k)] = v
    return z


PRESETS: dict[str, tuple[str, list[list[float]], list[float]]] = {
    "stand": ("서기", [[.5, .14], [.5, .22], [.41, .23], [.38, .36], [.37, .48], [.59, .23], [.62, .36], [.63, .48],
                      [.45, .5], [.45, .68], [.45, .87], [.55, .5], [.55, .68], [.55, .87],
                      [.485, .125], [.515, .125], [.465, .135], [.535, .135]], _z()),
    "arms_up": ("만세", [[.5, .2], [.5, .28], [.41, .29], [.37, .19], [.35, .08], [.59, .29], [.63, .19], [.65, .08],
                        [.45, .56], [.45, .73], [.45, .9], [.55, .56], [.55, .73], [.55, .9],
                        [.485, .185], [.515, .185], [.465, .195], [.535, .195]], _z()),
    "walk": ("걷기", [[.5, .14], [.5, .22], [.41, .23], [.39, .35], [.39, .46], [.59, .23], [.61, .35], [.61, .46],
                     [.45, .5], [.44, .67], [.44, .85], [.55, .5], [.56, .67], [.56, .85],
                     [.485, .125], [.515, .125], [.465, .135], [.535, .135]],
             _z(right_elbow=.04, right_wrist=.09, left_elbow=-.04, left_wrist=-.09,
                right_knee=-.07, right_ankle=-.03, left_knee=.03, left_ankle=.1)),
    "sit": ("앉기", [[.5, .26], [.5, .34], [.41, .35], [.39, .47], [.43, .56], [.59, .35], [.61, .47], [.57, .56],
                    [.45, .6], [.44, .62], [.44, .82], [.55, .6], [.56, .62], [.56, .82],
                    [.485, .245], [.515, .245], [.465, .255], [.535, .255]],
            _z(right_elbow=-.04, right_wrist=-.13, left_elbow=-.04, left_wrist=-.13,
               right_knee=-.17, right_ankle=-.17, left_knee=-.17, left_ankle=-.17)),
    "hands_hips": ("허리에 손", [[.5, .14], [.5, .22], [.41, .23], [.33, .33], [.43, .45], [.59, .23], [.67, .33],
                                 [.57, .45], [.45, .5], [.44, .68], [.43, .87], [.55, .5], [.56, .68], [.57, .87],
                                 [.485, .125], [.515, .125], [.465, .135], [.535, .135]], _z()),
}

# Hand shapes: wrist at 0,0; y along the forearm (elbow -> wrist), x toward the thumb; unit = forearm length.
_FIST_FINGERS = [[.11, .46], [.1, .42], [.09, .36], [.03, .48], [.03, .44], [.03, .38], [-.06, .46], [-.06, .42],
                 [-.06, .36], [-.13, .42], [-.13, .38], [-.12, .33]]
_MCP = {5: [.1, .38], 9: [.03, .4], 13: [-.05, .38], 17: [-.12, .34]}


def _hand(thumb, index, middle, ring, pinky) -> list[list[float]]:
    return [[0, 0], *thumb, _MCP[5], *index, _MCP[9], *middle, _MCP[13], *ring, _MCP[17], *pinky]


_OPEN_THUMB = [[.12, .08], [.2, .18], [.26, .27], [.3, .35]]
_FIST_THUMB = [[.1, .1], [.16, .2], [.14, .3], [.08, .36]]
_OPEN = {"index": [[.11, .52], [.12, .61], [.12, .69]], "middle": [[.03, .56], [.03, .66], [.03, .74]],
         "ring": [[-.06, .52], [-.07, .61], [-.07, .68]], "pinky": [[-.14, .45], [-.15, .52], [-.16, .58]]}
_FIST = {"index": _FIST_FINGERS[0:3], "middle": _FIST_FINGERS[3:6], "ring": _FIST_FINGERS[6:9],
         "pinky": _FIST_FINGERS[9:12]}
_RELAXED = {"index": [[.11, .5], [.1, .57], [.08, .62]], "middle": [[.03, .53], [.02, .61], [0, .66]],
            "ring": [[-.06, .5], [-.07, .57], [-.08, .61]], "pinky": [[-.13, .44], [-.14, .5], [-.14, .54]]}
HANDS: dict[str, tuple[str, list[list[float]]]] = {
    "open": ("펼친 손", _hand(_OPEN_THUMB, _OPEN["index"], _OPEN["middle"], _OPEN["ring"], _OPEN["pinky"])),
    "relaxed": ("편한 손", _hand([[.1, .08], [.16, .17], [.2, .25], [.22, .31]], _RELAXED["index"],
                                 _RELAXED["middle"], _RELAXED["ring"], _RELAXED["pinky"])),
    "fist": ("주먹", _hand(_FIST_THUMB, _FIST["index"], _FIST["middle"], _FIST["ring"], _FIST["pinky"])),
    "point": ("검지", _hand(_FIST_THUMB, _OPEN["index"], _FIST["middle"], _FIST["ring"], _FIST["pinky"])),
    "peace": ("브이", _hand(_FIST_THUMB, [[.13, .52], [.16, .61], [.18, .69]], [[.02, .56], [0, .66], [-.01, .74]],
                           _FIST["ring"], _FIST["pinky"])),
    "thumbs_up": ("엄지 척", _hand([[.12, .06], [.22, .1], [.32, .12], [.41, .13]], _FIST["index"], _FIST["middle"],
                                  _FIST["ring"], _FIST["pinky"])),
}

# 68-point face (iBUG layout), front view, in a unit box; eyes centred at (.29, .27) and (.71, .27)
FACE_EYES = ((.29, .27), (.71, .27))


def _face_template() -> list[list[float]]:
    jaw = [[.5 + .5 * math.cos(math.pi - i * math.pi / 16), .2 + .8 * math.sin(math.pi - i * math.pi / 16)]
           for i in range(17)]
    brow_r = [[.12 + .075 * k, .14 - .05 * math.sin(math.pi * k / 4)] for k in range(5)]
    brow_l = [[.58 + .075 * k, .14 - .05 * math.sin(math.pi * k / 4)] for k in range(5)]
    bridge = [[.5, .25], [.5, .35], [.5, .45], [.5, .55]]
    nose = [[.4, .62], [.45, .64], [.5, .65], [.55, .64], [.6, .62]]
    eye_r = [[.2, .27], [.25, .24], [.32, .24], [.38, .27], [.32, .3], [.25, .3]]
    eye_l = [[.62, .27], [.68, .24], [.75, .24], [.8, .27], [.75, .3], [.68, .3]]
    mouth = [[.35, .78], [.4, .75], [.46, .73], [.5, .74], [.54, .73], [.6, .75], [.65, .78], [.6, .82], [.55, .84],
             [.5, .845], [.45, .84], [.4, .82], [.38, .78], [.45, .77], [.5, .775], [.55, .77], [.62, .78],
             [.55, .79], [.5, .795], [.45, .79]]
    return [[round(x, 4), round(y, 4)] for x, y in jaw + brow_r + brow_l + bridge + nose + eye_r + eye_l + mouth]


FACE = _face_template()


# ------------------------------------------------------------------ building a person
def _fit(width: int, height: int):
    """Presets are drawn on 2:3; keep their pixel shape on another canvas."""
    s = min(width / 2, height / 3)
    return lambda x, y: [round(.5 + (x - .5) * 2 * s / width, 4), round(.5 + (y - .5) * 3 * s / height, 4)], 3 * s / height


def preset(name: str = "stand", width: int = 768, height: int = 1152) -> list:
    """A preset person's body joints [x, y] fitted to this canvas."""
    f, _ = _fit(width, height)
    return [f(x, y) for x, y in PRESETS.get(name, PRESETS["stand"])[1]]


def preset3d(name: str = "stand", width: int = 768, height: int = 1152) -> list:
    """The same with depth: [x, y, z]."""
    f, k = _fit(width, height)
    _, pts, zs = PRESETS.get(name, PRESETS["stand"])
    return [[*f(x, y), round(z * k, 4)] for (x, y), z in zip(pts, zs)]


def _px(j, W, H):
    return None if j is None else (j[0] * W, j[1] * H, (j[2] if len(j) > 2 else 0.0) * H)


def project(body3d: list, W: int, H: int, yaw: float = 0.0, pitch: float = 0.0,
            center: tuple[float, float] | None = None) -> list:
    """Turn the camera around the scene centre (orthographic): yaw = from the side, pitch = from above."""
    cy_, sy_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cx, cyc = center if center else (W / 2, H / 2)
    out = []
    for j in body3d:
        p = _px(j, W, H)
        if p is None:
            out.append(None)
            continue
        x, y, z = p[0] - cx, p[1] - cyc, p[2]
        x1, z1 = x * cy_ + z * sy_, -x * sy_ + z * cy_
        y1, z2 = y * cp - z1 * sp, y * sp + z1 * cp
        out.append([(cx + x1) / W, (cyc + y1) / H, z2 / H])
    return out


def scene_center(people3d: list[list], W: int, H: int) -> tuple[float, float]:
    hips = [j for b in people3d for j in (b[8], b[11]) if j]
    if not hips:
        return W / 2, H / 2
    return sum(j[0] for j in hips) / len(hips) * W, sum(j[1] for j in hips) / len(hips) * H


def hand_points(shape, body2d: list, side: str, W: int, H: int) -> list | None:
    """21 hand points placed at the wrist, along the forearm, thumb toward the body."""
    s, e, w = (body2d[i] for i in ARMS[side])
    if not (e and w):
        return None
    local = HANDS[shape][1] if isinstance(shape, str) else shape
    ex, ey, wx, wy = e[0] * W, e[1] * H, w[0] * W, w[1] * H
    dx, dy = wx - ex, wy - ey
    L = math.hypot(dx, dy)
    if L < 1:
        return None
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    ref = body2d[1] or s
    if ref and px * (ref[0] * W - wx) + py * (ref[1] * H - wy) < 0:
        px, py = -px, -py
    return [[round((wx + L * (lx * px + ly * ux)) / W, 4), round((wy + L * (lx * py + ly * uy)) / H, 4)]
            for lx, ly in local]


def hand_local(points: list, body2d: list, side: str, W: int, H: int) -> list | None:
    """The inverse of hand_points: absolute hand points -> the forearm frame (so they follow the arm)."""
    s, e, w = (body2d[i] for i in ARMS[side])
    if not (e and w) or not points or any(p is None for p in points):
        return None
    ex, ey, wx, wy = e[0] * W, e[1] * H, w[0] * W, w[1] * H
    dx, dy = wx - ex, wy - ey
    L = math.hypot(dx, dy)
    if L < 1:
        return None
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    ref = body2d[1] or s
    if ref and px * (ref[0] * W - wx) + py * (ref[1] * H - wy) < 0:
        px, py = -px, -py
    out = []
    for x, y in points:
        vx, vy = x * W - wx, y * H - wy
        out.append([round((vx * px + vy * py) / L, 4), round((vx * ux + vy * uy) / L, 4)])
    return out


def face_points(body2d: list, W: int, H: int) -> list | None:
    """68 face points from the eyes and nose (they turn with the head)."""
    n, er, el = body2d[0], body2d[14], body2d[15]
    if not (n and er and el):
        return None
    a = complex(er[0] * W, er[1] * H)
    b = complex(el[0] * W, el[1] * H)
    t1, t2 = complex(*FACE_EYES[0]), complex(*FACE_EYES[1])
    k = (b - a) / (t2 - t1)
    if abs(k) < 1e-6:
        return None
    pts = [a + (complex(x, y) - t1) * k for x, y in FACE]
    nose = complex(n[0] * W, n[1] * H)
    delta = nose - pts[30]
    weights = {27: .25, 28: .5, 29: .75, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1}
    out = []
    for i, p in enumerate(pts):
        wgt = weights.get(i, .5 if i >= 48 else .2 if i <= 16 else 0)
        q = p + delta * wgt
        out.append([round(q.real / W, 4), round(q.imag / H, 4)])
    return out


def feet3d(body3d: list, W: int, H: int) -> list:
    """Big toe, small toe, heel for the left then the right foot (SDPose order), in 3D, toes toward the camera."""
    out = []
    for side, inward in (("left", -1), ("right", 1)):
        _, kn, an = (body3d[i] for i in LEGS[side])
        if not (kn and an):
            out += [None, None, None]
            continue
        k, a = _px(kn, W, H), _px(an, W, H)
        L = max(1.0, math.dist(k, a))
        for dx, dy, dz in ((inward * .06, .18, -.38), (-inward * .08, .18, -.3), (0, .1, .12)):
            out.append([(a[0] + dx * L) / W, (a[1] + dy * L) / H, (a[2] + dz * L) / H])
    return out


def compose(people3d: list[list], W: int, H: int, *, yaw: float = 0, pitch: float = 0, hands=None,
            face: bool = True, feet: bool = True) -> list[dict]:
    """Bodies with depth -> what is drawn: body, hands, face and feet seen from the camera.
    hands: one {"right": shape|local points|None, "left": ...} per person (or one for everyone)."""
    c = scene_center(people3d, W, H)
    out = []
    for i, b3 in enumerate(people3d):
        body = project(b3, W, H, yaw, pitch, c)
        _hide_far_side(body, b3, W, H, yaw, pitch, c)
        h = hands[i] if isinstance(hands, list) and i < len(hands) else hands if isinstance(hands, dict) else {}
        person = {"body": [None if j is None else [round(j[0], 4), round(j[1], 4)] for j in body]}
        for side in ("right", "left"):
            shape = (h or {}).get(side)
            person[f"{side}_hand"] = hand_points(shape, person["body"], side, W, H) if shape else None
        person["face"] = face_points(person["body"], W, H) if face else None
        if feet:
            f = project(feet3d(b3, W, H), W, H, yaw, pitch, c)
            person["feet"] = [None if j is None else [round(j[0], 4), round(j[1], 4)] for j in f]
        else:
            person["feet"] = None
        out.append(person)
    return out


def _hide_far_side(body: list, b3: list, W: int, H: int, yaw: float, pitch: float, c) -> None:
    """Seen from the side, the far ear (from 30 degrees) and the far eye (from 55) are behind the head.
    Turning the camera to positive yaw shows more of the person's left side, so the right side is far."""
    eye, ear = (14, 16) if yaw > 0 else (15, 17)
    if abs(yaw) >= 30:
        body[ear] = None
    if abs(yaw) >= 55:
        body[eye] = None


# ------------------------------------------------------------------ checking and converting
def _pts(v, n: int, what: str):
    if v is None:
        return None
    if not isinstance(v, list) or len(v) != n:
        raise ValueError(f"{what}: 점 {n}개가 필요해요")
    out = []
    for j in v:
        if j is None:
            out.append(None)
            continue
        x, y = float(j[0]), float(j[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("관절 위치가 잘못됐어요")
        out.append([round(min(1.5, max(-0.5, x)), 4), round(min(1.5, max(-0.5, y)), 4)])
    return out


def clean(people, max_people: int = 10) -> list[dict]:
    """Validate people. Each is a list of 18 body joints (older saves) or {"body", "right_hand", "left_hand",
    "face", "feet"} with 18 / 21 / 21 / 68 / 6 points; every point [x, y] in 0..1 or None."""
    if not isinstance(people, list) or not people:
        raise ValueError("사람이 한 명 이상 있어야 해요")
    out = []
    for p in people[:max_people]:
        if isinstance(p, list):
            p = {"body": p}
        if not isinstance(p, dict) or not isinstance(p.get("body"), list) or len(p["body"]) != 18:
            raise ValueError("사람마다 관절 18개가 필요해요")
        out.append({"body": _pts(p["body"], 18, "몸"), "right_hand": _pts(p.get("right_hand"), 21, "오른손"),
                    "left_hand": _pts(p.get("left_hand"), 21, "왼손"), "face": _pts(p.get("face"), 68, "얼굴"),
                    "feet": _pts(p.get("feet"), 6, "발")})
    return out


def body_of(person) -> list:
    return person["body"] if isinstance(person, dict) else person


def _joint_index(k) -> int:
    key = str(k).strip().lower().replace(" ", "_").replace("-", "_")
    if key in JOINTS:
        return JOINTS.index(key)
    if str(k).strip() in JOINT_KO:
        return JOINT_KO.index(str(k).strip())
    raise ValueError(f"모르는 관절 이름: {k} (쓸 수 있는 이름: {', '.join(JOINTS)})")


def from_named(person: dict, base: list | None = None) -> list:
    """{"left_wrist": [x, y] or [x, y, z], ...} (any subset, Korean names too) on top of base (default:
    standing). Returns [x, y, z] joints."""
    joints = [None if j is None else [*j[:2], j[2] if len(j) > 2 else 0.0] for j in (base or preset3d())]
    for k, v in (person or {}).items():
        i = _joint_index(k)
        if v is None:
            joints[i] = None
        else:
            z = float(v[2]) if len(v) > 2 else (joints[i][2] if joints[i] else 0.0)
            joints[i] = [float(v[0]), float(v[1]), z]
    return joints


def named(person) -> dict:
    return {JOINTS[i]: j for i, j in enumerate(body_of(person)) if j is not None}


def mirror(person: list) -> list:
    return [None if person[MIRROR[i]] is None else [round(1 - person[MIRROR[i]][0], 4), *person[MIRROR[i]][1:]]
            for i in range(18)]


def copy_side(body3d: list, src: str) -> list:
    """Copy one side's arm and leg to the other side, mirrored around the body (src: 'left' or 'right')."""
    dst = "right" if src == "left" else "left"
    out = [None if j is None else list(j) for j in body3d]
    for group in (ARMS, LEGS):
        a, b = group[src], group[dst]
        if not (body3d[a[0]] and body3d[b[0]]):
            continue
        for i_s, i_d in zip(a[1:], b[1:]):
            js = body3d[i_s]
            if js is None:
                out[i_d] = None
                continue
            dx = js[0] - body3d[a[0]][0]
            out[i_d] = [round(body3d[b[0]][0] - dx, 4), js[1], *(js[2:] or [0.0])]
    return out


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

    def pts(flat, n, need=0.5):
        if not flat or len(flat) < 3 * n:
            return None
        out = [[round(flat[3 * i] / w, 4), round(flat[3 * i + 1] / h, 4)] if flat[3 * i + 2] >= min_score else None
               for i in range(n)]
        return out if sum(p is not None for p in out) >= need * n else None

    people = []
    for p in f.get("people") or []:
        body = pts(p.get("pose_keypoints_2d"), 18, 0)
        if not body or sum(j is not None for j in body) < 4:
            continue
        people.append({"body": body, "right_hand": pts(p.get("hand_right_keypoints_2d"), 21),
                       "left_hand": pts(p.get("hand_left_keypoints_2d"), 21),
                       "face": pts((p.get("face_keypoints_2d") or [])[:204], 68),
                       "feet": pts(p.get("foot_keypoints_2d"), 6)})
    return {"people": people} if people else None


# ------------------------------------------------------------------ drawing
def _ellipse_limb(d: ImageDraw.ImageDraw, p, q, stick: float, color) -> None:
    length = math.dist(p, q)
    if length < 1:
        return
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    ang = math.atan2(p[1] - q[1], p[0] - q[0])
    ca, sa = math.cos(ang), math.sin(ang)
    poly = [(mx + length / 2 * math.cos(t) * ca - stick * math.sin(t) * sa,
             my + length / 2 * math.cos(t) * sa + stick * math.sin(t) * ca)
            for t in (k * math.pi / 180 for k in range(0, 360, 4))]
    d.polygon(poly, fill=color)


def render(people: list, width: int, height: int) -> bytes:
    """The skeleton picture (PNG) the image models read."""
    im = Image.new("RGB", (width, height), (0, 0, 0))
    d = ImageDraw.Draw(im)
    k = max(width, height) / 1024
    stick, marker = max(3, round(4 * k)), max(3, round(4 * k))
    hand_w, hand_r, face_r = max(1, round(2 * k)), max(2, round(4 * k)), max(1, round(2 * k))

    def dot(p, r, color):
        if p is not None and 0 <= p[0] < width and 0 <= p[1] < height:
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=color)

    for person in people:
        person = person if isinstance(person, dict) else {"body": person}
        px = lambda pts: [None if j is None else (j[0] * width, j[1] * height) for j in pts] if pts else None  # noqa: E731
        body = px(person["body"])
        for i, (a, b) in enumerate(LIMBS):
            if body[a - 1] is not None and body[b - 1] is not None:
                _ellipse_limb(d, body[a - 1], body[b - 1], stick, COLORS[i % len(COLORS)])
        for i, p in enumerate(body):
            dot(p, marker, COLORS[i % len(COLORS)])
        for i, p in enumerate(px(person.get("feet")) or []):
            dot(p, marker, COLORS[(18 + i) % len(COLORS)])
        for side in ("right_hand", "left_hand"):
            hand = px(person.get(side))
            if not hand:
                continue
            for i, (a, b) in enumerate(HAND_EDGES):
                if hand[a] is not None and hand[b] is not None:
                    d.line((hand[a], hand[b]), fill=HAND_COLORS[i], width=hand_w)
            for p in hand:
                dot(p, hand_r, (0, 0, 255))
        for p in px(person.get("face")) or []:
            dot(p, face_r, (255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def canvas_size(aspect_w: float, aspect_h: float, area: int = 1024 * 1024) -> tuple[int, int]:
    k = (area / (aspect_w * aspect_h)) ** 0.5
    return max(256, round(aspect_w * k / 64) * 64), max(256, round(aspect_h * k / 64) * 64)


def meta() -> dict:
    """What the editor needs to draw and build poses the same way."""
    return {"joints": JOINTS, "joints_ko": JOINT_KO, "limbs": LIMBS, "colors": COLORS, "mirror": MIRROR,
            "hand_edges": HAND_EDGES, "hand_colors": HAND_COLORS, "face": FACE, "face_eyes": FACE_EYES,
            "presets": [{"id": k, "name": v[0], "joints": v[1], "z": v[2]} for k, v in PRESETS.items()],
            "hands": [{"id": k, "name": v[0], "points": v[1]} for k, v in HANDS.items()]}
