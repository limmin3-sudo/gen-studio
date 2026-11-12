"""MCP server: lets Codex, ChatGPT and Claude drive Gen Studio.

Tools only pass requests to the local models and hand back results.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mcp.server.apps import Apps
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image as MCPImage
from PIL import Image

from .jobs import CharacterBuildRequest, CutoutRequest, EditRequest, GenerateRequest, SegmentRequest
from .library import flat_rgb

if TYPE_CHECKING:
    from .app import State

INSTRUCTIONS = """Gen Studio runs image models on the user's own GPU (no paid APIs).
- generate_image: text to image. Style is picked automatically (realistic photo, semi-realistic, or text/design
  for posters, signs and lettering) unless given.
- edit_image: change an existing image.
  * instruction (default when no mask): describe the change in words ("make the background a starship bridge,
    keep the person"); extra reference images (outfits, people, places) can be given. Best for SFW edits.
  * mask: white = area to change, only that area is redrawn. fill='auto' lets the model decide what goes there.
  * expand: grow the canvas to a new aspect ratio (e.g. portrait head-shot -> head-to-thigh 3:4, anchor='top').
- Segments: segment_image(image_id) finds the parts (person, hair, shirt, epaulets ...) and returns a numbered
  overview picture plus the list. Then edit_image(image_id, segments=["3"] or ["white collared shirt"],
  prompt="red leather jacket") redraws just those parts; fill="auto" deletes them and lets the model fill in,
  fill="clear" deletes and blends with the surroundings. points=[{x,y}] selects whatever is at a pixel.
- open_editor(image_id) shows a clickable segment editor inside the chat when the host supports MCP Apps
  (ChatGPT, Claude; other hosts get the overview and a link). Prefer it when the user wants to pick parts by hand.
- LoRAs: list_loras / add_lora / train_lora (learn a person or look from a few pictures); pass loras=[{"name": ..., "strength": 0.8}] and the matching model is chosen,
  trigger words are added automatically.
- Characters: build_character(name, image_id) turns ONE photo into a full character (face, full body, side
  shots + description); create_character(name, image ids or paths) saves existing pictures as one.
  Then generate_image(character=name, ...) keeps the same face and outfit across scenes (tutorials, series).
- change_pose(image_id, pose_id or prompt) moves the person into a new pose, same person and background.
- Exact poses: read_pose(image_id) gives the joints; make_pose(people=[{"left_wrist": [0.7, 0.1], ...}],
  start_pose_id=...) moves them (or starts from a preset) and returns a pose_id with a picture to check. Use it
  with generate_image(pose_id=...) or change_pose(pose_id=...).
- VIDEO: use create_video(prompt, image_id=None, seconds=5) — LTX-2.5 with sound, from text or an image — for
  every clip the user will look at. Write a detailed English shot description: subject, action beat by beat,
  camera, lighting, mood and the audio (dialogue in quotes, ambience, effects, music). It is slow (minutes to tens
  of minutes): start it, tell the user, and poll get_job_status. If it says the LTX model is missing, tell the user
  to download it in Gen Studio → 설정 → "Codex 전용 고성능 영상" (needs a Hugging Face token) and offer
  image_to_video as a draft meanwhile.
- Keyframes: create_video(image_id=start, mid_image_id=..., end_image_id=..., keyframes=[{"image_id":..,"at":3.5}])
  pins pictures to moments; make the stills first (generate_image / edit_image / change_pose), then animate between
  them. Everything is adjustable: width/height, seconds (2-20), fps, image_strength, audio on/off, negative, seed.
- Longer films: get_video_frame(video_id, "last") gives the last frame as an image, start the next clip from it,
  then join_videos([clip1, clip2, ...]) makes one file (sound included).
- image_to_video(image_id, prompt) is a quicker Wan 2.2 clip (no sound); it also takes end_image_id/mid_image_id,
  size, steps and smoothing (frame interpolation x2-x4).
- upscale_image(image_id, scale=2|4) makes a 2K/4K version.
- fix_details(image_id, targets=["face","eyes","hands"]) sharpens small faces/hands (after full-body shots).
- remove_background(image_id) keeps the people/objects and makes the rest transparent (or white, #hex ...);
  export it as png/webp to keep the transparency.
- export_image writes a result into any folder (e.g. a project's assets) as webp/png/jpg with a size.

Working style: users speak casually ("디렉터 튜토리얼 장면 하나", "브로커 허벅지까지 늘려줘"). Work out the details
yourself: match people to saved characters (list_characters; loose names and aliases work), follow the project's
AGENTS.md for where and how to save, pick sizes and file names, and only ask when something is truly ambiguous.
Show the result image to the user and ask before overwriting files that already exist.
- Calls wait up to wait_seconds (default 45, set 0 to return at once). If the job is not finished
  they return a job_id: poll get_job_status(job_id, wait_seconds=30) until status is done, then get_image.
Images are referenced by image_id.

Writing prompts (important for quality):
- Always write the prompt in English, even when the user speaks Korean; translate and keep every detail.
- style=semi (Illustrious) understands comma-separated tags best, e.g. "1girl, white collared shirt, black pencil skirt, city street, night, rain".
- style=realistic / wan understand natural sentences: subject, clothing, pose, place, light, camera.
- Text in the picture: put the exact words in double quotes and describe the layout.
- The user's request decides the content; pass it through faithfully."""


PANEL_URI = "ui://gen-studio/panel.html"
PANEL_HTML = (Path(__file__).with_name("panel.html")).read_text(encoding="utf-8")


def _data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def build_mcp(state: "State") -> MCPServer:
    # ---- MCP Apps: an interactive segment editor shown inside the chat where the host supports it.
    # Hosts without MCP Apps ignore the UI and get the same information as text, an overview image and a link.
    apps = Apps()

    @apps.tool(resource_uri=PANEL_URI, visibility=["model", "app"], description=(
        "Open the interactive segment editor for an image (image_id or image_path) so the user can click parts, "
        "delete them, and type edits. Where the chat cannot show the editor, returns the numbered overview and a "
        "link to the full editor."))
    async def open_editor(image_id: str | None = None, image_path: str | None = None):
        if not image_id:
            image_id = _bring_in(None, [image_path] if image_path else None)[0]
        if not state.library.get(image_id):
            raise ValueError(f"이미지를 찾을 수 없어요: {image_id}")
        segs = state.engine.segments.load(image_id)
        info = {"image_id": image_id, "editor_url": _editor_url(image_id), "segments": _seg_list(image_id),
                "note": "패널이 보이지 않으면 editor_url을 브라우저로 여세요." if segs else
                        "아직 세그먼트가 없어요. 패널이 열리면 자동으로 찾고, 패널이 없으면 segment_image를 호출하세요."}
        out: list = [info]
        out.append(MCPImage(data=state.engine.segments.overview(image_id), format="jpeg") if segs else _preview(image_id))
        return out

    @apps.tool(resource_uri=PANEL_URI, visibility=["app"], description="Data for the editor panel (app only).")
    async def panel_state(image_id: str, max_side: int = 1024):
        rec = state.library.get(image_id)
        if not rec:
            raise ValueError(f"이미지를 찾을 수 없어요: {image_id}")
        with Image.open(state.library.path(image_id)) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            pw, ph = im.size
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=86)
        segs = []
        for s in state.engine.segments.load(image_id):
            raw = state.engine.segments.alpha_mask(image_id, s["id"])
            with Image.open(io.BytesIO(raw)) as m:
                m = m.resize((pw, ph), Image.NEAREST)
                mb = io.BytesIO()
                m.save(mb, "PNG", optimize=True)
            segs.append({**{k: s.get(k) for k in ("id", "label", "ko", "group", "instance", "area")},
                         "mask": _data_url(mb.getvalue(), "image/png"),
                         "thumb": _data_url(state.engine.segments.cutout(image_id, s["id"], 80), "image/png")})
        return {"image_id": image_id, "width": rec["width"], "height": rec["height"], "label": rec.get("label"),
                "prompt": rec.get("prompt"), "model_id": rec.get("model_id"),
                "image": _data_url(buf.getvalue(), "image/jpeg"), "segments": segs,
                "versions": [{"id": v["id"], "label": v.get("label")} for v in state.library.versions(image_id)],
                "editor_url": _editor_url(image_id)}

    apps.add_html_resource(PANEL_URI, PANEL_HTML, name="Gen Studio editor panel")

    def _editor_url(image_id: str) -> str:
        return f"{state.public_base()}/#edit={image_id}"

    mcp = MCPServer("gen-studio", title="Gen Studio", instructions=INSTRUCTIONS, version="0.1.0", extensions=[apps])

    def _preview(image_id: str, max_side: int = 1024) -> MCPImage:
        path = state.library.path(image_id)
        if not path:
            raise ValueError(f"이미지를 찾을 수 없어요: {image_id}")
        if (state.library.get(image_id) or {}).get("media") == "video":
            path = state.library.thumb_path(image_id)       # poster frame for clips
        with Image.open(path) as im:
            im = flat_rgb(im)
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=88)
        return MCPImage(data=buf.getvalue(), format="jpeg")

    def _info(image_id: str) -> dict:
        rec = state.library.get(image_id) or {}
        return {
            "image_id": image_id,
            "width": rec.get("width"),
            "height": rec.get("height"),
            "model": rec.get("model_id"),
            "seed": rec.get("seed"),
            "prompt": rec.get("prompt"),
            "label": rec.get("label"),
            "parent_id": rec.get("parent_id"),
            "file": str(state.library.path(image_id)),
            "editor_url": f"{state.public_base()}/#edit={image_id}",
            **({"media": "video", "duration": rec.get("duration"), "fps": rec.get("fps")} if rec.get("media") == "video" else {}),
        }

    def _bring_in(ids: list[str] | None, paths: list[str] | None, b64s: list[str] | None = None) -> list[str]:
        out_ids = list(ids or [])
        for pth in paths or []:
            fp = Path(pth).expanduser()
            if not fp.is_file():
                raise ValueError(f"파일이 없어요: {pth}")
            out_ids.append(state.library.import_image(fp.read_bytes(), fp.name)["id"])
        for b in b64s or []:
            data = b.split(",", 1)[1] if b.startswith("data:") else b
            out_ids.append(state.library.import_image(base64.b64decode(data), "mcp-upload.png")["id"])
        return out_ids

    def _one(iid: str | None, path: str | None) -> str | None:
        return iid or (_bring_in(None, [path])[0] if path else None)

    async def _finish(job, wait_seconds: int, include_image: bool):
        job = await state.engine.wait(job.id, max(0, min(wait_seconds, 600)))
        summary = {"job_id": job.id, "status": job.status, "progress": job.progress, "message": job.message,
                   "decision": job.decision, "error": job.error,
                   "images": [_info(i) for i in job.result_ids]}
        if job.result.get("character"):
            c = job.result["character"]
            summary["character"] = {"name": c["name"], "id": c["id"], "description": c["description"],
                                    "reference_ids": c["ref_ids"]}
        if job.status in ("queued", "running"):
            summary["next"] = "아직 진행 중이에요. get_job_status(job_id)로 확인한 뒤 get_image를 호출하세요."
        out: list = [summary]
        if include_image and job.status == "done":
            out += [_preview(i) for i in job.result_ids]
        return out

    @mcp.tool(description="Generate images from a text prompt on the local GPU. Returns previews and image_ids.")
    async def generate_image(
        prompt: str,
        style: Literal["auto", "realistic", "semi", "text"] = "auto",
        aspect: Literal["auto", "square", "portrait", "landscape", "tall", "wide"] = "auto",
        model: str = "auto",
        negative: str = "",
        seed: int = -1,
        count: int = 1,
        character: str | None = None,
        reference_ids: list[str] | None = None,
        reference_paths: list[str] | None = None,
        loras: list[dict] | None = None,
        denoise: float | None = None,
        regions: list[dict] | None = None,
        pose_id: str | None = None,
        pose_path: str | None = None,
        pose_strength: float = 0.85,
        wait_seconds: int = 45,
        include_image: bool = True,
    ):
        """character: a saved character name. reference_ids / reference_paths: images to follow
        (people, outfits, places, style); refer to them in the prompt as image 1, image 2 ...
        One reference + style='semi' (or "반실사로", "실사화", a non-Qwen model or LoRA) redraws that picture in the
        new look with the style's own model; denoise 0.3-0.8 sets how much changes (default 0.55 semi, 0.6 real).
        regions: [{"box": [left, top, right, bottom] as 0-1 fractions, "prompt": "..."}] (up to 6) to place
        different people or things in different parts of the picture; prompt then describes the whole scene.
        pose_id / pose_path: a photo (or a pose map) whose body pose the new picture copies; the size follows it."""
        refs = _bring_in(reference_ids, reference_paths)
        req = GenerateRequest(prompt=prompt, style=style, aspect=aspect, model=model, negative=negative,
                              seed=seed, count=max(1, min(count, 4)), character=character, references=refs,
                              loras=loras or [], denoise=denoise, regions=regions or [],
                              pose=(pose_id or (_bring_in(None, [pose_path])[0] if pose_path else None)),
                              pose_strength=pose_strength)
        job = state.engine.submit_generate(req)
        return await _finish(job, wait_seconds, include_image)

    @mcp.tool(description=(
        "Edit an image. Give image_id (from earlier results) or image_base64 / image_path to bring in a new image. "
        "mask_base64: PNG where white marks the area to redraw. fill: 'prompt' draws the prompt in that area, "
        "'auto' lets the model decide, 'clear' removes what is there. Without a mask the whole image is redrawn "
        "with strength `denoise` (0.3 subtle .. 0.8 strong)."))
    async def edit_image(
        prompt: str = "",
        image_id: str | None = None,
        image_base64: str | None = None,
        image_path: str | None = None,
        mask_base64: str | None = None,
        segments: list[str] | None = None,
        fill: Literal["prompt", "auto", "clear"] = "prompt",
        mode: Literal["auto", "instruction", "mask", "redraw", "expand"] = "auto",
        reference_ids: list[str] | None = None,
        reference_paths: list[str] | None = None,
        character: str | None = None,
        expand_aspect: str | None = None,
        expand_anchor: Literal["center", "top", "bottom", "left", "right"] = "center",
        loras: list[dict] | None = None,
        denoise: float | None = None,
        model: str = "auto",
        seed: int = -1,
        wait_seconds: int = 45,
        include_image: bool = True,
    ):
        if not image_id:
            if image_base64:
                data = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
                rec = state.library.import_image(base64.b64decode(data), "mcp-upload.png")
            elif image_path:
                p = Path(image_path).expanduser()
                if not p.is_file():
                    raise ValueError(f"파일이 없어요: {image_path}")
                rec = state.library.import_image(p.read_bytes(), p.name)
            else:
                raise ValueError("image_id, image_base64, image_path 중 하나가 필요해요")
            image_id = rec["id"]
        refs = _bring_in(reference_ids, reference_paths)
        if mode == "auto":
            mode = "mask" if (mask_base64 or segments) else "expand" if expand_aspect else "instruction"
        if mode == "instruction" and not any(m.family.encoder == "qwen21" and m.installed(state.settings.models_dir) is not False
                                             for m in state.registry.models):
            mode = "redraw"   # Qwen not installed: fall back to a light whole-image redraw
        internal = {"mask": "inpaint", "redraw": "variation", "instruction": "reference", "expand": "outpaint"}[mode]
        req = EditRequest(image_id=image_id, prompt=prompt, mode=internal, fill=fill, mask_png=mask_base64,
                          denoise=denoise, model=model, seed=seed, references=refs, character=character,
                          aspect=expand_aspect, anchor=expand_anchor, segments=segments or [], loras=loras or [])
        job = state.engine.submit_edit(req)
        return await _finish(job, wait_seconds, include_image)

    @mcp.tool(description=(
        "Cancel a job that is waiting or running (e.g. a long video started with the wrong settings). A running job "
        "stops within seconds; nothing is saved. Start a new one with the changed settings."))
    async def cancel_job(job_id: str):
        if job_id not in state.engine.jobs:
            raise ValueError(f"작업을 찾을 수 없어요: {job_id}")
        ok = await state.engine.cancel(job_id)
        job = await state.engine.wait(job_id, 15)
        return {"job_id": job_id, "canceled": ok, "status": job.status}

    @mcp.tool(description="Check a running job. Returns status, progress and image_ids when done.")
    async def get_job_status(job_id: str, wait_seconds: int = 0):
        job = state.engine.jobs.get(job_id)
        if not job:
            raise ValueError(f"작업을 찾을 수 없어요: {job_id}")
        if wait_seconds:
            job = await state.engine.wait(job_id, min(wait_seconds, 600))
        return {"job_id": job.id, "status": job.status, "progress": job.progress, "message": job.message,
                "queue_position": state.engine.queue_position(job.id), "error": job.error,
                "images": [_info(i) for i in job.result_ids]}

    @mcp.tool(description="Return an image (JPEG preview) and its details by image_id.")
    async def get_image(image_id: str, max_side: int = 1024):
        return [_info(image_id), _preview(image_id, max(256, min(max_side, 2048)))]

    @mcp.tool(description="List recent images with their image_ids, newest first.")
    async def list_images(limit: int = 20):
        return {"images": [_info(r["id"]) for r in state.library.list(limit=max(1, min(limit, 100)))]}

    def _seg_list(image_id: str) -> list[dict]:
        return [{"number": i, "id": s["id"], "label": s["label"], "ko": s.get("ko"), "parent": s.get("parent"),
                 "instance": s.get("instance"), "bbox": s["bbox"], "area": s["area"]}
                for i, s in enumerate(state.engine.segments.load(image_id), 1)]

    @mcp.tool(description=(
        "Find the parts of an image (people, hair, clothing items, accessories, objects, background). Returns a "
        "numbered overview picture and the list. labels: optional English nouns to look for (e.g. ['epaulets']); "
        "points: optional [{x, y}] pixels to select whatever is there. Reuses earlier results unless refresh=true."))
    async def segment_image(image_id: str | None = None, image_path: str | None = None, labels: list[str] | None = None,
                            points: list[dict] | None = None, refresh: bool = False, wait_seconds: int = 45):
        if not image_id:
            image_id = _bring_in(None, [image_path] if image_path else None)[0]
        existing = state.engine.segments.load(image_id)
        if existing and not (refresh or labels or points):
            return [{"image_id": image_id, "segments": _seg_list(image_id)},
                    MCPImage(data=state.engine.segments.overview(image_id), format="jpeg")]
        job = state.engine.submit_segment(SegmentRequest(image_id=image_id, labels=labels, points=points))
        job = await state.engine.wait(job.id, max(0, min(wait_seconds, 600)))
        if job.status != "done":
            return {"image_id": image_id, "job_id": job.id, "status": job.status, "error": job.error,
                    "next": "get_job_status(job_id) 후 segment_image(image_id)를 다시 호출하세요."}
        return [{"image_id": image_id, "segments": _seg_list(image_id)},
                MCPImage(data=state.engine.segments.overview(image_id), format="jpeg")]

    @mcp.tool(description=(
        "Build a body pose by placing joints (the skeleton editor, for Codex). people = one dict per person: joint "
        "name -> [x, y], 0..1 of the canvas (0,0 top-left), the person's right side is on the picture's left when "
        "facing the camera. Joints: nose, neck, right_/left_ shoulder, elbow, wrist, hip, knee, ankle, right_/left_ "
        "eye, ear. Give only the joints you want to move: the rest come from start_pose_id (the same person index) "
        "or from preset ('stand', 'arms_up', 'walk', 'sit', 'hands_hips'; a person dict may also carry its own "
        "\"preset\"). A joint set to null is hidden. Canvas width/height set the picture shape. Returns pose_id and "
        "a picture of the skeleton to check; then generate_image(pose_id=...) or change_pose(pose_id=...)."))
    async def make_pose(people: list[dict], width: int = 768, height: int = 1152, name: str = "",
                        start_pose_id: str | None = None, preset: str = "stand"):
        from . import poses
        from .jobs import PoseSaveRequest
        base = []
        if start_pose_id:
            rec = state.library.get(start_pose_id) or {}
            base = (rec.get("keypoints") or {}).get("people") or []
            if not base:
                raise ValueError("start_pose_id에 관절 정보가 없어요. read_pose로 먼저 읽어 주세요.")
        out = []
        for i, person in enumerate(people or [{}]):
            person = dict(person or {})
            p = person.pop("preset", None)
            start = poses.preset(p, width, height) if p else base[i] if i < len(base) else poses.preset(preset, width, height)
            out.append(poses.from_named(person, start))
        rec = state.engine.save_pose(PoseSaveRequest(people=out, width=width, height=height, name=name,
                                                     parent_id=start_pose_id))
        return [{"pose_id": rec["id"], "width": width, "height": height,
                 "people": [poses.named(p) for p in rec["keypoints"]["people"]],
                 "next": "generate_image(prompt, pose_id=...) 또는 change_pose(image_id, pose_id=...)"},
                _preview(rec["id"])]

    @mcp.tool(description=(
        "Read the body joints of every person in a picture (or a saved pose). Returns pose_id and the joints as "
        "name -> [x, y] (0..1), ready to adjust with make_pose(start_pose_id=pose_id, people=[{...moved joints}])."))
    async def read_pose(image_id: str | None = None, image_path: str | None = None, wait_seconds: int = 60):
        from . import poses
        image_id = _one(image_id, image_path)
        if not image_id:
            raise ValueError("image_id나 image_path가 필요해요")
        job = state.engine.submit_read_pose(image_id)
        job = await state.engine.wait(job.id, max(0, min(wait_seconds, 600)))
        if job.status != "done":
            return {"job_id": job.id, "status": job.status, "error": job.error}
        rec = state.library.get(job.result["pose_id"])
        return [{"pose_id": rec["id"], "width": rec["width"], "height": rec["height"],
                 "people": [poses.named(p) for p in job.result["keypoints"]["people"]]}, _preview(rec["id"])]

    @mcp.tool(description=(
        "Change the pose of the person in an image, keeping the same person, clothing and background. Give a photo "
        "whose pose to copy (pose_id / pose_path) and/or describe it in prompt ('sitting on the bed, legs crossed')."))
    async def change_pose(image_id: str | None = None, image_path: str | None = None, pose_id: str | None = None,
                          pose_path: str | None = None, prompt: str = "", wait_seconds: int = 90,
                          include_image: bool = True):
        from .jobs import ReposeRequest
        if not image_id:
            image_id = _bring_in(None, [image_path] if image_path else None)[0]
        if pose_path and not pose_id:
            pose_id = _bring_in(None, [pose_path])[0]
        job = state.engine.submit_repose(ReposeRequest(image_id=image_id, pose=pose_id, prompt=prompt))
        return await _finish(job, wait_seconds, include_image)

    @mcp.tool(description=(
        "High-quality video WITH SOUND (LTX-2.5 22B) from a text prompt, or from an image (image_id / image_path) "
        "as the first shot. Slow (several minutes to tens of minutes on the home GPU) but much better than "
        "image_to_video. Write the prompt like a shot description in English: subject and look, action beat by beat, "
        "camera (angle, lens, movement), lighting, mood, and the sound (dialogue in quotes with who says it, "
        "ambience, effects, music). "
        "Keyframes: image_id = first shot, end_image_id = last shot, mid_image_id = the middle, keyframes = "
        "[{image_id, at: seconds}] for any other moments (each id may instead be a file path in *_path / "
        "keyframes[].path). With keyframes it runs ComfyUI's First & Last Frame recipe (one pass, full size). "
        "Knobs: seconds 2-20 (over 10 is slow and tight on 16GB), quality 540p/720p or width/height directly "
        "(multiples of 64), orientation auto/landscape/portrait/square, fps 12-50 (24 default), image_strength "
        "0.1-1 (how strictly the pictures are followed, 0.7 default), audio true/false, negative, seed. "
        "Returns a job_id: poll get_job_status; the finished mp4 path is in the result (export_image copies it)."))
    async def create_video(prompt: str, image_id: str | None = None, image_path: str | None = None, seconds: int = 5,
                           quality: str = "720p", orientation: str = "auto", negative: str = "", seed: int = -1,
                           end_image_id: str | None = None, end_image_path: str | None = None,
                           mid_image_id: str | None = None, mid_image_path: str | None = None,
                           keyframes: list[dict] | None = None, width: int | None = None, height: int | None = None,
                           fps: int = 24, image_strength: float = 0.7, audio: bool = True, wait_seconds: int = 30):
        from .jobs import LtxRequest
        image_id = _one(image_id, image_path)
        keys = [{"image_id": _one(k.get("image_id"), k.get("path")), "at": float(k.get("at", 0))}
                for k in keyframes or []]
        job = state.engine.submit_ltx(LtxRequest(
            prompt=prompt, image_id=image_id, end_image_id=_one(end_image_id, end_image_path),
            mid_image_id=_one(mid_image_id, mid_image_path), keyframes=keys, seconds=max(2, min(int(seconds), 20)),
            quality="540p" if str(quality).startswith("540") else "720p",
            orientation=orientation if orientation in ("auto", "landscape", "portrait", "square") else "auto",
            width=width, height=height, fps=fps, image_strength=image_strength, audio=audio,
            negative=negative, seed=seed))
        job = await state.engine.wait(job.id, max(0, min(wait_seconds, 600)))
        out = {"job_id": job.id, "status": job.status, "message": job.message, "error": job.error,
               "decision": job.decision}
        if job.status == "done":
            out["video"] = _info(job.result_ids[0])
        else:
            out["next"] = "오래 걸려요. get_job_status(job_id, wait_seconds=60)로 가끔 확인하세요."
        return out

    @mcp.tool(description=(
        "Quicker video from an image (Wan 2.2 14B DaSiWa, 16 fps, no sound). Lower quality than create_video: use "
        "it for a fast preview or when the user asks for this model. prompt: how things move, in English (empty = "
        "natural subtle motion). end_image_id / end_image_path = last shot; mid_image_id / mid_image_path = middle "
        "shot (needs an end shot; two halves are made and joined). seconds 2-10 (5 is what the model knows best; "
        "with a middle shot up to 10 stays natural), quality 720p (default) or 480p, or width/height directly "
        "(multiples of 16), steps 2-30 (4 default), smooth = frame interpolation on/off, smooth_multiplier 2-4 "
        "(32/48/64 fps), negative, seed. Returns a job_id to poll; the result is an mp4 in the library."))
    async def image_to_video(image_id: str | None = None, image_path: str | None = None, prompt: str = "",
                             seconds: int = 5, quality: str = "720p", end_image_id: str | None = None,
                             end_image_path: str | None = None, mid_image_id: str | None = None,
                             mid_image_path: str | None = None, width: int | None = None, height: int | None = None,
                             steps: int = 4, smooth: bool = True, smooth_multiplier: int = 2, negative: str = "",
                             seed: int = -1, wait_seconds: int = 30):
        from .jobs import VideoRequest
        if not image_id:
            image_id = _bring_in(None, [image_path] if image_path else None)[0]
        job = state.engine.submit_video(VideoRequest(
            image_id=image_id, prompt=prompt, seconds=max(2, min(int(seconds), 10)),
            quality="720p" if str(quality).startswith("720") else "480p",
            end_image_id=_one(end_image_id, end_image_path), mid_image_id=_one(mid_image_id, mid_image_path),
            width=width, height=height, steps=steps, smooth=smooth, smooth_multiplier=max(2, min(int(smooth_multiplier), 4)),
            negative=negative, seed=seed))
        job = await state.engine.wait(job.id, max(0, min(wait_seconds, 600)))
        out = {"job_id": job.id, "status": job.status, "message": job.message, "error": job.error,
               "decision": job.decision}
        if job.status == "done":
            out["video"] = _info(job.result_ids[0])
        else:
            out["next"] = "영상은 몇 분 걸려요. get_job_status(job_id)로 확인하세요."
        return out

    @mcp.tool(description=(
        "Take one frame out of a video as a new image: at = 'first', 'last' (default) or a time in seconds. Use the "
        "last frame as the first shot of the next clip to continue a scene, or edit it like any picture."))
    async def get_video_frame(video_id: str, at: str | float = "last", wait_seconds: int = 60,
                              include_image: bool = True):
        from .jobs import VideoFrameRequest
        a = at if at in ("first", "last") else float(at)
        job = state.engine.submit_video_frame(VideoFrameRequest(video_id=video_id, at=a))
        return await _finish(job, wait_seconds, include_image)

    @mcp.tool(description=(
        "Join clips one after another into one mp4 (2-20 video ids, in order). Clips may have different frame "
        "rates (e.g. Wan 32 fps + LTX 24 fps): the result runs at the highest one and every clip keeps its own "
        "length and speed. Later clips are scaled to the first one's size. audio: keep the sound (default: when "
        "any clip has sound; silent clips get silence so sound stays in sync)."))
    async def join_videos(video_ids: list[str], audio: bool | None = None, wait_seconds: int = 60):
        from .jobs import JoinVideosRequest
        job = state.engine.submit_join(JoinVideosRequest(video_ids=video_ids, audio=audio))
        job = await state.engine.wait(job.id, max(0, min(wait_seconds, 600)))
        out = {"job_id": job.id, "status": job.status, "message": job.message, "error": job.error,
               "decision": job.decision}
        if job.status == "done":
            out["video"] = _info(job.result_ids[0])
        return out

    @mcp.tool(description=(
        "Upscale an image 2x (about 2K) or 4x (about 4K). refine (default on) lightly redraws at the new size so "
        "skin, hair and fabric get real detail. Saved as a new version."))
    async def upscale_image(image_id: str | None = None, image_path: str | None = None, scale: int = 2,
                            refine: bool | None = None, wait_seconds: int = 120, include_image: bool = True):
        from .jobs import UpscaleRequest
        if not image_id:
            image_id = _bring_in(None, [image_path] if image_path else None)[0]
        job = state.engine.submit_upscale(UpscaleRequest(image_id=image_id, scale=4 if scale >= 4 else 2, refine=refine))
        return await _finish(job, wait_seconds, include_image)

    @mcp.tool(description=(
        "Fix small or blurry faces, eyes or hands: finds each one, redraws it enlarged and blends it back. "
        "targets: face, eyes, hands (default face). strength 0.3 keeps the face, 0.5-0.6 changes more. "
        "Use after full-body or group pictures."))
    async def fix_details(image_id: str | None = None, image_path: str | None = None, targets: list[str] | None = None,
                          strength: float = 0.4, prompt: str = "", wait_seconds: int = 90, include_image: bool = True):
        from .detail import DetailRequest
        if not image_id:
            image_id = _bring_in(None, [image_path] if image_path else None)[0]
        job = state.engine.submit_detail(DetailRequest(image_id=image_id, targets=targets or ["face"],
                                                       strength=strength, prompt=prompt))
        return await _finish(job, wait_seconds, include_image)

    @mcp.tool(description=(
        "Remove the background: keeps the main people/objects and makes everything else transparent. "
        "background: 'transparent' (default), 'white', 'black', 'gray' or '#rrggbb'. Saved as a new version."))
    async def remove_background(image_id: str | None = None, image_path: str | None = None,
                                background: str = "transparent", wait_seconds: int = 45, include_image: bool = True):
        if not image_id:
            image_id = _bring_in(None, [image_path] if image_path else None)[0]
        job = state.engine.submit_cutout(CutoutRequest(image_id=image_id, background=background))
        return await _finish(job, wait_seconds, include_image)

    @mcp.tool(description="Save a character (name + 1-6 reference images) so later generations keep the same person.")
    async def create_character(name: str, image_ids: list[str] | None = None, image_paths: list[str] | None = None,
                               description: str = "", aliases: list[str] | None = None, replace: bool = False):
        """aliases: other names people use, e.g. ["디렉터", "director"]. replace=true updates an existing one."""
        ids = _bring_in(image_ids, image_paths)
        existing = state.engine.characters.get(name)
        c = state.engine.characters.save(name, ids, description, existing["id"] if (existing and replace) else None, aliases)
        return {"character": c["name"], "id": c["id"], "aliases": c["aliases"], "reference_ids": c["ref_ids"]}

    @mcp.tool(description=(
        "Make a character from ONE photo: creates clean reference shots of the same person (views: face, full, "
        "three_quarter, profile; default face+full+three_quarter), reads their look into a description, and saves "
        "them all as a character for generate_image(character=name). Takes about a minute per view."))
    async def build_character(name: str, image_id: str | None = None, image_path: str | None = None,
                              views: list[str] | None = None, aliases: list[str] | None = None,
                              wait_seconds: int = 120, include_image: bool = True):
        if not image_id:
            image_id = _bring_in(None, [image_path] if image_path else None)[0]
        req = CharacterBuildRequest(name=name, image_id=image_id, aliases=aliases or [],
                                    **({"views": views} if views else {}))
        return await _finish(state.engine.submit_character_build(req), wait_seconds, include_image)

    @mcp.tool(description=(
        "Diagnostic report (versions, GPU/RAM, missing models or ComfyUI nodes, recent failed jobs with errors, end "
        "of the ComfyUI log). Call it when a job fails or something seems broken, before guessing."))
    async def get_diagnostics():
        from . import config as cfg
        from .diagnostics import build
        return await build(state, [str(v) for v in cfg.read_secrets().values() if isinstance(v, str)])

    @mcp.tool(description="List saved characters.")
    async def list_characters():
        return {"characters": [{"name": c["name"], "id": c["id"], "aliases": c.get("aliases", []),
                                "description": c["description"], "reference_ids": c["ref_ids"]}
                               for c in state.engine.characters.list()]}

    @mcp.tool(description=(
        "Copy an image into a folder on this PC, e.g. a project's assets. format: webp/png/jpg (default from the "
        "file extension). Resize with max_side, or width/height (one keeps the aspect). Refuses to overwrite unless "
        "overwrite=true."))
    async def export_image(image_id: str, path: str, format: str | None = None, max_side: int | None = None,
                           width: int | None = None, height: int | None = None, quality: int = 90,
                           overwrite: bool = False):
        return state.library.export(image_id, Path(path), format, max_side, width, height, quality, overwrite)

    @mcp.tool(description=(
        "List installed LoRAs (style, character, outfit add-ons) with the model family each works with and its "
        "trigger words. Use them as loras=[{\"name\": ..., \"strength\": 0.8}] in generate_image / edit_image; the "
        "right model is picked automatically."))
    async def list_loras():
        return {"loras": [{"name": x["name"], "id": x["id"], "family": x.get("family"), "family_name": x["family_name"],
                           "triggers": x.get("triggers"), "strength": x.get("strength"), "present": x["present"],
                           "note": x.get("note")} for x in state.engine.loras.list()]}

    @mcp.tool(description=(
        "Train a LoRA on this PC from 3-60 pictures (image ids/paths, or a saved character's pictures). "
        "kind='character' learns a person, 'style' learns a look. family auto/illustrious(semi)/zimage(realistic). "
        "Takes 20-60 minutes; returns a job_id to poll with get_job_status. The finished LoRA appears in list_loras "
        "with its trigger word and is used with loras=[{name}]."))
    async def train_lora(name: str, image_ids: list[str] | None = None, image_paths: list[str] | None = None,
                         character: str | None = None, kind: str = "character", family: str = "auto",
                         trigger: str | None = None, steps: int | None = None, rank: int = 16):
        from .training import TrainRequest
        ids = _bring_in(image_ids, image_paths)
        job = state.engine.submit_train(TrainRequest(name=name, image_ids=ids, character=character, kind=kind,
                                                     family=family, trigger=trigger, steps=steps, rank=rank))
        return {"job_id": job.id, "status": job.status, "decision": job.decision,
                "next": "학습은 오래 걸려요. get_job_status(job_id)로 가끔 확인하세요."}

    @mcp.tool(description=(
        "Add a LoRA from a Civitai page URL (downloads in the background) or from a .safetensors path on this PC."))
    async def add_lora(civitai_url: str | None = None, path: str | None = None, wait_seconds: int = 60):
        import asyncio
        L = state.engine.loras
        if path:
            p = Path(path).expanduser()
            if not p.is_file():
                raise ValueError(f"파일이 없어요: {path}")
            return await L.add_file(p.name, p.read_bytes())
        if not civitai_url:
            raise ValueError("civitai_url 또는 path가 필요해요")
        did = await L.add_from_civitai(civitai_url)
        for _ in range(max(1, wait_seconds)):
            if L.downloads[did]["status"] != "running":
                break
            await asyncio.sleep(1)
        d = L.downloads[did]
        return {**d, "lora": L.get(d["lora_id"]) if d.get("lora_id") else None}

    @mcp.tool(description=(
        "Which model to use for a task, with the recommended one marked and why. task: generate | inpaint | redraw | "
        "instruction | expand. Pass the chosen id as `model` to generate_image / edit_image, or leave model='auto'."))
    async def recommend_model(task: str = "generate", prompt: str = "", image_id: str | None = None):
        return state.engine.recommend(task, prompt, image_id)

    @mcp.tool(description="List the image models, what each is best for, and whether they are installed.")
    async def list_models():
        return {"models": state.registry.status(state.settings.models_dir)}

    return mcp
