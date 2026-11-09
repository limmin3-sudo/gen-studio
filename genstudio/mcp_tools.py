"""MCP server: lets Codex, ChatGPT and Claude drive Gen Studio.

Tools only pass requests to the local models and hand back results.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image as MCPImage
from PIL import Image

from .jobs import EditRequest, GenerateRequest

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
- Characters: create_character(name, image ids or paths) once, then generate_image(character=name, ...) keeps
  the same face and outfit across scenes (tutorials, series).
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
- style=text (Qwen): put the exact words to render in double quotes and describe layout.
- The user's request decides the content; pass it through faithfully."""


def build_mcp(state: "State") -> MCPServer:
    mcp = MCPServer("gen-studio", title="Gen Studio", instructions=INSTRUCTIONS, version="0.1.0")

    def _preview(image_id: str, max_side: int = 1024) -> MCPImage:
        path = state.library.path(image_id)
        if not path:
            raise ValueError(f"이미지를 찾을 수 없어요: {image_id}")
        with Image.open(path) as im:
            im = im.convert("RGB")
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

    async def _finish(job, wait_seconds: int, include_image: bool):
        job = await state.engine.wait(job.id, max(0, min(wait_seconds, 600)))
        summary = {"job_id": job.id, "status": job.status, "progress": job.progress, "message": job.message,
                   "decision": job.decision, "error": job.error,
                   "images": [_info(i) for i in job.result_ids]}
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
        wait_seconds: int = 45,
        include_image: bool = True,
    ):
        """character: a saved character name. reference_ids / reference_paths: images to follow
        (people, outfits, places, style); refer to them in the prompt as image 1, image 2 ..."""
        refs = _bring_in(reference_ids, reference_paths)
        req = GenerateRequest(prompt=prompt, style=style, aspect=aspect, model=model, negative=negative,
                              seed=seed, count=max(1, min(count, 4)), character=character, references=refs)
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
        fill: Literal["prompt", "auto", "clear"] = "prompt",
        mode: Literal["auto", "instruction", "mask", "redraw", "expand"] = "auto",
        reference_ids: list[str] | None = None,
        reference_paths: list[str] | None = None,
        character: str | None = None,
        expand_aspect: str | None = None,
        expand_anchor: Literal["center", "top", "bottom", "left", "right"] = "center",
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
            mode = "mask" if mask_base64 else "expand" if expand_aspect else "instruction"
        if mode == "instruction" and not any(m.family.encoder == "qwen21" and m.installed(state.settings.models_dir) is not False
                                             for m in state.registry.models):
            mode = "redraw"   # Qwen not installed: fall back to a light whole-image redraw
        internal = {"mask": "inpaint", "redraw": "variation", "instruction": "reference", "expand": "outpaint"}[mode]
        req = EditRequest(image_id=image_id, prompt=prompt, mode=internal, fill=fill, mask_png=mask_base64,
                          denoise=denoise, model=model, seed=seed, references=refs, character=character,
                          aspect=expand_aspect, anchor=expand_anchor)
        job = state.engine.submit_edit(req)
        return await _finish(job, wait_seconds, include_image)

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
        return [_info(r["id"]) for r in state.library.list(limit=max(1, min(limit, 100)))]

    @mcp.tool(description="Save a character (name + 1-6 reference images) so later generations keep the same person.")
    async def create_character(name: str, image_ids: list[str] | None = None, image_paths: list[str] | None = None,
                               description: str = "", aliases: list[str] | None = None, replace: bool = False):
        """aliases: other names people use, e.g. ["디렉터", "director"]. replace=true updates an existing one."""
        ids = _bring_in(image_ids, image_paths)
        existing = state.engine.characters.get(name)
        c = state.engine.characters.save(name, ids, description, existing["id"] if (existing and replace) else None, aliases)
        return {"character": c["name"], "id": c["id"], "aliases": c["aliases"], "reference_ids": c["ref_ids"]}

    @mcp.tool(description="List saved characters.")
    async def list_characters():
        return [{"name": c["name"], "id": c["id"], "aliases": c.get("aliases", []), "description": c["description"],
                 "reference_ids": c["ref_ids"]} for c in state.engine.characters.list()]

    @mcp.tool(description=(
        "Copy an image into a folder on this PC, e.g. a project's assets. format: webp/png/jpg (default from the "
        "file extension). Resize with max_side, or width/height (one keeps the aspect). Refuses to overwrite unless "
        "overwrite=true."))
    async def export_image(image_id: str, path: str, format: str | None = None, max_side: int | None = None,
                           width: int | None = None, height: int | None = None, quality: int = 90,
                           overwrite: bool = False):
        return state.library.export(image_id, Path(path), format, max_side, width, height, quality, overwrite)

    @mcp.tool(description="List the image models, their style (realistic/semi) and whether they are installed.")
    async def list_models():
        return state.registry.status(state.settings.models_dir)

    return mcp
