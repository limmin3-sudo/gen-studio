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
- edit_image: change an existing image. With a mask (white = area to change) only that area is redrawn;
  without a mask the whole image is lightly redrawn. fill='auto' lets the model decide what goes in the area.
- Long jobs return a job_id; poll get_job_status, then get_image.
Images are referenced by image_id. Prompts may be Korean or English."""


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
        wait_seconds: int = 180,
        include_image: bool = True,
    ):
        req = GenerateRequest(prompt=prompt, style=style, aspect=aspect, model=model, negative=negative,
                              seed=seed, count=max(1, min(count, 4)))
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
        denoise: float | None = None,
        model: str = "auto",
        seed: int = -1,
        wait_seconds: int = 180,
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
        mode = "inpaint" if mask_base64 else "variation"
        req = EditRequest(image_id=image_id, prompt=prompt, mode=mode, fill=fill, mask_png=mask_base64,
                          denoise=denoise, model=model, seed=seed)
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

    @mcp.tool(description="List the image models, their style (realistic/semi) and whether they are installed.")
    async def list_models():
        return state.registry.status(state.settings.models_dir)

    return mcp
