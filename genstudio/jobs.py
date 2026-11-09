"""Job queue: runs one generation/edit at a time on the GPU and records the results."""
from __future__ import annotations

import asyncio
import base64
import io
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from . import router, workflows
from .comfy import ComfyClient, ComfyError
from .config import Settings
from .library import Library
from .registry import Registry

CLEAR_PROMPT = "empty background, nothing there, seamless continuation of the surroundings, clean"


# ------------------------------------------------------------------ requests
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="그리고 싶은 내용")
    negative: str = ""
    style: Literal["auto", "realistic", "semi"] = "auto"
    model: str = "auto"
    aspect: Literal["auto", "square", "portrait", "landscape", "tall", "wide"] = "auto"
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    seed: int = -1
    count: int = Field(1, ge=1, le=4)
    steps: int | None = Field(None, ge=1, le=80)
    cfg: float | None = Field(None, ge=0, le=20)


class EditRequest(BaseModel):
    image_id: str
    prompt: str = ""
    negative: str = ""
    # inpaint: redraw only the mask; variation: redraw the whole image lightly
    mode: Literal["inpaint", "variation"] = "inpaint"
    # for inpaint: prompt = draw what the prompt says, auto = let the model decide, clear = remove and blend
    fill: Literal["prompt", "auto", "clear"] = "prompt"
    mask_png: str | None = Field(None, description="base64 PNG, white = area to redraw")
    denoise: float | None = Field(None, ge=0.05, le=1.0)
    model: str = "auto"
    seed: int = -1
    label: str | None = None


@dataclass
class Job:
    id: str
    kind: str
    request: dict
    status: str = "queued"           # queued | running | done | error | canceled
    progress: float = 0.0
    message: str = "대기 중"
    decision: dict = field(default_factory=dict)
    result_ids: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        d["request"] = {k: v for k, v in self.request.items() if k != "mask_png"}
        return d


class Engine:
    def __init__(self, settings: Settings, registry: Registry, library: Library, comfy: ComfyClient):
        self.settings = settings
        self.registry = registry
        self.library = library
        self.comfy = comfy
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._last_family: str | None = None
        self._running: str | None = None
        self._done_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------- control
    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass

    def route(self, req: GenerateRequest) -> router.Decision:
        prompt, _neg = router.split_negative(req.prompt)
        return router.choose(
            self.registry, prompt, models_dir=self.settings.models_dir, style=req.style, model_id=req.model,
            aspect=req.aspect, width=req.width, height=req.height, default_style=self.settings.default_style,
        )

    def submit_generate(self, req: GenerateRequest) -> Job:
        d = self.route(req)
        job = Job(id=uuid.uuid4().hex[:12], kind="generate", request=req.model_dump(), decision=d.as_dict())
        return self._enqueue(job)

    def submit_edit(self, req: EditRequest) -> Job:
        src = self.library.get(req.image_id)
        if not src:
            raise KeyError("편집할 이미지를 찾을 수 없어요")
        if req.mode == "inpaint" and not req.mask_png:
            raise ValueError("다시 그릴 영역(마스크)이 없어요. 브러시로 영역을 칠하거나 세그먼트를 선택하세요.")
        if req.mode == "inpaint" and req.fill == "prompt" and not req.prompt.strip():
            raise ValueError("무엇을 그릴지 입력하거나 '모델이 알아서 채우기'를 고르세요.")
        if req.model and req.model != "auto":
            model, why = self.registry.get(req.model), "직접 고른 모델"
        else:
            model, why = router.model_for_edit(self.registry, src.get("model_id"), self.settings.models_dir,
                                               self.settings.default_style)
        decision = {"model_id": model.id, "model_name": model.name, "family": model.family.id, "reasons": [why],
                    "action": _action_label(req)}
        job = Job(id=uuid.uuid4().hex[:12], kind="edit", request=req.model_dump(), decision=decision)
        return self._enqueue(job)

    def _enqueue(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self._done_events[job.id] = asyncio.Event()
        self._queue.put_nowait(job.id)
        self._trim()
        return job

    def _trim(self, keep: int = 200) -> None:
        if len(self.jobs) > keep:
            for jid in sorted(self.jobs, key=lambda j: self.jobs[j].created_at)[: len(self.jobs) - keep]:
                if self.jobs[jid].status in ("done", "error", "canceled"):
                    self.jobs.pop(jid, None)
                    self._done_events.pop(jid, None)

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status in ("done", "error", "canceled"):
            return False
        if job.status == "queued":
            job.status, job.message = "canceled", "취소됨"
            self._finish(job)
            return True
        await self.comfy.interrupt()
        return True

    async def wait(self, job_id: str, timeout: float) -> Job:
        ev = self._done_events.get(job_id)
        if ev:
            try:
                await asyncio.wait_for(ev.wait(), timeout)
            except asyncio.TimeoutError:
                pass
        return self.jobs[job_id]

    def queue_position(self, job_id: str) -> int:
        waiting = [j for j in self.jobs.values() if j.status == "queued"]
        waiting.sort(key=lambda j: j.created_at)
        ids = [j.id for j in waiting]
        return ids.index(job_id) + 1 if job_id in ids else 0

    # -------------------------------------------------------------- worker
    async def _loop(self) -> None:
        while True:
            jid = await self._queue.get()
            job = self.jobs.get(jid)
            if not job or job.status != "queued":
                continue
            job.status, job.started_at, job.message = "running", time.time(), "준비 중"
            self._running = jid
            try:
                if job.kind == "generate":
                    await self._run_generate(job)
                else:
                    await self._run_edit(job)
                job.status, job.progress, job.message = "done", 1.0, "완료"
            except ComfyError as e:
                job.status, job.error, job.message = "error", str(e), "실패"
            except Exception as e:  # keep the worker alive no matter what
                job.status, job.error, job.message = "error", f"{type(e).__name__}: {e}", "실패"
            finally:
                self._running = None
                self._finish(job)

    def _finish(self, job: Job) -> None:
        job.finished_at = time.time()
        ev = self._done_events.get(job.id)
        if ev:
            ev.set()

    async def _switch_family(self, family_id: str) -> None:
        # different model families rarely fit in 16GB together; unload before switching
        if self._last_family and self._last_family != family_id:
            await self.comfy.free()
        self._last_family = family_id

    def _progress(self, job: Job, base: float, span: float):
        def cb(p: float, text: str) -> None:
            job.progress = round(base + span * p, 3)
            job.message = text
        return cb

    async def _run_generate(self, job: Job) -> None:
        req = GenerateRequest(**job.request)
        model = self.registry.get(job.decision["model_id"])
        prompt, neg_inline = router.split_negative(req.prompt)
        negative = ", ".join(x for x in (req.negative, neg_inline) if x)
        await self._switch_family(model.family.id)
        n = req.count
        for i in range(n):
            seed = req.seed + i if req.seed >= 0 else random.randint(0, 2**31 - 1)
            graph, extra = workflows.txt2img(
                model, prompt, negative, job.decision["width"], job.decision["height"], seed,
                models_dir=self.settings.models_dir, steps=req.steps, cfg=req.cfg)
            outs = await self.comfy.run(graph, self._progress(job, i / n, 1 / n))
            data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
            rec = self.library.add(data, {
                "kind": "generate", "source": "gen-studio", "prompt": prompt, "negative": negative,
                "final_prompt": extra["final_prompt"], "model_id": model.id, "family": model.family.id,
                "style": job.decision["style"], "seed": seed, "steps": req.steps or model.family.steps,
                "cfg": req.cfg if req.cfg is not None else model.family.cfg, "label": "원본",
            })
            job.result_ids.append(rec["id"])

    async def _run_edit(self, job: Job) -> None:
        req = EditRequest(**job.request)
        src = self.library.get(req.image_id)
        src_path = self.library.path(req.image_id)
        if not src or not src_path:
            raise ComfyError("편집할 이미지가 사라졌어요")
        model = self.registry.get(job.decision["model_id"])
        await self._switch_family(model.family.id)
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
        image_name = await self.comfy.upload_image(src_path.read_bytes(), f"{src['id']}.png")

        base_prompt = src.get("prompt", "") or ""
        if req.mode == "inpaint":
            if req.fill == "auto":
                prompt = base_prompt            # model fills to match the whole picture
            elif req.fill == "clear":
                prompt = CLEAR_PROMPT
            else:
                prompt = req.prompt.strip()
            mask = _normalize_mask(req.mask_png, (src["width"], src["height"]))
            mask_bytes = mask
            mask_name = await self.comfy.upload_image(mask, f"{src['id']}-mask-{job.id}.png")
            graph, extra = workflows.inpaint(model, image_name, mask_name, prompt, req.negative, seed,
                                             denoise=req.denoise or 1.0, models_dir=self.settings.models_dir)
        else:
            mask_bytes = None
            prompt = req.prompt.strip() or base_prompt
            graph, extra = workflows.img2img(model, image_name, prompt, req.negative, seed,
                                             req.denoise or 0.55, models_dir=self.settings.models_dir)
        outs = await self.comfy.run(graph, self._progress(job, 0, 1))
        data = await self.comfy.view(outs[0]["filename"], outs[0].get("subfolder", ""))
        rec = self.library.add(data, {
            "kind": "edit", "source": src.get("source", "gen-studio"), "parent_id": src["id"],
            "prompt": base_prompt if req.mode == "inpaint" else prompt, "edit_prompt": req.prompt,
            "final_prompt": extra["final_prompt"], "model_id": model.id, "family": model.family.id, "seed": seed,
            "edit": {"mode": req.mode, "fill": req.fill, "denoise": req.denoise}, "label": req.label or _action_label(req),
        })
        if mask_bytes:
            self.library.update(rec["id"], mask_file=self.library.save_side_file(rec["id"], ".mask.png", mask_bytes))
        job.result_ids.append(rec["id"])

    def regenerate(self, image_id: str) -> Job:
        """Same request as the one that made this image, with a new seed."""
        rec = self.library.get(image_id)
        if not rec:
            raise KeyError("이미지를 찾을 수 없어요")
        if rec.get("kind") == "generate":
            return self.submit_generate(GenerateRequest(
                prompt=rec.get("prompt") or " ", negative=rec.get("negative", ""), model=rec.get("model_id") or "auto",
                width=rec["width"], height=rec["height"], steps=rec.get("steps"), cfg=rec.get("cfg")))
        if rec.get("kind") == "edit" and rec.get("parent_id"):
            edit = rec.get("edit") or {}
            mask_b64 = None
            if rec.get("mask_file"):
                mask_b64 = base64.b64encode((self.library.root / rec["mask_file"]).read_bytes()).decode()
            return self.submit_edit(EditRequest(
                image_id=rec["parent_id"], prompt=rec.get("edit_prompt", ""), mode=edit.get("mode", "variation"),
                fill=edit.get("fill", "prompt"), mask_png=mask_b64, denoise=edit.get("denoise"),
                model=rec.get("model_id") or "auto", label=rec.get("label")))
        raise ValueError("불러온 원본 이미지는 재생성할 수 없어요. 편집으로 바꿔 보세요.")


def _action_label(req: EditRequest) -> str:
    if req.mode == "variation":
        return f"전체 다시 그리기: {req.prompt.strip()}" if req.prompt.strip() else "전체 다시 그리기"
    if req.fill == "auto":
        return "모델이 알아서 채우기"
    if req.fill == "clear":
        return "지우고 비우기"
    return f"부분 다시 그리기: {req.prompt.strip()}"


def _normalize_mask(b64: str | None, size: tuple[int, int]) -> bytes:
    """Accepts any PNG (alpha or white strokes) and returns a black/white RGB mask at the image size."""
    if not b64:
        raise ValueError("마스크가 비어 있어요")
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            m = im.convert("RGBA").split()[3]
        else:
            m = im.convert("L")
        m = m.resize(size, Image.NEAREST).point(lambda v: 255 if v > 20 else 0)
        if m.getbbox() is None:
            raise ValueError("다시 그릴 영역이 비어 있어요")
        buf = io.BytesIO()
        Image.merge("RGB", (m, m, m)).save(buf, "PNG")
        return buf.getvalue()
