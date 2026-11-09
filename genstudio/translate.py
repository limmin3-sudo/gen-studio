"""Korean -> English prompt rewriting for models that only follow English well."""
from __future__ import annotations

import re

HANGUL = re.compile(r"[\uac00-\ud7a3\u3131-\u318e]")
# Qwen3-VL understands Korean, Z-Image's Qwen3 encoder mostly does too; CLIP (Illustrious) and umT5 (Wan) do not.
NEEDS_ENGLISH = {"illustrious", "wan22"}

TAGS = """Rewrite the image request below as a prompt for an Illustrious (SDXL anime/semi-realistic) model:
comma-separated English Danbooru-style tags, most important first (subject count like 1girl, then appearance,
clothing, pose, expression, place, lighting, camera). Keep every detail the user asked for and add nothing that
contradicts it. Keep any text inside quotes exactly as written. Reply with the tags only.

Request: """

SENTENCE = """Rewrite the image request below as a detailed English image-generation prompt: one or two natural
sentences covering subject, clothing, pose, place, lighting and camera. Keep every detail the user asked for and add
nothing that contradicts it. Keep any text inside quotes exactly as written. Reply with the prompt only.

Request: """


def has_korean(text: str) -> bool:
    return bool(HANGUL.search(text or ""))


def should_translate(text: str, family: str, mode: str) -> bool:
    if mode == "never" or not has_korean(text):
        return False
    return mode == "always" or family in NEEDS_ENGLISH


def instruction(text: str, family: str) -> str:
    return (TAGS if family == "illustrious" else SENTENCE) + text.strip()


def clean(reply: str, original: str) -> str:
    t = (reply or "").strip()
    t = re.sub(r"^```\w*|```$", "", t, flags=re.M).strip()
    t = re.sub(r"^(prompt|tags|answer)\s*:\s*", "", t, flags=re.I).strip().strip('"').strip()
    t = t.split("\n\n")[0].strip()
    return t if len(t) >= 3 else original


ENHANCE_TAGS = """Expand the image request below into a prompt for an Illustrious (SDXL semi-realistic) model:
comma-separated English Danbooru-style tags, 20-35 tags, most important first: subject count (1girl ...), look,
hair, clothing, pose, expression, place, lighting, camera angle and framing. Keep everything the user asked for and
change nothing they specified; only add details that fit. Keep any text inside quotes exactly. Reply with the tags only.

Request: """

ENHANCE_SENTENCE = """Expand the image request below into a detailed English prompt for a photorealistic image model:
three or four natural sentences covering the subject (age range, look), clothing, pose, expression, setting,
lighting, camera and lens, and mood. Keep everything the user asked for and change nothing they specified; only add
details that fit. Keep any text inside quotes exactly. Reply with the prompt only.

Request: """


def enhance_instruction(text: str, family: str) -> str:
    return (ENHANCE_TAGS if family == "illustrious" else ENHANCE_SENTENCE) + text.strip()


EDIT = """Rewrite the image edit request below as one short, clear English instruction for an image editing model
(imperative, e.g. "Change the jacket to red leather."). Keep every detail of the request and add nothing.
Reply with the instruction only.

Request: """


def edit_instruction(text: str) -> str:
    return EDIT + text.strip()


EDIT_ENHANCE = """Expand the image edit request below into a detailed English instruction for an image editing model.
Say exactly what changes{where}, what it should look like afterwards, and that everything else stays the same
(same person, face, body, pose, lighting and background). Keep every detail of the request and add nothing that
contradicts it. Reply with the instruction only, in at most three sentences.

Request: """


def edit_enhance_instruction(text: str, names: list[str]) -> str:
    where = f" in the selected area ({', '.join(names)})" if names else ""
    return EDIT_ENHANCE.format(where=where) + text.strip()
