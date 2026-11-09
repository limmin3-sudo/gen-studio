"""Saved characters: a name plus reference images (and a short description) reused across generations.

Stored in outputs/characters.json next to the images they point at.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

MAX_REFS = 6


def slugify(name: str) -> str:
    s = re.sub(r"[^\w가-힣-]+", "-", name.strip().lower()).strip("-")
    return s or f"character-{int(time.time())}"


class Characters:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        try:
            for c in json.loads(path.read_text(encoding="utf-8")):
                self._items[c["id"]] = c
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(list(self._items.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    def list(self) -> list[dict]:
        return sorted(self._items.values(), key=lambda c: c["created_at"])

    def get(self, key: str) -> dict | None:
        if key in self._items:
            return self._items[key]
        low = key.strip().lower()
        return next((c for c in self._items.values() if c["name"].lower() == low), None)

    def save(self, name: str, ref_ids: list[str], description: str = "", character_id: str | None = None) -> dict:
        if not name.strip():
            raise ValueError("캐릭터 이름을 입력하세요")
        if not 1 <= len(ref_ids) <= MAX_REFS:
            raise ValueError(f"참고 이미지는 1~{MAX_REFS}장이에요")
        with self._lock:
            cid = character_id or slugify(name)
            if character_id is None and cid in self._items:
                raise ValueError(f"이미 같은 이름의 캐릭터가 있어요: {name}")
            prev = self._items.get(cid, {})
            self._items[cid] = {
                "id": cid, "name": name.strip(), "description": description.strip(), "ref_ids": list(ref_ids),
                "created_at": prev.get("created_at", time.time()), "updated_at": time.time(),
            }
            self._save()
            return self._items[cid]

    def delete(self, key: str) -> bool:
        c = self.get(key)
        if not c:
            return False
        with self._lock:
            self._items.pop(c["id"], None)
            self._save()
        return True
