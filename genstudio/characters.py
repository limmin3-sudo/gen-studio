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


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.strip().lower())


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
        """Exact id, name or alias first; then a loose match ("디렉터" finds "플라이트 디렉터")."""
        if not key:
            return None
        if key in self._items:
            return self._items[key]
        low = _norm(key)
        def names(c):
            return [_norm(c["name"]), _norm(c["id"]), *(_norm(a) for a in c.get("aliases", []))]
        exact = [c for c in self._items.values() if low in names(c)]
        if exact:
            return exact[0]
        loose = [c for c in self._items.values() if any(low in n or (len(n) >= 2 and n in low) for n in names(c))]
        return loose[0] if len(loose) == 1 else None

    def save(self, name: str, ref_ids: list[str], description: str = "", character_id: str | None = None,
             aliases: list[str] | None = None) -> dict:
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
                "aliases": [a.strip() for a in (aliases if aliases is not None else prev.get("aliases", [])) if a.strip()],
                "created_at": prev.get("created_at", time.time()), "updated_at": time.time(),
            }
            self._save()
            return self._items[cid]

    def clear(self) -> int:
        with self._lock:
            n = len(self._items)
            self._items.clear()
            self._save()
        return n

    def delete(self, key: str) -> bool:
        c = self.get(key)
        if not c:
            return False
        with self._lock:
            self._items.pop(c["id"], None)
            self._save()
        return True
