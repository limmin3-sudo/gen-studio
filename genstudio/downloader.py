"""Model file downloads (shared by scripts/download_models.py and the app's settings screen)."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import httpx

from .config import get_secret, write_secret
from .registry import FileSpec, Registry

UA = "gen-studio-downloader/0.1"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def civitai_key(interactive: bool) -> str | None:  # noqa: C901
    key = get_secret("civitai_api_key", "CIVITAI_API_KEY")
    if key or not interactive:
        return key
    print("\nCivitai 모델을 받으려면 API 키가 필요해요 (무료).")
    print("  civitai.com 로그인 → 오른쪽 위 프로필 → Settings → API Keys → Add API key")
    key = input("  API 키를 붙여넣고 Enter: ").strip()
    if key:
        write_secret("civitai_api_key", key)
        print("  저장했어요 (data/secrets.json, 이 PC에만 저장됩니다).")
    return key or None


def resolve_url(f: FileSpec, client: httpx.Client, key: str | None) -> tuple[str, dict]:
    if f.url:
        headers = {}
        tok = get_secret("hf_token", "HF_TOKEN")
        if tok and "huggingface.co" in f.url:
            headers["Authorization"] = f"Bearer {tok}"
        return f.url, headers
    # Civitai: look the file up in its version so we fetch exactly this file (versions can hold several)
    r = client.get(f"https://civitai.com/api/v1/model-versions/{f.civitai_version}")
    r.raise_for_status()
    files = r.json().get("files", [])
    match = next((x for x in files if x.get("name") == f.name), None)
    url = (match or {}).get("downloadUrl") or f"https://civitai.com/api/download/models/{f.civitai_version}"
    if not key:
        raise RuntimeError("Civitai API 키가 없어요")
    return url, {"Authorization": f"Bearer {key}"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(f: FileSpec, dest: Path, client: httpx.Client, key: str | None, verify: bool = True,
             on_progress=None) -> None:
    """Resumable download into dest (.part while running), then size and sha256 checks.
    on_progress(done_bytes, total_bytes, speed_bytes_per_s) is called while it runs; without it, prints a line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url, headers = resolve_url(f, client, key)
    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        h = dict(headers)
        if have:
            h["Range"] = f"bytes={have}-"
        try:
            with client.stream("GET", url, headers=h, follow_redirects=True, timeout=httpx.Timeout(60, read=120)) as r:
                if r.status_code in (401, 403):
                    raise RuntimeError("권한이 없어요. Civitai API 키를 확인하거나, 사이트에서 성인 콘텐츠 보기를 켜세요."
                                       if f.civitai_version else f"다운로드가 거부됐어요 ({r.status_code})")
                if r.status_code == 416:            # already complete
                    break
                r.raise_for_status()
                if have and r.status_code != 206:   # server ignored Range; start over
                    have = 0
                total = int(r.headers.get("content-length", 0)) + have
                mode = "ab" if have else "wb"
                t0, done = time.monotonic(), have
                with open(part, mode) as out:
                    for chunk in r.iter_bytes(4 * 1024 * 1024):
                        out.write(chunk)
                        done += len(chunk)
                        dt = max(0.1, time.monotonic() - t0)
                        speed = (done - have) / dt
                        pct = f"{done / total * 100:5.1f}%" if total else ""
                        eta = f"남은 시간 {int((total - done) / speed // 60)}분" if total and speed > 0 else ""
                        if on_progress:
                            on_progress(done, total, speed)
                        else:
                            print(f"\r    {pct} {human(done)} / {human(total)}  {human(speed)}/s  {eta}     ", end="", flush=True)
            if not on_progress:
                print()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if not on_progress:
                print(f"\n    연결이 끊겼어요 ({type(e).__name__}). {attempt}/5번째 다시 시도…")
            time.sleep(3 * attempt)
    else:
        raise RuntimeError("다섯 번 시도했지만 받지 못했어요. 잠시 후 다시 실행하면 이어서 받습니다.")

    if f.size and not f.size_ok(part.stat().st_size):
        raise RuntimeError(f"크기가 맞지 않아요 ({part.stat().st_size} != {f.size}). 다시 실행하면 이어서 받습니다.")
    if verify and f.sha256:
        if not on_progress:
            print("    파일 검사 중…", end="", flush=True)
        if sha256_of(part).lower() != f.sha256.lower():
            part.unlink()
            raise RuntimeError("파일이 손상됐어요. 다시 실행하면 처음부터 받습니다.")
        if not on_progress:
            print(" 이상 없음")
    os.replace(part, dest)


def plan(reg: Registry, models_dir: Path, model_ids: list[str] | None = None, include_optional: bool = False):
    """[(label, FileSpec, installed)] for the chosen models plus every tool, each file once."""
    if model_ids:
        chosen = [reg.get(m) for m in model_ids]
    else:
        chosen = [m for m in reg.models if include_optional or not m.optional or m.installed(models_dir)]
    groups = [(m.name, m.all_files()) for m in chosen]
    if not model_ids:
        groups += [(f"도구: {name}", files) for name, files in reg.tools.items()]
    out, seen = [], set()
    for label, files in groups:
        for f in files:
            k = f"{f.dir}/{f.name}"
            if k in seen:
                continue
            seen.add(k)
            out.append((label, f, bool(f.installed(models_dir))))
    return out
