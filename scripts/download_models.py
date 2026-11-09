"""Download the model files listed in config/models.yaml into ComfyUI/models.

    python scripts/download_models.py            # everything that is missing (optional models skipped)
    python scripts/download_models.py --all      # include optional models
    python scripts/download_models.py --check    # only report what is missing
    python scripts/download_models.py --models cyberrealistic-zit realism-illustrious

Interrupted downloads resume where they stopped. Civitai needs a free API key
(civitai.com → Settings → API Keys); you are asked once and it is saved in data/secrets.json.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from genstudio.config import get_secret, load_settings, write_secret  # noqa: E402
from genstudio.registry import FileSpec, Registry  # noqa: E402

UA = "gen-studio-downloader/0.1"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def civitai_key(interactive: bool) -> str | None:
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


def download(f: FileSpec, dest: Path, client: httpx.Client, key: str | None, verify: bool = True) -> None:
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
                        print(f"\r    {pct} {human(done)} / {human(total)}  {human(speed)}/s  {eta}     ", end="", flush=True)
            print()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            print(f"\n    연결이 끊겼어요 ({type(e).__name__}). {attempt}/5번째 다시 시도…")
            time.sleep(3 * attempt)
    else:
        raise RuntimeError("다섯 번 시도했지만 받지 못했어요. 잠시 후 다시 실행하면 이어서 받습니다.")

    if f.size and part.stat().st_size != f.size:
        raise RuntimeError(f"크기가 맞지 않아요 ({part.stat().st_size} != {f.size}). 다시 실행하면 이어서 받습니다.")
    if verify and f.sha256:
        print("    파일 검사 중…", end="", flush=True)
        if sha256_of(part).lower() != f.sha256.lower():
            part.unlink()
            raise RuntimeError("파일이 손상됐어요. 다시 실행하면 처음부터 받습니다.")
        print(" 이상 없음")
    os.replace(part, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="받을 모델 id (기본: 선택 모델을 뺀 전부)")
    ap.add_argument("--all", action="store_true", help="선택(optional) 모델까지 전부 받기")
    ap.add_argument("--check", action="store_true", help="받지 않고 상태만 확인")
    ap.add_argument("--no-verify", action="store_true", help="다운로드 후 sha256 검사 생략")
    ap.add_argument("--yes", action="store_true", help="묻지 않고 진행")
    args = ap.parse_args()

    settings = load_settings()
    if not settings.models_dir:
        print("config/settings.yaml 의 comfyui.path 를 먼저 설정하세요.")
        return 1
    models_dir = settings.models_dir
    reg = Registry.load()
    if args.models:
        chosen = [reg.get(m) for m in args.models]
    else:
        chosen = [m for m in reg.models if args.all or not m.optional or m.installed(models_dir)]
        skipped = [m for m in reg.models if m not in chosen]
        if skipped:
            print("  선택 모델 (기본 설치에서 제외): " + ", ".join(f"{m.name} [--models {m.id}]" for m in skipped))

    todo: list[tuple[str, FileSpec]] = []
    seen: set[str] = set()
    for m in chosen:
        for f in m.all_files():
            key = f"{f.dir}/{f.name}"
            if key in seen:
                continue
            seen.add(key)
            ok = f.installed(models_dir)
            print(f"  {'✓' if ok else '·'} {f.dir}/{f.name}  ({human(f.size or 0)})  [{m.name}]")
            if not ok:
                todo.append((m.name, f))

    if not todo:
        print("\n모든 모델이 준비돼 있어요.")
        return 0
    total = sum(f.size or 0 for _, f in todo)
    print(f"\n받을 파일 {len(todo)}개, 합계 {human(total)}  →  {models_dir}")
    if args.check:
        return 0

    need_civitai = any(f.civitai_version for _, f in todo)
    key = civitai_key(interactive=not args.yes) if need_civitai else None
    if need_civitai and not key:
        print("Civitai API 키가 없어서 Civitai 모델은 건너뜁니다.")
        todo = [(n, f) for n, f in todo if not f.civitai_version]

    failed = []
    with httpx.Client(headers={"User-Agent": UA}) as client:
        for i, (name, f) in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {f.name}  ({name}, {f.source})")
            try:
                download(f, f.path(models_dir), client, key, verify=not args.no_verify)
            except KeyboardInterrupt:
                print("\n중단했어요. 다시 실행하면 이어서 받습니다.")
                return 130
            except Exception as e:
                print(f"    실패: {e}")
                failed.append(f.name)
    if failed:
        print(f"\n받지 못한 파일: {', '.join(failed)}\n다시 실행하면 이어서 받습니다.")
        return 2
    print("\n모든 모델을 받았어요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
