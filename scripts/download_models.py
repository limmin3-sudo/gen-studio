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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from genstudio.config import load_settings  # noqa: E402
from genstudio.downloader import civitai_key, download, human, plan  # noqa: E402,F401
from genstudio.registry import FileSpec, Registry  # noqa: E402

import httpx  # noqa: E402

UA = "gen-studio-downloader/0.1"


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
    if not args.models:
        skipped = [m for m in reg.models if m.optional and not m.installed(models_dir) and not args.all]
        if skipped:
            print("  선택 모델 (기본 설치에서 제외): " + ", ".join(f"{m.name} [--models {m.id}]" for m in skipped))

    todo: list[tuple[str, FileSpec]] = []
    for label, f, ok in plan(reg, models_dir, args.models, include_optional=args.all):
        print(f"  {'✓' if ok else '·'} {f.dir}/{f.name}  ({human(f.size or 0)})  [{label}]")
        if not ok:
            todo.append((label, f))

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
