"""Register Gen Studio as an MCP server in Codex (~/.codex/config.toml).

On the home PC (Gen Studio runs here):
    connect-codex.bat
On the laptop (reaching the home PC over Tailscale):
    python scripts/connect_codex.py --remote https://home-pc.tailXXXX.ts.net --token <token>
    (or edit ~/.codex/config.toml by hand, see docs/codex.md)

The Codex app and the Codex CLI read the same config file.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SECTION = "mcp_servers.gen_studio"


def build_section(url: str, remote: bool) -> str:
    lines = [f"[{SECTION}]", f'url = "{url}"', "enabled = true"]
    if remote:
        lines.append('bearer_token_env_var = "GEN_STUDIO_TOKEN"')
    lines += ["startup_timeout_sec = 30", "tool_timeout_sec = 900"]
    return "\n".join(lines) + "\n"


def upsert(config: Path, section_text: str) -> str:
    text = config.read_text(encoding="utf-8") if config.exists() else ""
    # the section header plus every following line up to the next "[table]" header
    pattern = re.compile(rf"^\[{re.escape(SECTION)}\][ \t]*\n(?:(?!\[).*(?:\n|$))*", re.M)
    if pattern.search(text):
        new = pattern.sub(section_text + "\n", text, count=1)
        action = "바꿨어요"
    else:
        new = (text.rstrip() + "\n\n" if text.strip() else "") + section_text
        action = "추가했어요"
    if config.exists():
        shutil.copy2(config, config.with_suffix(f".toml.bak-{time.strftime('%Y%m%d%H%M%S')}"))
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(new, encoding="utf-8")
    return action


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", help="집 PC의 Tailscale 주소 (예: https://home-pc.tail1234.ts.net)")
    ap.add_argument("--token", help="집 PC의 접속 토큰 (data/secrets.json 의 access_token)")
    ap.add_argument("--config", default=str(Path.home() / ".codex" / "config.toml"))
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    config = Path(args.config)
    if args.remote:
        url = args.remote.rstrip("/") + "/mcp"
        action = upsert(config, build_section(url, remote=True))
        if args.token:
            if os.name == "nt":
                subprocess.run(["setx", "GEN_STUDIO_TOKEN", args.token], capture_output=True)
                print("  환경변수 GEN_STUDIO_TOKEN 을 저장했어요. Codex 앱을 완전히 껐다 켜야 적용됩니다.")
            else:
                print(f"  셸 설정에 다음 줄을 추가하세요:  export GEN_STUDIO_TOKEN={args.token}")
        else:
            print("  --token 을 주지 않았어요. 환경변수 GEN_STUDIO_TOKEN 에 토큰을 넣어 주세요.")
    else:
        from genstudio.config import load_settings
        port = load_settings().port if args.port == 7860 else args.port
        url = f"http://127.0.0.1:{port}/mcp"
        action = upsert(config, build_section(url, remote=False))

    print(f"\n  Codex 설정에 Gen Studio를 {action}: {config}")
    print(f"  MCP 주소: {url}")
    print("\n  다음 단계:")
    print("   1) Gen Studio를 켜 둡니다 (start.bat).")
    print("   2) Codex 앱을 다시 시작합니다.")
    print('   3) Codex에서 "gen_studio로 바닷가 노을 사진 만들어줘" 처럼 요청해 보세요.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
