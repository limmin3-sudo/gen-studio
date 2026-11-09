# Codex에서 Gen Studio 쓰기

Gen Studio는 `http://127.0.0.1:7860/mcp` 에서 MCP 서버를 엽니다. Codex 앱과 Codex CLI는 같은 설정 파일
`%USERPROFILE%\.codex\config.toml` 을 읽습니다.

## 집 PC (Gen Studio와 Codex가 같은 PC)

1. `start.bat` 으로 Gen Studio를 켭니다.
2. `connect-codex.bat` 을 실행합니다. 설정 파일에 아래 내용을 추가합니다(기존 파일은 `.bak-날짜` 로 백업).

   ```toml
   [mcp_servers.gen_studio]
   url = "http://127.0.0.1:7860/mcp"
   enabled = true
   startup_timeout_sec = 30
   tool_timeout_sec = 900
   ```

3. Codex 앱을 완전히 종료했다가 다시 켭니다.
4. Codex에 이렇게 요청해 봅니다.
   - "gen_studio로 비 오는 밤 골목에 서 있는 여자 실사 사진 세로로 만들어줘"
   - "방금 이미지 배경을 해변으로 바꿔줘"

- `startup_timeout_sec`(기본 10초): 서버 연결·초기화 제한.
- `tool_timeout_sec`(기본 60초): 도구 호출 한 번의 제한. 15분으로 늘려 둡니다.
- 도구는 기본으로 최대 45초만 기다리고, 그 안에 안 끝나면 `job_id` 를 돌려줍니다.
  Codex는 `get_job_status` → `get_image` 순서로 이어서 확인합니다. 그래서 기본 60초 설정에서도 끊기지 않습니다.
- 플러그인으로 묶을 필요는 없습니다. `config.toml` 에 등록하는 것으로 충분합니다.
- 설정을 바꾼 뒤에는 Codex 앱에서 MCP 서버를 다시 시작하거나 새 작업을 열어 연결을 확인하세요.

## 노트북 (집 PC의 Gen Studio를 원격으로)

먼저 [remote.md](remote.md)대로 Tailscale을 연결하고, 집 PC에서 `start-remote.bat` 으로 켭니다.
집 PC 창에 표시된 주소와 토큰을 사용합니다.

```toml
[mcp_servers.gen_studio]
url = "https://home-pc.tailXXXX.ts.net/mcp"
bearer_token_env_var = "GEN_STUDIO_TOKEN"
enabled = true
startup_timeout_sec = 30
tool_timeout_sec = 900
```

토큰은 설정 파일에 직접 적지 않고 환경변수로 넣습니다. 노트북의 PowerShell에서:

```powershell
setx GEN_STUDIO_TOKEN "집 PC data\secrets.json 의 access_token 값"
```

노트북에 이 저장소가 있으면 `python scripts/connect_codex.py --remote https://home-pc.tailXXXX.ts.net --token 토큰` 으로 위 작업을 한 번에 할 수 있습니다.

## 대화창 안 편집 패널

"이 이미지 세그먼트 편집기로 열어줘"라고 하면 `open_editor` 가 호출됩니다.

- **패널을 지원하는 프로그램**(ChatGPT, Claude 등 MCP Apps 지원): 대화창 안에 이미지와 세그먼트 목록이 뜹니다.
  목록이나 그림을 눌러 고르고, 🗑으로 삭제하고, 입력칸에 바꿀 내용을 쓰면 됩니다. 결과는 버전으로 쌓입니다.
- **지원하지 않는 프로그램**: 번호가 칠해진 이미지와 목록, 그리고 브라우저 편집기 링크(`editor_url`)가 나옵니다.
  "3번 빨간 재킷으로"처럼 대화로 하거나, 링크를 열어 같은 화면에서 작업하면 됩니다.

Codex 앱이 패널을 띄우는지는 앱 버전에 따라 다릅니다. 한 번 시도해 보면 바로 알 수 있습니다.

## 쓸 수 있는 도구

| 도구 | 하는 일 |
|---|---|
| `generate_image` | 프롬프트로 생성. 스타일(실사/반실사), 비율, 모델, 개수, 시드 지정 가능 |
| `edit_image` | 이미지 편집. `image_id`, `image_base64`, `image_path` 중 하나로 이미지를 지정. `mask_base64`(흰색 = 바꿀 곳)가 있으면 그 부분만, 없으면 전체를 `denoise` 세기로 다시 그림. `fill`: `prompt` / `auto`(모델이 알아서) / `clear`(비우기) |
| `get_job_status` | 진행 상태 조회 (오래 걸리면 `job_id` 로 확인) |
| `get_image` | 결과 이미지(미리보기)와 정보 |
| `list_images` | 최근 이미지 목록 |
| `list_models` | 모델 목록, 모델별로 잘하는 것, 설치 여부 |
| `recommend_model` | 작업(생성, 부분 편집, 전체 다시 그리기, 문장 편집, 캔버스 늘리기)별 추천 모델과 이유 |
| `segment_image` | 부분(사람, 머리카락, 옷 등) 찾기. 번호를 칠한 이미지와 목록을 돌려줌. 점(`points`)으로 클릭 선택도 가능 |
| `open_editor` | 대화창 안 세그먼트 편집 패널 (지원하지 않으면 번호 이미지와 링크) |
| `create_character` / `list_characters` | 캐릭터 등록과 목록. 이름은 대충 불러도 찾음 |
| `export_image` | 원하는 폴더에 webp/png/jpg로 저장 (크기 지정, 덮어쓰기 확인) |

결과 이미지는 Gen Studio 편집기에서도 바로 열립니다. 도구 결과의 `editor_url` 을 브라우저로 여세요.

## 연결 테스트

Gen Studio가 켜진 상태에서 PowerShell로 확인할 수 있습니다.

```powershell
curl.exe -s -X POST http://127.0.0.1:7860/mcp -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}"
```

도구 6개가 담긴 JSON이 나오면 정상입니다.
