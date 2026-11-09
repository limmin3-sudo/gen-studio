# Gen Studio

집 PC 그래픽카드로 실사·반실사 이미지를 만들고 편집하는 로컬 툴입니다.
여러 모델을 넣어 두면 요청 내용에 맞는 모델을 알아서 고르고, Codex·ChatGPT·Claude에서도 MCP로 불러 쓸 수 있습니다.
외부 유료 API는 쓰지 않습니다.

- **생성**: 한국어로 적어도 됩니다. "실사", "반실사", "포스터", "세로" 같은 말을 보고 모델과 크기를 고릅니다.
- **편집**: 바꿀 부분을 브러시로 칠하고 무엇으로 바꿀지 입력합니다. 칠한 부분만 다시 그리고 나머지는 그대로 둡니다.
  - 칠한 부분을 삭제하면 그 자리를 어떻게 채울지 묻습니다(모델이 알아서 / 비워두기 / 직접 입력). 묻지 않고 바로 채우도록 설정할 수도 있습니다.
  - 편집할 때마다 왼쪽에 버전이 쌓입니다.
- **Codex·ChatGPT 연결**: `/mcp` 주소로 생성, 편집, 진행 상태 조회, 결과 이미지 반환을 할 수 있습니다.

권장 사양: NVIDIA RTX(VRAM 16GB 권장), RAM 32GB, 여유 공간 120GB 이상(SSD), Windows 10/11.

## 설치 (집 PC)

1. 이 저장소를 받습니다.
   - Git이 있으면: `git clone https://github.com/limmin3-sudo/gen-studio`
   - 없으면 GitHub 페이지에서 **Code → Download ZIP** 후 압축을 풉니다. 경로에 한글이나 공백이 없는 곳이 좋습니다(예: `C:\gen-studio`).
2. Civitai API 키를 준비합니다(무료). civitai.com 로그인 → 프로필 → **Settings → API Keys → Add API key**.
   Content & Browsing에서 성인 콘텐츠 보기를 켜 두세요. 일부 모델은 이 설정이 꺼져 있으면 받을 수 없습니다.
3. **`install.bat`** 을 더블클릭합니다. 다음을 자동으로 합니다.
   - 파이썬 환경 준비 (uv 사용, 파이썬을 따로 설치할 필요 없음)
   - ComfyUI 포터블 설치 (약 2GB)
   - 모델 다운로드 (약 90GB). 처음 한 번 Civitai API 키를 물어봅니다. 키는 이 PC의 `data/secrets.json` 에만 저장됩니다.

   중간에 끊겨도 다시 실행하면 이어서 진행합니다.
4. **`start.bat`** 을 더블클릭하면 ComfyUI가 켜지고 브라우저에서 Gen Studio가 열립니다 (`http://127.0.0.1:7860`).

## 기본 모델

| 용도 | 모델 | 기반 |
|---|---|---|
| 실사 (기본) | CyberRealistic Z-Image Turbo v10 | Z-Image Turbo |
| 실사 (보조) | Real Dream ZIT v7 | Z-Image Turbo |
| 반실사 (기본) | Realism Illustrious v5.5 (Stable Yogi) | Illustrious (SDXL) |
| 반실사 (보조) | Semi-Real (Illustrious) MM v5 | Illustrious (SDXL) |
| 글자·디자인 | Qwen-Image 2.1 Turbo | Qwen-Image 2.1 (7B) |
| 극실사 | Wan 2.2 14B (LightX2V 4단계) | Wan 2.2 (영상 모델의 1프레임) |

"포스터", "간판", "로고"처럼 글자가 들어가는 요청이나 "따옴표로 감싼 문구"가 있으면 Qwen-Image 2.1을 고릅니다.
Qwen은 학습 단계에서 노출 표현이 약하므로, 인물 위주 이미지는 실사·반실사 모델이 맡습니다.
"극실사", "초실사", "hyperrealistic" 같은 말이 있으면 Wan 2.2를 고릅니다. Wan 2.2 파일은 4단계 영상 기능에서도 그대로 씁니다.

모델은 `config/models.yaml` 에 적혀 있습니다. 항목을 추가하고 `update.bat` 을 실행하면 새 모델을 받습니다.
편집할 때는 원본을 만든 모델로 다시 그려서 그림체를 유지합니다. 외부에서 불러온 이미지는 기본 실사 모델로 편집합니다.

## Codex·ChatGPT에서 쓰기

- **Codex (집 PC)**: `connect-codex.bat` 을 실행하고 Codex 앱을 다시 시작하면 됩니다. 자세한 내용은 [docs/codex.md](docs/codex.md)를 보세요.
- **노트북에서 집 PC 쓰기**: Tailscale로 연결합니다. [docs/remote.md](docs/remote.md)를 보세요.
- **ChatGPT**: ChatGPT는 인터넷에 공개된 HTTPS 주소와 OAuth 로그인이 필요합니다. 2단계에서 추가합니다.

## 파일 구성

```
install.bat / start.bat / start-remote.bat / update.bat / connect-codex.bat
config/settings.yaml   포트, ComfyUI 위치 등 설정
config/models.yaml     모델 목록과 권장 설정
genstudio/             서버 (FastAPI + MCP)
web/                   편집기 화면
scripts/               설치, 모델 다운로드, Codex 연결
outputs/               만든 이미지 (날짜별 폴더, 이미지마다 설정 기록 .json 포함)
data/secrets.json      API 키와 접속 토큰 (이 PC에만 저장, GitHub에 올라가지 않음)
prototype/editor.html  편집기 디자인 미리보기
```

## 문제가 생기면

- **상태 점검**: Gen Studio가 켜진 상태에서 `http://127.0.0.1:7860/api/doctor` 를 열면 ComfyUI 연결, 필요한 노드, 모델 파일 상태를 보여줍니다.
- **"ComfyUI 꺼짐"**: `start.bat` 이 ComfyUI를 새 창으로 켭니다. 그 창에 오류가 보이면 NVIDIA 드라이버를 최신으로 업데이트하세요.
- **메모리 부족(out of memory)**: 다른 프로그램(게임, 브라우저 탭)을 닫고 다시 시도하세요. 실사와 반실사 모델을 번갈아 쓰면 바꿀 때마다 메모리를 비웁니다.
- **모델이 안 받아짐**: Civitai API 키와 성인 콘텐츠 설정을 확인하고 `update.bat` 을 다시 실행하세요.

## 개발

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest
```

테스트는 가짜 ComfyUI로 생성, 편집, MCP, 원격 접속 보안을 확인합니다. 실제 그래픽카드는 필요 없습니다.

## 로드맵

1. **1단계 (현재)**: 생성과 모델 자동 선택, 브러시 편집, 삭제 후 채우기, 버전 기록, Codex MCP, 설치 파일
2. **2단계**: 자동 세그먼트(사람·머리카락·옷 등 클릭 선택), 물체 제거(LaMa), 참고 이미지로 생성, ChatGPT 연결(OAuth)
3. **3단계**: 문장 편집(Qwen-Image-Edit), 참고 이미지로 수정, 종횡비 늘리기, 배경 제거
4. **4단계**: 로컬 LLM으로 한국어 이해와 자동 판단, LoRA 관리, 이미지→영상
