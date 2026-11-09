# 노트북에서 집 PC의 Gen Studio 쓰기 (Tailscale)

Tailscale은 내 기기들끼리만 연결되는 개인 네트워크를 만들어 주는 무료 프로그램입니다.
집 PC를 인터넷에 공개하지 않고, 내 Tailscale 계정에 로그인한 기기에서만 접속할 수 있습니다.

## 한 번만 하는 설정

1. 집 PC와 노트북 둘 다 [tailscale.com/download](https://tailscale.com/download) 에서 Tailscale을 설치합니다.
2. 두 기기에서 **같은 계정**으로 로그인합니다.
3. Tailscale 관리 페이지(login.tailscale.com/admin)의 **DNS** 탭에서 **MagicDNS** 와 **HTTPS Certificates** 를 켭니다.
   그러면 집 PC에 `https://home-pc.tailXXXX.ts.net` 같은 고정 주소가 생깁니다.

## 쓸 때

1. 집 PC에서 **`start-remote.bat`** 으로 Gen Studio를 켭니다.
   - Tailscale Serve가 Gen Studio를 개인 네트워크에만 연결합니다.
   - 창에 노트북용 주소가 표시됩니다: `https://home-pc.tailXXXX.ts.net/?token=...`
2. 노트북 브라우저에서 그 주소를 엽니다. 한 번 열면 토큰이 쿠키로 저장돼서, 다음부터는 `https://home-pc.tailXXXX.ts.net` 만 열면 됩니다.
3. 노트북 Codex 연결은 [codex.md](codex.md)의 "노트북" 부분을 보세요.

## 보안

- Gen Studio는 항상 집 PC 내부 주소(127.0.0.1)에서만 열립니다. 바깥 기기는 Tailscale을 거쳐서만 들어올 수 있습니다.
- Tailscale을 거쳐 들어오는 요청은 접속 토큰이 있어야 합니다. 토큰은 집 PC의 `data/secrets.json` 에 있습니다.
- 토큰이 유출됐다면 `data/secrets.json` 에서 `access_token` 줄을 지우고 다시 켜세요. 새 토큰이 만들어집니다.
- 원격 접속을 끄려면 `start.bat` 으로 켜고, 집 PC에서 `tailscale serve reset` 을 실행하세요.
- 집 PC가 켜져 있고 Gen Studio가 실행 중이어야 밖에서 쓸 수 있습니다.

## ChatGPT 연결 (2단계 예정)

ChatGPT는 인터넷에 공개된 HTTPS 주소가 있어야 MCP 서버에 붙을 수 있고, 인증은 OAuth만 지원합니다.
2단계에서 Tailscale Funnel(공개 주소)과 OAuth 로그인을 붙여, `/mcp` 경로만 공개하고 본인만 쓸 수 있게 만들 예정입니다.
