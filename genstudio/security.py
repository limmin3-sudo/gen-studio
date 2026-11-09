"""Requests from this PC pass freely; anything else needs the access token.

Remote access (laptop over Tailscale, Codex on another machine) sends the token as
  Authorization: Bearer <token>    (API / MCP clients)
or opens the editor once with ?token=<token>, which sets a cookie for the browser.

Tailscale Serve forwards remote requests from 127.0.0.1, so "local" also requires a
localhost Host header and no proxy headers. Cross-site writes from other websites are refused.
"""
from __future__ import annotations

import hmac
import ipaddress
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

COOKIE = "gs_token"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}
PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "forwarded", "tailscale-user-login")


def _loopback(ip: str | None) -> bool:
    if not ip:
        return False
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return ip == "testclient"


def is_local_request(request: Request) -> bool:
    client = request.client.host if request.client else None
    host = (urlsplit("//" + (request.headers.get("host") or "")).hostname or "").lower()
    return (_loopback(client) and host in LOCAL_HOSTS
            and not any(h in request.headers for h in PROXY_HEADERS))


def _cross_site(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin or request.method in ("GET", "HEAD", "OPTIONS"):
        return False
    return urlsplit(origin).netloc.lower() != (request.headers.get("host") or "").lower()


class TokenGate(BaseHTTPMiddleware):
    def __init__(self, app, token_getter, remote_enabled_getter):
        super().__init__(app)
        self._token = token_getter
        self._remote = remote_enabled_getter

    async def dispatch(self, request: Request, call_next):
        if _cross_site(request):
            return JSONResponse({"error": "다른 사이트에서 보낸 요청은 받지 않아요."}, status_code=403)
        if is_local_request(request):
            return await call_next(request)
        if not self._remote():
            return JSONResponse({"error": "이 PC에서만 접속할 수 있어요. start-remote.bat으로 실행하면 다른 기기에서도 열 수 있어요."},
                                status_code=403)
        token = self._token()
        sent = request.headers.get("authorization", "")
        sent = sent[7:] if sent.lower().startswith("bearer ") else ""
        sent = sent or request.cookies.get(COOKIE, "") or request.query_params.get("token", "")
        if not sent or not hmac.compare_digest(sent, token):
            return JSONResponse({"error": "접속 토큰이 필요해요. 집 PC의 Gen Studio 창에 표시된 주소로 접속하세요."},
                                status_code=401, headers={"WWW-Authenticate": "Bearer"})
        if request.query_params.get("token") and request.method == "GET" and not request.url.path.startswith(("/api", "/mcp")):
            # move the token from the URL into a cookie so it does not stay in the address bar
            # relative Location: behind Tailscale Serve the outside scheme is https even though we see http
            query = str(request.url.remove_query_params("token").query)
            resp = RedirectResponse(request.url.path + (f"?{query}" if query else ""), status_code=303)
            https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
            resp.set_cookie(COOKIE, token, httponly=True, secure=https, samesite="lax", max_age=60 * 60 * 24 * 90)
            return resp
        return await call_next(request)
