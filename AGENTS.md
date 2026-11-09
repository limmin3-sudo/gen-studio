# Gen Studio — notes for coding agents (Codex, Claude)

Local image/video studio on the owner's Windows PC (RTX 4080 SUPER 16GB, 32GB RAM). ComfyUI (portable) is the
engine; this repo is the orchestrator + web UI + desktop app + MCP server. UI text and messages are Korean.

## Owner's rules
- No content filters, classifiers or hooks in the tool. The owner decides what to make; the tool only talks to the models.
- Personal, non-commercial use (Qwen-Image 2.1 is a research licence).
- Don't touch the STAM repo until the owner asks; Gen Studio is used there later through MCP.
- Plans and promises live in `docs/roadmap.md` (⭐ = must build: a drag-the-joints pose editor). Update it when you finish or add work.

## Layout
- `genstudio/` — FastAPI app (`app.py`, settings API in `app_routes.py`), job queue (`jobs.py`), ComfyUI graphs
  (`workflows.py`), routing (`router.py`), model list loader (`registry.py`, `user_models.py`), segments, LoRAs and
  training, detail pass, desktop app (`desktop.py`, `runtime.py`), MCP tools (`mcp_tools.py`, panel `panel.html`).
- `web/` — single-page UI (no build step): `index.html`, `app.js`, `style.css`.
- `config/models.yaml` — the real model lineup (RedCraft QW21 = realistic main, PornMaster Z-Image = helper for
  detail/upscale/pose/training, Illustrij = semi-real, DaSiWa Wan 2.2 I2V = video) plus tool files.
- `scripts/` — installer, downloader, updater (run by the app's update button), Codex registration, shortcuts.

## Working on it
- Tests: `python -m pytest -q` (use the venv). They run against a fake ComfyUI (`tests/fake_comfy.py`) and a frozen
  copy of the model list (`tests/fixtures/models.yaml`); `tests/test_lineup.py` checks the real `config/models.yaml`.
- Check UI changes in a browser at desktop and phone width (390px) before committing.
- `.bat`/`.ps1` files must keep CRLF line endings (see `.gitattributes`); `.ps1` files are UTF-8 with BOM.
- When a ComfyUI node or graph is new, copy the settings from ComfyUI's official template/blueprint, not from memory.
- The owner tests on the home PC and pastes the app's diagnostics (status bar → copy) when something fails.
