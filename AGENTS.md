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
  `optional_tools` (none now) are only downloaded from the settings screen. LTX-2.5 was tried and removed on
  2026-11-12 (quality on the 16GB PC was not worth it); don't add it back unless the owner asks.
- `scripts/` — installer, downloader, updater (run by the app's update button), Codex registration, shortcuts.

## Working on it
- Tests: `python -m pytest -q` (use the venv). They run against a fake ComfyUI (`tests/fake_comfy.py`) and a frozen
  copy of the model list (`tests/fixtures/models.yaml`); `tests/test_lineup.py` checks the real `config/models.yaml`.
- Check UI changes in a browser at desktop and phone width (390px) before committing.
- `.bat`/`.ps1` files must keep CRLF line endings (see `.gitattributes`); `.ps1` files are UTF-8 with BOM.
- When a ComfyUI node or graph is new, copy the settings from ComfyUI's official template/blueprint, not from memory.
- The owner tests on the home PC and pastes the app's diagnostics (status bar → copy) when something fails.

## Several AI tools work on this repo (Claude, Codex, Grok …) — read before every push
- **Pull first, then test, then push.** `git fetch origin main` and merge (or rebase) before you start and again
  before you push. Never force-push `main`.
- **Run the tests** (`python -m pytest -q`) after merging other tools' work and before pushing. If they fail, fix or
  ask; don't push red.
- **Look at the diff before committing** (`git diff --stat` and the changed lines). A file that shrinks by hundreds
  of lines, or turns into a single line, is almost always a mistake.
- **Never write placeholders into real files** ("PLACEHOLDER", "...", "rest of file unchanged"). Edit the lines you
  mean to change; if you can't see the whole file, read it first.
- **Front end and back end together.** A handler for a new button needs the button in `web/index.html`
  (`$('#id').onclick` on a missing element throws and stops the whole editor script) and a matching API route
  that calls a method that actually exists.
- **Don't drop existing routes** in `genstudio/app.py`: the MCP route (`app.router.routes.extend(mcp_app.routes)`),
  the `ValueError` handler and the static mount must stay, in that order.
- Prefer one tool pushing at a time. Say in the commit message what you changed and why.
- **Tools that upload whole files through a size-limited editor must not edit big files.** `web/app.js` (~90KB,
  1200+ lines) and `tests/test_*.py` were cut down or replaced by placeholders three times on 2026-11-09
  (`PLACEHOLDER_FULL_FILE`, `PLACEHOLDER_TEST`, functions turned into `/* original */`). If you can't write the
  complete file, describe the exact lines to change and let a tool with a local checkout (Codex, Claude) apply them.
  After any push, `wc -l web/app.js` should still be 1200+ and the test count should not drop.

### What happened on 2026-11-09 (why the rules above exist)
Seven commits pushed from another tool replaced `web/app.js` with the text `PLACEHOLDER_FULL_FILE` (the whole UI
would not load), removed the MCP route and error handler from `app.py`, and added an `/api/enhance-edit` route that
called a non-existent `engine.enhance_edit_prompt`. Merge `fd30ced` kept their history but restored the working
files; the editor's ✨ button, `/api/enhance-edit` (`Engine.enhance_edit`) and its test came from `4be3a2d`. Their
`translate.py` instruction text (explicit undressing / sex-position wording, commit `08adfc3`) was not merged:
the tool's prompt helpers stay neutral and expand whatever the owner writes; the owner can restore that text
from history if they want it.

