import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_comfy import FakeComfy  # noqa: E402
from genstudio.config import Settings  # noqa: E402


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _frozen_model_list(monkeypatch):
    """Tests run against a fixed copy of the model list, so changing the real lineup doesn't rewrite them."""
    monkeypatch.setattr("genstudio.registry.CONFIG_DIR", FIXTURES)


@pytest.fixture
def settings(tmp_path):
    return Settings(comfy_url="http://comfy.test", comfy_path=None, output_dir=tmp_path / "outputs")


@pytest.fixture
def fake():
    return FakeComfy()


@pytest.fixture
def client(settings, fake, monkeypatch, tmp_path):
    monkeypatch.setattr("genstudio.config.SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr("genstudio.config.DATA_DIR", tmp_path)
    from fastapi.testclient import TestClient
    from genstudio.app import create_app
    app = create_app(settings, comfy_transport=fake.transport())
    with TestClient(app) as c:
        yield c
