from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_nanobot_home(monkeypatch, tmp_path_factory):
    nanobot_home = tmp_path_factory.mktemp("nanobot-home")
    monkeypatch.setenv("NANOBOT_HOME", str(nanobot_home))
    return nanobot_home
