import sys
import types

import pytest

from symbolic_ts import events as ev
from symbolic_ts import vocabulary as v


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path, monkeypatch):
    # TestSetLock.read() logs to MLflow when it is installed. Point it at a
    # per-test directory, and close the run it opens before the env var is
    # restored: an active run left open is ended at interpreter exit against
    # the default tracking URI, which writes ./mlruns into the repo.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    yield
    mlflow = sys.modules.get("mlflow")
    if isinstance(mlflow, types.ModuleType) and mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture(autouse=True)
def _isolated_event_registries():
    # Constructing an event adapter registers its tokens in a module-level
    # set; without a reset, whether "E_WEEKEND" validates would depend on
    # which tests happened to run first.
    tokens, adapters = set(v._registered_event_tokens), dict(ev._registered_adapters)
    yield
    v._registered_event_tokens.clear()
    v._registered_event_tokens.update(tokens)
    ev._registered_adapters.clear()
    ev._registered_adapters.update(adapters)
