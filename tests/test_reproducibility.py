import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from _pipeline import synthetic_finance, tokenize_split
from symbolic_ts import reproducibility as rp
from symbolic_ts.projection import FINANCE_ADAPTER
from symbolic_ts.vocabulary import SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _restore_hash_seed_env():
    before = os.environ.get("PYTHONHASHSEED")
    yield
    if before is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = before


def _stochastic_pipeline() -> list[str]:
    """Tokens from a series whose noise comes from the GLOBAL generators --
    exactly what set_seed controls."""
    raw = synthetic_finance(n=600)
    noise = np.random.normal(0.0, 0.002, len(raw)) + random.gauss(0.0, 0.001)
    raw["Adj Close"] = raw["Adj Close"] * np.exp(noise)
    return tokenize_split(raw, FINANCE_ADAPTER)["tokens"]


def test_same_seed_runs_the_pipeline_twice_with_identical_output():
    with rp.reproducible_run(42):
        first = _stochastic_pipeline()
    with rp.reproducible_run(42):
        second = _stochastic_pipeline()
    assert first == second


def test_a_different_seed_changes_the_output():
    # guards the test above: identical output must come from the seed, not
    # from a pipeline that ignores the random draws
    rp.set_seed(42)
    first = _stochastic_pipeline()
    rp.set_seed(43)
    assert _stochastic_pipeline() != first


def test_set_seed_seeds_random_and_numpy_global_generators():
    rp.set_seed(7)
    a = (random.random(), np.random.random())
    rp.set_seed(7)
    assert (random.random(), np.random.random()) == a


def test_set_seed_does_not_reach_an_unseeded_default_rng():
    # documented limit: default_rng() needs its own seed
    rp.set_seed(7)
    a = np.random.default_rng().random()
    rp.set_seed(7)
    assert np.random.default_rng().random() != a


def test_set_seed_sets_pythonhashseed_for_child_processes():
    os.environ.pop("PYTHONHASHSEED", None)
    report = rp.set_seed(1234)
    child = subprocess.run([sys.executable, "-c", "import os; print(os.environ['PYTHONHASHSEED'])"],
                           capture_output=True, text=True, check=True)
    assert child.stdout.strip() == "1234"
    # this interpreter started without it, so its own string hashing is unchanged
    assert not report.hash_seed_applies_to_current_process


def test_hash_seed_flag_is_true_only_when_the_process_already_started_with_it():
    os.environ["PYTHONHASHSEED"] = "99"
    assert rp.set_seed(99).hash_seed_applies_to_current_process
    assert not rp.set_seed(100).hash_seed_applies_to_current_process


def test_set_seed_seeds_torch_cpu_and_cuda_when_installed(monkeypatch):
    calls = []
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda s: calls.append(("cpu", s))
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True,
                                            manual_seed_all=lambda s: calls.append(("cuda", s)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert rp.set_seed(5).torch_seeded
    assert calls == [("cpu", 5), ("cuda", 5)]


def test_set_seed_works_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert not rp.set_seed(5).torch_seeded


def test_fingerprint_records_python_platform_packages_and_schema():
    fp = rp.environment_fingerprint(repo_path=REPO_ROOT)
    assert fp["python_version"] == platform.python_version()
    assert fp["platform"] == platform.platform()
    assert fp["schema_version"] == SCHEMA_VERSION
    assert fp["packages"]["numpy"] == importlib.metadata.version("numpy")


def test_fingerprint_marks_a_missing_package_as_none():
    fp = rp.environment_fingerprint(packages=("numpy", "no-such-package-xyz"))
    assert fp["packages"]["no-such-package-xyz"] is None


def test_fingerprint_records_the_git_commit_and_dirty_state():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    fp = rp.environment_fingerprint(repo_path=REPO_ROOT)
    assert fp["git_commit"] == head
    assert fp["git_dirty"] is bool(status.strip())


def test_fingerprint_outside_a_git_repo_reports_none(tmp_path):
    fp = rp.environment_fingerprint(repo_path=tmp_path)
    assert fp["git_commit"] is None
    assert fp["git_dirty"] is None


def test_reproducible_run_logs_seed_and_fingerprint_to_mlflow():
    mlflow = pytest.importorskip("mlflow")
    with rp.reproducible_run(42, run_name="fingerprint-test", repo_path=REPO_ROOT) as run:
        pass
    logged = mlflow.get_run(run.mlflow_run_id)
    assert logged.data.params["seed"] == "42"
    assert logged.data.params["env.python_version"] == platform.python_version()
    assert logged.data.params["env.schema_version"] == SCHEMA_VERSION
    assert logged.data.params["env.git_commit"] == str(run.fingerprint["git_commit"])
    artifact = mlflow.artifacts.load_text(f"runs:/{run.mlflow_run_id}/environment_fingerprint.json")
    assert json.loads(artifact) == run.fingerprint


def test_reproducible_run_still_seeds_without_mlflow(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with rp.reproducible_run(3) as run:
        value = random.random()
    random.seed(3)
    assert value == random.random()
    assert run.mlflow_run_id is None
