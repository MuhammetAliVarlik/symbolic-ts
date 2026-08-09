import sys
import types
import warnings

import numpy as np
import pytest

from symbolic_ts import splits as sp

RNG = np.random.default_rng(seed=123)


def _synthetic_series(n: int = 500) -> np.ndarray:
    return RNG.normal(size=n)


def test_walk_forward_split_produces_n_folds():
    folds = sp.walk_forward_split(_synthetic_series(500), n_folds=4, context_len=10, embargo=2)
    assert len(folds) == 4


def test_walk_forward_split_zero_index_overlap_after_purging():
    folds = sp.walk_forward_split(_synthetic_series(500), n_folds=4, context_len=10, embargo=2)
    for fold in folds:
        assert set(fold.train_indices.tolist()).isdisjoint(fold.validation_indices.tolist())


def test_walk_forward_split_purge_gap_is_at_least_context_plus_embargo():
    context_len, embargo = 15, 3
    folds = sp.walk_forward_split(_synthetic_series(500), n_folds=3, context_len=context_len, embargo=embargo)
    for fold in folds:
        gap = fold.validation_indices.min() - fold.train_indices.max()
        assert gap >= context_len + embargo


def test_walk_forward_split_validation_blocks_are_contiguous_and_ordered():
    folds = sp.walk_forward_split(_synthetic_series(500), n_folds=3, context_len=5, embargo=0)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.validation_indices.max() < later.validation_indices.min()


def test_walk_forward_split_is_deterministic_given_the_same_seeded_input():
    data = _synthetic_series(500)
    first = sp.walk_forward_split(data, n_folds=4, context_len=10, embargo=2)
    second = sp.walk_forward_split(data, n_folds=4, context_len=10, embargo=2)
    for f1, f2 in zip(first, second):
        assert np.array_equal(f1.train_indices, f2.train_indices)
        assert np.array_equal(f1.validation_indices, f2.validation_indices)


def test_walk_forward_split_rejects_gap_that_consumes_all_training_data():
    with pytest.raises(ValueError):
        sp.walk_forward_split(_synthetic_series(20), n_folds=5, context_len=50, embargo=0)


def test_walk_forward_split_rejects_less_than_one_fold():
    with pytest.raises(ValueError):
        sp.walk_forward_split(_synthetic_series(100), n_folds=0, context_len=5, embargo=0)


def test_test_set_lock_hashes_the_locked_slice_at_creation():
    data = _synthetic_series(200)
    lock = sp.TestSetLock.create(data, test_fraction=0.2)
    assert lock.data_hash == sp._hash_array(data[lock.indices])


def test_test_set_lock_holds_out_the_most_recent_data():
    data = _synthetic_series(200)
    lock = sp.TestSetLock.create(data, test_fraction=0.2)
    assert lock.indices.tolist() == list(range(160, 200))


def test_test_set_lock_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        sp.TestSetLock.create(_synthetic_series(100), test_fraction=1.5)
    with pytest.raises(ValueError):
        sp.TestSetLock.create(_synthetic_series(100), test_fraction=0)


def test_test_set_lock_read_increments_peek_count_and_warns():
    lock = sp.TestSetLock.create(_synthetic_series(100), test_fraction=0.1)
    assert lock.test_peek_count == 0
    with pytest.warns(sp.TestSetAccessWarning):
        lock.read()
    assert lock.test_peek_count == 1
    with pytest.warns(sp.TestSetAccessWarning):
        lock.read()
    assert lock.test_peek_count == 2


def test_test_set_lock_read_logs_to_mlflow_when_available(monkeypatch):
    logged = []
    fake_mlflow = types.SimpleNamespace(log_metric=lambda name, value: logged.append((name, value)))
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    lock = sp.TestSetLock.create(_synthetic_series(100), test_fraction=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.TestSetAccessWarning)
        lock.read()

    assert logged == [("test_peek_count", 1)]


def test_test_set_lock_read_does_not_raise_when_mlflow_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    lock = sp.TestSetLock.create(_synthetic_series(100), test_fraction=0.1)
    with pytest.warns(sp.TestSetAccessWarning):
        lock.read()  # must not raise even though "import mlflow" fails


def test_test_set_lock_save_load_round_trip(tmp_path):
    lock = sp.TestSetLock.create(_synthetic_series(150), test_fraction=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.TestSetAccessWarning)
        lock.read()

    path = tmp_path / "test_lock.json"
    lock.save(path)
    loaded = sp.TestSetLock.load(path)

    assert loaded.data_hash == lock.data_hash
    assert loaded.test_peek_count == lock.test_peek_count == 1
    assert loaded.indices.tolist() == lock.indices.tolist()
    assert loaded.schema_version == lock.schema_version
