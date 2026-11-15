import pytest

from src.model.run_checkpoint import (
    CheckpointConflictError,
    SQLiteCheckpointStore,
)


def test_completed_batches_survive_restart_and_only_incomplete_batches_resume(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    store = SQLiteCheckpointStore(path)
    store.start_run("run-1", "history", "input-v1", {"data_version": "daily-r1"})
    store.register_batches("run-1", "history", {
        "000001-000100": "symbols-a",
        "000101-000200": "symbols-b",
        "000201-000300": "symbols-c",
    })
    store.mark_started("run-1", "history", "000001-000100")
    store.mark_completed("run-1", "history", "000001-000100", {"rows": 100})
    store.mark_started("run-1", "history", "000101-000200")
    # Simulate process termination while the second batch is running.

    resumed = SQLiteCheckpointStore(path)
    assert resumed.pending_batches("run-1", "history") == [
        "000101-000200", "000201-000300"
    ]
    assert resumed.load_result("run-1", "history", "000001-000100") == {"rows": 100}
    assert resumed.summary("run-1", "history") == {
        "total": 3, "pending": 1, "running": 1, "completed": 1, "failed": 0
    }


def test_failed_batch_can_be_retried_without_repeating_completed_batches(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite")
    store.start_run("run-2", "scan", "snapshot-r3")
    store.register_batches("run-2", "scan", {"batch-a": "a1", "batch-b": "b1"})
    store.mark_started("run-2", "scan", "batch-a")
    store.mark_completed("run-2", "scan", "batch-a", [1, 2])
    store.mark_started("run-2", "scan", "batch-b")
    store.mark_failed("run-2", "scan", "batch-b", "temporary provider error")

    assert store.pending_batches("run-2", "scan") == ["batch-b"]
    store.mark_started("run-2", "scan", "batch-b")
    store.mark_completed("run-2", "scan", "batch-b", [3, 4])
    assert store.pending_batches("run-2", "scan") == []
    assert store.summary("run-2", "scan")["completed"] == 2


def test_run_and_batch_fingerprints_prevent_unsafe_resume(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite")
    store.start_run("run-3", "backtest", "prices-v1")
    store.register_batches("run-3", "backtest", {"2026-01": "jan-v1"})

    with pytest.raises(CheckpointConflictError, match="run input fingerprint changed"):
        store.start_run("run-3", "backtest", "prices-v2")
    with pytest.raises(CheckpointConflictError, match="batch input fingerprint changed"):
        store.register_batches("run-3", "backtest", {"2026-01": "jan-v2"})


def test_unknown_or_incomplete_batch_result_is_not_returned(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "state.sqlite")
    store.start_run("run-4", "scan", "input-v1")
    store.register_batches("run-4", "scan", {"batch-a": "a1"})

    with pytest.raises(ValueError, match="not completed"):
        store.load_result("run-4", "scan", "batch-a")
    with pytest.raises(KeyError, match="unknown batch"):
        store.mark_completed("run-4", "scan", "missing", {})
