import json
import time
from pathlib import Path

from waste_early_exit.progress import ProgressReporter, expected_stage_keys


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_expected_stage_keys_include_eda_only_for_primary_seed() -> None:
    keys = expected_stage_keys("full", [42, 123, 2026])

    assert "42:eda" in keys
    assert "123:eda" not in keys
    assert "2026:eda" not in keys
    assert len(keys) == 34


def test_stage_completion_is_deduplicated(tmp_path: Path) -> None:
    reporter = ProgressReporter(
        ("42:setup",),
        tmp_path / "all.log",
        {42: tmp_path / "seed.log"},
        heartbeat_seconds=1.0,
        display=False,
    )

    with reporter.stage(42, "setup"):
        pass
    with reporter.stage(42, "setup"):
        pass
    reporter.close("complete")

    assert reporter.completed == 1
    completed = [row for row in read_records(tmp_path / "all.log") if row["kind"] == "stage_complete"]
    assert len(completed) == 2
    assert completed[-1]["overall_completed"] == 1


def test_heartbeat_emits_during_stage_and_stops_afterward(tmp_path: Path) -> None:
    aggregate = tmp_path / "all.log"
    reporter = ProgressReporter(
        ("42:audit",),
        aggregate,
        {42: tmp_path / "seed.log"},
        heartbeat_seconds=0.02,
        display=False,
    )

    with reporter.stage(42, "audit"):
        time.sleep(0.065)
    count_after_stage = sum(row["kind"] == "heartbeat" for row in read_records(aggregate))
    time.sleep(0.05)
    reporter.close("complete")

    assert count_after_stage >= 2
    assert sum(row["kind"] == "heartbeat" for row in read_records(aggregate)) == count_after_stage


def test_events_are_structured_and_mirrored_once(tmp_path: Path) -> None:
    aggregate = tmp_path / "all.log"
    seed_log = tmp_path / "seed.log"
    reporter = ProgressReporter(
        ("42:setup",),
        aggregate,
        {42: seed_log},
        display=False,
    )

    reporter.event(
        "epoch_complete",
        42,
        stage="training",
        model="resnet18",
        epoch=1,
        elapsed_seconds=2.5,
        eta_seconds=20.0,
        validation_macro_f1=0.7,
    )
    reporter.close("complete")

    for path in (aggregate, seed_log):
        matching = [row for row in read_records(path) if row["kind"] == "epoch_complete"]
        assert len(matching) == 1
        assert matching[0]["seed"] == 42
        assert matching[0]["model"] == "resnet18"
        assert matching[0]["epoch"] == 1
        assert "timestamp_utc" in matching[0]
        assert isinstance(matching[0]["run_id"], str) and matching[0]["run_id"]


def test_batch_iteration_updates_position_without_persisting_each_batch(tmp_path: Path) -> None:
    aggregate = tmp_path / "all.log"
    reporter = ProgressReporter(
        ("42:training",),
        aggregate,
        {42: tmp_path / "seed.log"},
        display=False,
    )

    observed = list(reporter.batch(["a", "b", "c"], 42, "train", model="resnet18", epoch=1))
    reporter.close("complete")

    assert observed == ["a", "b", "c"]
    assert reporter.context["batch"] == 3
    assert reporter.context["total_batches"] == 3
    assert all(row["kind"] != "batch" for row in read_records(aggregate))
