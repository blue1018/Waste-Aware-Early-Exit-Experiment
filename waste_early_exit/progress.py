from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

from tqdm.auto import tqdm


_T = TypeVar("_T")

PIPELINE_STAGES = (
    "setup",
    "audit",
    "splits",
    "static_training",
    "early_training",
    "calibration",
    "threshold_search",
    "freeze",
    "locked_test",
    "ablations",
    "profiling",
)


def expected_stage_keys(mode: str, seeds: Sequence[int]) -> tuple[str, ...]:
    selected = tuple(int(seed) for seed in seeds)
    if not selected:
        raise ValueError("At least one seed is required for progress tracking")
    keys: list[str] = []
    for index, seed in enumerate(selected):
        stages = list(PIPELINE_STAGES)
        if index == 0:
            stages.insert(3, "eda")
        keys.extend(f"{seed}:{stage}" for stage in stages)
    return tuple(keys)


class ProgressReporter:
    def __init__(
        self,
        expected_keys: Sequence[str],
        aggregate_log_path: str | Path,
        seed_log_paths: Mapping[int, str | Path],
        heartbeat_seconds: float = 30.0,
        display: bool = True,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.expected_keys = tuple(str(key) for key in expected_keys)
        self.aggregate_log_path = Path(aggregate_log_path)
        self.seed_log_paths = {int(seed): Path(path) for seed, path in seed_log_paths.items()}
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.display = bool(display)
        self._lock = threading.RLock()
        self._completed_keys: set[str] = set()
        self._context: dict[str, Any] = {}
        self._context_stack: list[dict[str, Any]] = []
        self._stage_depth = 0
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._run_started = time.monotonic()
        self.run_id = uuid.uuid4().hex
        self._stage_started = self._run_started
        self._last_progress = self._run_started
        self._last_display_refresh = 0.0
        self._closed = False
        self.aggregate_log_path.parent.mkdir(parents=True, exist_ok=True)
        for path in self.seed_log_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        self._overall_bar = (
            tqdm(
                total=len(self.expected_keys),
                desc="Overall experiment",
                unit="stage",
                position=0,
                leave=True,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if self.display
            else None
        )

    @property
    def completed(self) -> int:
        with self._lock:
            return len(self._completed_keys)

    @property
    def context(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._context)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def _write_record(self, record: dict[str, Any]) -> None:
        seed = record.get("seed")
        paths = [self.aggregate_log_path]
        if seed is not None and int(seed) in self.seed_log_paths:
            paths.append(self.seed_log_paths[int(seed)])
        serialized = json.dumps(record, sort_keys=True, default=self._json_default)
        for path in dict.fromkeys(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
                stream.flush()

    def _refresh_display(self, force: bool = False) -> None:
        if self._overall_bar is None:
            return
        now = time.monotonic()
        if not force and now - self._last_display_refresh < 1.0:
            return
        self._last_display_refresh = now
        context = self._context
        seed = context.get("seed", "-")
        stage = context.get("stage", "idle")
        detail = context.get("model") or context.get("variant") or context.get("phase") or ""
        self._overall_bar.set_description_str(f"Overall seed={seed} stage={stage}", refresh=False)
        fields = []
        if detail:
            fields.append(str(detail))
        if "epoch" in context:
            fields.append(f"epoch={context['epoch']}/{context.get('total_epochs', '?')}")
        if "batch" in context:
            fields.append(f"batch={context['batch']}/{context.get('total_batches', '?')}")
        fields.append(f"alive={int(now - self._run_started)}s")
        self._overall_bar.set_postfix_str(" ".join(fields), refresh=True)

    def event(self, kind: str, seed: int | None, **fields: Any) -> None:
        with self._lock:
            record: dict[str, Any] = {
                "timestamp_utc": self._timestamp(),
                "run_id": self.run_id,
                "kind": str(kind),
                "seed": int(seed) if seed is not None else None,
            }
            record.update(fields)
            self._write_record(record)
            self._refresh_display(force=True)

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._context.update(fields)
            self._last_progress = time.monotonic()
            self._refresh_display()

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            with self._lock:
                if self._stage_depth <= 0 or stop.is_set():
                    return
                now = time.monotonic()
                context = dict(self._context)
                seed = context.pop("seed", None)
                self.event(
                    "heartbeat",
                    seed,
                    **context,
                    elapsed_seconds=now - self._run_started,
                    stage_elapsed_seconds=now - self._stage_started,
                    seconds_since_progress=now - self._last_progress,
                )

    def _start_heartbeat(self) -> None:
        stop = threading.Event()
        thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(stop,),
            name="waste-early-exit-heartbeat",
            daemon=True,
        )
        self._heartbeat_stop = stop
        self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat(self) -> None:
        with self._lock:
            stop = self._heartbeat_stop
            thread = self._heartbeat_thread
            self._heartbeat_stop = None
            self._heartbeat_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.heartbeat_seconds * 2))

    @contextmanager
    def stage(self, seed: int, name: str, **context: Any) -> Iterator[None]:
        started = time.monotonic()
        with self._lock:
            if self._closed:
                raise RuntimeError("Progress reporter is closed")
            self._context_stack.append(dict(self._context))
            self._context = {"seed": int(seed), "stage": str(name), **context}
            self._stage_depth += 1
            self._stage_started = started
            self._last_progress = started
            start_heartbeat = self._stage_depth == 1
            self.event("stage_started", seed, stage=name, **context)
        if start_heartbeat:
            self._start_heartbeat()
        try:
            yield
        except BaseException as error:
            self.event(
                "stage_failed",
                seed,
                stage=name,
                elapsed_seconds=time.monotonic() - started,
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
            raise
        else:
            key = f"{int(seed)}:{name}"
            with self._lock:
                if key in self.expected_keys and key not in self._completed_keys:
                    self._completed_keys.add(key)
                    if self._overall_bar is not None:
                        self._overall_bar.update(1)
                self.event(
                    "stage_complete",
                    seed,
                    stage=name,
                    elapsed_seconds=time.monotonic() - started,
                    overall_completed=len(self._completed_keys),
                    overall_total=len(self.expected_keys),
                )
        finally:
            with self._lock:
                self._stage_depth -= 1
                previous = self._context_stack.pop()
                if self._stage_depth > 0:
                    self._context = previous
                stop_heartbeat = self._stage_depth == 0
            if stop_heartbeat:
                self._stop_heartbeat()

    def batch(
        self,
        iterable: Iterable[_T],
        seed: int,
        phase: str,
        **context: Any,
    ) -> Iterator[_T]:
        total = len(iterable) if hasattr(iterable, "__len__") else None
        description = f"seed={seed} {context.get('model', context.get('label', ''))} {phase}".strip()
        wrapped: Iterable[_T]
        if self.display:
            wrapped = tqdm(
                iterable,
                total=total,
                desc=description,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                mininterval=5.0,
                maxinterval=30.0,
            )
        else:
            wrapped = iterable
        try:
            for index, item in enumerate(wrapped, start=1):
                self.update(
                    seed=int(seed),
                    phase=str(phase),
                    batch=index,
                    total_batches=total,
                    **context,
                )
                yield item
        finally:
            close = getattr(wrapped, "close", None)
            if callable(close):
                close()

    def close(self, status: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            seed = self._context.get("seed")
        self._stop_heartbeat()
        self.event(
            "run_complete",
            int(seed) if seed is not None else None,
            status=str(status),
            elapsed_seconds=time.monotonic() - self._run_started,
            overall_completed=self.completed,
            overall_total=len(self.expected_keys),
        )
        if self._overall_bar is not None:
            self._overall_bar.close()
