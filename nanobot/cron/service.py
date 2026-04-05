"""Cron service for scheduling agent tasks."""

import asyncio
import copy
import os
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronRunRecord, CronSchedule, CronStore

try:
    import fcntl
except ImportError:  # pragma: no cover - Unix cron workers use fcntl
    fcntl = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """Compute next run time in ms."""
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        # Next interval from now
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter
            # Use caller-provided reference time for deterministic scheduling
            base_time = now_ms / 1000
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_time, tz=tz)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception:
            return None

    return None


def _validate_schedule_for_add(schedule: CronSchedule) -> None:
    """Validate schedule fields that would otherwise create non-runnable jobs."""
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")

    if schedule.kind == "cron" and schedule.tz:
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(schedule.tz)
        except Exception:
            raise ValueError(f"unknown timezone '{schedule.tz}'") from None


class CronService:
    """Service for managing and executing scheduled jobs."""

    _MAX_RUN_HISTORY = 20
    _RUN_LEASE_MS = 15 * 60 * 1000
    _STORE_POLL_INTERVAL_MS = 5 * 1000

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
    ):
        self.store_path = store_path
        self.on_job = on_job
        self._store: CronStore | None = None
        self._last_mtime: float = 0.0
        self._timer_task: asyncio.Task | None = None
        self._running = False

    @property
    def _lock_path(self) -> Path:
        return self.store_path.with_name(f"{self.store_path.name}.lock")

    @staticmethod
    def _has_active_lease(job: CronJob, now_ms: int) -> bool:
        return bool(job.state.lease_until_ms and now_ms < job.state.lease_until_ms)

    @contextmanager
    def _store_lock(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.touch(exist_ok=True)
        lock_fd = os.open(self._lock_path, os.O_RDWR)
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._store = None
            self._last_mtime = 0.0
            yield
        finally:
            self._store = None
            self._last_mtime = 0.0
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _load_store(self) -> CronStore:
        """Load jobs from disk. Reloads automatically if file was modified externally."""
        if self._store and self.store_path.exists():
            mtime = self.store_path.stat().st_mtime
            if mtime != self._last_mtime:
                logger.info("Cron: jobs.json modified externally, reloading")
                self._store = None
        if self._store:
            return self._store

        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                jobs = []
                for j in data.get("jobs", []):
                    jobs.append(CronJob(
                        id=j["id"],
                        name=j["name"],
                        enabled=j.get("enabled", True),
                        schedule=CronSchedule(
                            kind=j["schedule"]["kind"],
                            at_ms=j["schedule"].get("atMs"),
                            every_ms=j["schedule"].get("everyMs"),
                            expr=j["schedule"].get("expr"),
                            tz=j["schedule"].get("tz"),
                        ),
                        payload=CronPayload(
                            kind=j["payload"].get("kind", "agent_turn"),
                            message=j["payload"].get("message", ""),
                            deliver=j["payload"].get("deliver", False),
                            channel=j["payload"].get("channel"),
                            to=j["payload"].get("to"),
                        ),
                        state=CronJobState(
                            next_run_at_ms=j.get("state", {}).get("nextRunAtMs"),
                            last_run_at_ms=j.get("state", {}).get("lastRunAtMs"),
                            last_status=j.get("state", {}).get("lastStatus"),
                            last_error=j.get("state", {}).get("lastError"),
                            lease_until_ms=j.get("state", {}).get("leaseUntilMs"),
                            run_history=[
                                CronRunRecord(
                                    run_at_ms=r["runAtMs"],
                                    status=r["status"],
                                    duration_ms=r.get("durationMs", 0),
                                    error=r.get("error"),
                                )
                                for r in j.get("state", {}).get("runHistory", [])
                            ],
                        ),
                        created_at_ms=j.get("createdAtMs", 0),
                        updated_at_ms=j.get("updatedAtMs", 0),
                        delete_after_run=j.get("deleteAfterRun", False),
                    ))
                self._store = CronStore(jobs=jobs)
            except Exception as e:
                logger.warning("Failed to load cron store: {}", e)
                self._store = CronStore()
        else:
            self._store = CronStore()

        return self._store

    def _save_store(self) -> None:
        """Save jobs to disk."""
        if not self._store:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self._store.version,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "enabled": j.enabled,
                    "schedule": {
                        "kind": j.schedule.kind,
                        "atMs": j.schedule.at_ms,
                        "everyMs": j.schedule.every_ms,
                        "expr": j.schedule.expr,
                        "tz": j.schedule.tz,
                    },
                    "payload": {
                        "kind": j.payload.kind,
                        "message": j.payload.message,
                        "deliver": j.payload.deliver,
                        "channel": j.payload.channel,
                        "to": j.payload.to,
                    },
                    "state": {
                        "nextRunAtMs": j.state.next_run_at_ms,
                        "lastRunAtMs": j.state.last_run_at_ms,
                        "lastStatus": j.state.last_status,
                        "lastError": j.state.last_error,
                        "leaseUntilMs": j.state.lease_until_ms,
                        "runHistory": [
                            {
                                "runAtMs": r.run_at_ms,
                                "status": r.status,
                                "durationMs": r.duration_ms,
                                "error": r.error,
                            }
                            for r in j.state.run_history
                        ],
                    },
                    "createdAtMs": j.created_at_ms,
                    "updatedAtMs": j.updated_at_ms,
                    "deleteAfterRun": j.delete_after_run,
                }
                for j in self._store.jobs
            ]
        }

        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_mtime = self.store_path.stat().st_mtime
    
    async def start(self) -> None:
        """Start the cron service."""
        self._running = True
        self._load_store()
        self._recompute_next_runs()
        self._save_store()
        self._arm_timer()
        logger.info("Cron service started with {} jobs", len(self._store.jobs if self._store else []))

    def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    def _recompute_next_runs(self) -> None:
        """Recompute next run times for all enabled jobs."""
        if not self._store:
            return
        now = _now_ms()
        for job in self._store.jobs:
            if job.enabled:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, now)

    def _get_next_wake_ms(self) -> int | None:
        """Get the earliest next run time across all jobs."""
        store = self._load_store()
        if not store:
            return None
        now = _now_ms()
        times = [
            j.state.lease_until_ms if self._has_active_lease(j, now) else j.state.next_run_at_ms
            for j in store.jobs
            if j.enabled and (j.state.next_run_at_ms or self._has_active_lease(j, now))
        ]
        return min(times) if times else None

    def _claim_due_jobs(self, now_ms: int) -> list[CronJob]:
        """Claim due jobs so only one process executes each run."""
        claimed: list[CronJob] = []
        with self._store_lock():
            store = self._load_store()
            dirty = False
            for job in store.jobs:
                if not job.enabled or not job.state.next_run_at_ms or now_ms < job.state.next_run_at_ms:
                    continue
                if self._has_active_lease(job, now_ms):
                    continue
                job.state.lease_until_ms = now_ms + self._RUN_LEASE_MS
                claimed.append(copy.deepcopy(job))
                dirty = True
            if dirty:
                self._save_store()
        return claimed

    def _finalize_job_run(
        self,
        job_id: str,
        start_ms: int,
        end_ms: int,
        status: str,
        error: str | None,
    ) -> None:
        """Persist the result of a claimed job run."""
        with self._store_lock():
            store = self._load_store()
            job = next((item for item in store.jobs if item.id == job_id), None)
            if not job:
                return

            job.state.last_status = status
            job.state.last_error = error
            job.state.last_run_at_ms = start_ms
            job.state.lease_until_ms = None
            job.updated_at_ms = end_ms

            job.state.run_history.append(CronRunRecord(
                run_at_ms=start_ms,
                status=status,
                duration_ms=end_ms - start_ms,
                error=error,
            ))
            job.state.run_history = job.state.run_history[-self._MAX_RUN_HISTORY:]

            if job.schedule.kind == "at":
                if job.delete_after_run:
                    store.jobs = [item for item in store.jobs if item.id != job.id]
                else:
                    job.enabled = False
                    job.state.next_run_at_ms = None
            else:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, end_ms)

            self._save_store()

        self._arm_timer()

    def _arm_timer(self) -> None:
        """Schedule the next timer tick."""
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if self._timer_task and self._timer_task is not current_task:
            self._timer_task.cancel()

        if not self._running:
            if self._timer_task is current_task:
                self._timer_task = None
            return

        next_wake = self._get_next_wake_ms()
        if next_wake is None:
            delay_ms = self._STORE_POLL_INTERVAL_MS
        else:
            delay_ms = max(0, next_wake - _now_ms())
            delay_ms = min(delay_ms, self._STORE_POLL_INTERVAL_MS)
        delay_s = delay_ms / 1000

        async def tick():
            try:
                await asyncio.sleep(delay_s)
                if self._running:
                    await self._on_timer()
            finally:
                if self._timer_task is asyncio.current_task():
                    self._timer_task = None

        self._timer_task = asyncio.create_task(tick())

    async def _on_timer(self) -> None:
        """Handle timer tick - run due jobs."""
        now = _now_ms()
        due_jobs = self._claim_due_jobs(now)
        self._arm_timer()

        for job in due_jobs:
            await self._execute_job(job)

    async def _execute_job(self, job: CronJob) -> None:
        """Execute a single job."""
        start_ms = _now_ms()
        logger.info("Cron: executing job '{}' ({})", job.name, job.id)

        status = "ok"
        error: str | None = None
        cancelled = False
        try:
            if self.on_job:
                await self.on_job(job)
            logger.info("Cron: job '{}' completed", job.name)
        except asyncio.CancelledError:
            status = "error"
            error = "cancelled"
            cancelled = True
            logger.warning("Cron: job '{}' was cancelled", job.name)
        except Exception as e:
            status = "error"
            error = str(e)
            logger.error("Cron: job '{}' failed: {}", job.name, e)

        end_ms = _now_ms()
        self._finalize_job_run(job.id, start_ms, end_ms, status, error)
        if cancelled:
            raise asyncio.CancelledError

    # ========== Public API ==========

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """List all jobs."""
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float('inf'))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        """Add a new job."""
        store = self._load_store()
        _validate_schedule_for_add(schedule)
        now = _now_ms()

        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )

        store.jobs.append(job)
        self._save_store()
        self._arm_timer()

        logger.info("Cron: added job '{}' ({})", name, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        store = self._load_store()
        before = len(store.jobs)
        store.jobs = [j for j in store.jobs if j.id != job_id]
        removed = len(store.jobs) < before

        if removed:
            self._save_store()
            self._arm_timer()
            logger.info("Cron: removed job {}", job_id)

        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable or disable a job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                job.enabled = enabled
                job.updated_at_ms = _now_ms()
                if enabled:
                    job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())
                else:
                    job.state.next_run_at_ms = None
                self._save_store()
                self._arm_timer()
                return job
        return None

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        """Manually run a job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                if not force and not job.enabled:
                    return False
                await self._execute_job(job)
                self._save_store()
                self._arm_timer()
                return True
        return False

    def get_job(self, job_id: str) -> CronJob | None:
        """Get a job by ID."""
        store = self._load_store()
        return next((j for j in store.jobs if j.id == job_id), None)

    def status(self) -> dict:
        """Get service status."""
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }
