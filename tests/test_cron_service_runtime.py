import asyncio
import time

import pytest

from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


@pytest.mark.asyncio
async def test_due_job_runs_once_across_multiple_services(tmp_path) -> None:
    store_path = tmp_path / "jobs.json"
    run_at_ms = int(time.time() * 1000) + 50
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def on_job(job) -> None:
        calls.append(job.id)
        started.set()
        await release.wait()

    service_a = CronService(store_path, on_job=on_job)
    service_b = CronService(store_path, on_job=on_job)
    job = service_a.add_job(
        name="once",
        schedule=CronSchedule(kind="at", at_ms=run_at_ms),
        message="hello",
    )

    await asyncio.sleep(0.08)
    task_a = asyncio.create_task(service_a._on_timer())
    await started.wait()
    task_b = asyncio.create_task(service_b._on_timer())
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(task_a, task_b)

    assert calls == [job.id]

    stored = service_a.get_job(job.id)
    assert stored is not None
    assert stored.enabled is False
    assert stored.state.last_status == "ok"
    assert stored.state.lease_until_ms is None


def test_claimed_job_can_run_again_after_lease_expires(tmp_path) -> None:
    store_path = tmp_path / "jobs.json"
    run_at_ms = int(time.time() * 1000) + 50
    service_a = CronService(store_path)
    service_b = CronService(store_path)
    service_a.add_job(
        name="retryable",
        schedule=CronSchedule(kind="at", at_ms=run_at_ms),
        message="hello",
    )

    time.sleep(0.08)
    now_ms = int(time.time() * 1000)
    first_claim = service_a._claim_due_jobs(now_ms)
    second_claim = service_b._claim_due_jobs(now_ms + CronService._RUN_LEASE_MS + 1)

    assert len(first_claim) == 1
    assert len(second_claim) == 1


@pytest.mark.asyncio
async def test_scheduled_timer_does_not_cancel_running_job(tmp_path) -> None:
    store_path = tmp_path / "jobs.json"
    run_at_ms = int(time.time() * 1000) + 100
    calls: list[str] = []

    async def on_job(job) -> None:
        calls.append(job.id)
        await asyncio.sleep(0)
        await asyncio.sleep(0.05)

    service = CronService(store_path, on_job=on_job)
    job = service.add_job(
        name="scheduled",
        schedule=CronSchedule(kind="at", at_ms=run_at_ms),
        message="hello",
    )

    await service.start()
    await asyncio.sleep(0.3)
    service.stop()

    assert calls == [job.id]
    stored = service.get_job(job.id)
    assert stored is not None
    assert stored.enabled is False
    assert stored.state.last_status == "ok"
    assert stored.state.lease_until_ms is None


@pytest.mark.asyncio
async def test_cancelled_job_releases_lease_and_records_error(tmp_path) -> None:
    store_path = tmp_path / "jobs.json"
    service = CronService(store_path)
    job = service.add_job(
        name="cancelled",
        schedule=CronSchedule(kind="at", at_ms=int(time.time() * 1000) + 1000),
        message="hello",
    )

    async def on_job(_job) -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)

    service.on_job = on_job

    with pytest.raises(asyncio.CancelledError):
        await service.run_job(job.id)

    stored = service.get_job(job.id)
    assert stored is not None
    assert stored.state.last_status == "error"
    assert stored.state.last_error == "cancelled"
    assert stored.state.lease_until_ms is None


def test_get_next_wake_reloads_store_after_lock(tmp_path) -> None:
    store_path = tmp_path / "jobs.json"
    service = CronService(store_path)
    job = service.add_job(
        name="wake",
        schedule=CronSchedule(kind="at", at_ms=int(time.time() * 1000) + 60_000),
        message="hello",
    )

    service._store = None

    assert service._get_next_wake_ms() == job.state.next_run_at_ms


@pytest.mark.asyncio
async def test_running_service_picks_up_external_jobs(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "jobs.json"
    calls: list[str] = []

    async def on_job(job) -> None:
        calls.append(job.id)

    monkeypatch.setattr(CronService, "_STORE_POLL_INTERVAL_MS", 20)

    service = CronService(store_path, on_job=on_job)
    await service.start()

    external = CronService(store_path)
    job = external.add_job(
        name="external",
        schedule=CronSchedule(kind="at", at_ms=int(time.time() * 1000) + 80),
        message="hello",
    )

    await asyncio.sleep(0.3)
    service.stop()

    assert calls == [job.id]
    stored = service.get_job(job.id)
    assert stored is not None
    assert stored.enabled is False
    assert stored.state.last_status == "ok"
