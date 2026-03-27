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
