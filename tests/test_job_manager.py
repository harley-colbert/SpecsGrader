import time

from backend.app.services.job_manager import JobManager


def test_job_progress_and_completion():
    rows = [{"id": i} for i in range(5)]
    jm = JobManager()
    jm.start(rows, lambda row: row)
    jm._thread.join(timeout=2)
    status = jm.current_status()
    assert status["status"] == "completed"
    assert status["progress"] == 1.0
    assert status["processed"] == 5


def test_job_cancel():
    rows = [{"id": i} for i in range(10)]
    jm = JobManager()
    jm.start(rows, lambda row: row, delay=0.05)
    time.sleep(0.1)
    jm.cancel()
    jm._thread.join(timeout=2)
    status = jm.current_status()
    assert status["status"] == "canceled"
    assert status["processed"] < 10
