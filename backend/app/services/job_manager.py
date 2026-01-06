import threading
import time
from typing import Callable, Dict, List


class JobManager:
    def __init__(self):
        self.status: Dict[str, object] = {
            "status": "idle",
            "progress": 0.0,
            "processed": 0,
            "total": 0,
            "results": [],
        }
        self._thread: threading.Thread | None = None
        self._cancel = False

    def start(
        self,
        rows: List[Dict[str, object]],
        worker: Callable[[Dict[str, object]], Dict[str, object]],
        delay: float = 0.0,
    ) -> None:
        if self.status.get("status") == "running":
            return
        self._cancel = False
        self.status.update(
            {"status": "running", "progress": 0.0, "processed": 0, "total": len(rows), "results": []}
        )

        def _run():
            total = len(rows) or 1
            for idx, row in enumerate(rows):
                if self._cancel:
                    self.status["status"] = "canceled"
                    return
                result = worker(row)
                self.status["results"].append(result)
                self.status["processed"] = idx + 1
                self.status["progress"] = min(1.0, (idx + 1) / total)
                if delay:
                    time.sleep(delay)
            self.status["status"] = "completed"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel = True

    def current_status(self) -> Dict[str, object]:
        return dict(self.status)


__all__ = ["JobManager"]
