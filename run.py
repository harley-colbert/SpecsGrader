import asyncio
import threading
import time
from typing import Optional

import uvicorn
import webview

from backend.app.main import create_app
from backend.app.settings import get_settings


class ServerController:
    """Manage the Uvicorn server lifecycle."""

    def __init__(self, port: int):
        self.port = port
        config = uvicorn.Config(
            app=create_app,
            host="127.0.0.1",
            port=self.port,
            log_level="info",
            factory=True,
        )
        self.server = uvicorn.Server(config)
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the server in a background thread."""

        def _run_server() -> None:
            asyncio.run(self.server.serve())

        self.thread = threading.Thread(target=_run_server, daemon=True)
        self.thread.start()

        while not self.server.started and not self.server.should_exit:
            time.sleep(0.05)

    def stop(self) -> None:
        """Signal the server to shut down and wait for completion."""

        self.server.should_exit = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)


def main() -> None:
    settings = get_settings()
    controller = ServerController(port=settings.port)
    controller.start()

    url = f"http://127.0.0.1:{settings.port}/"

    webview.create_window("SpecsGrader", url, width=1200, height=800)
    webview.start(debug=True)
    controller.stop()


if __name__ == "__main__":
    main()
