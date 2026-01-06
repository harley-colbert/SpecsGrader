from dataclasses import dataclass
import os


@dataclass
class Settings:
    """Application settings.

    Attributes:
        port: Port for the local server to listen on.
        never_send_externally: Flag to prevent external calls.
    """

    port: int = int(os.getenv("APP_PORT", "7860"))
    never_send_externally: bool = os.getenv("NEVER_SEND_EXTERNALLY", "0") == "1"


def get_settings() -> Settings:
    """Return settings instance.

    This indirection keeps settings import-friendly for tests.
    """

    return Settings()
