from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure the repository root is on sys.path.

    In some environments, pytest is invoked from a different working directory,
    which can cause `import backend...` to fail during collection.
    """

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
