"""Abstracts away file path operations."""

import datetime as dt
import shutil
from pathlib import Path
from tempfile import mkdtemp


class WorkingDirectory:
    """Abstracts away file path operations.

    Probably should be instantiated using for_metaflow_run
    unless you know what you're doing.
    """

    def __init__(self, path: Path) -> None:
        """Creates a new WorkingDirectory."""
        self.path = path
        self.path.mkdir(mode=0o755, parents=True, exist_ok=True)

    @classmethod
    def for_metaflow_run(cls, base: str, run_id: int) -> "WorkingDirectory":
        """Creates a new WorkingDirectory for a Metaflow run."""
        return cls(
            Path(
                mkdtemp(
                    prefix=base + "::" + str(run_id) + "::",
                    suffix="::" + dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M"),
                ),
            ),
        )

    def push_segment(self, segment: str) -> "WorkingDirectory":
        """Creates a new subdirectory of the WorkingDirectory."""
        return WorkingDirectory(self.path / segment)

    def cleanup(self) -> None:
        shutil.rmtree(str(self.path))

    def __str__(self) -> str:
        return str(self.path)
