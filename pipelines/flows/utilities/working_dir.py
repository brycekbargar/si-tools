"""Abstracts away file path operations."""

import datetime as dt
import shutil
from functools import reduce
from pathlib import Path
from tempfile import mkdtemp
from typing import Any


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
        shutil.rmtree(str(self.ephemeral))

    def push_partitions(self, *args: tuple[str, Any]) -> "WorkingDirectory":
        """Creates a new subdirectory of the WorkingDirectory.

        The subdirectory name will be based on the passed key/value pairs
        allowing for partitioning data based on its content.
        """
        return reduce(
            lambda prev, part: WorkingDirectory(prev.path / f"{part[0]}={part[1]!s}"),
            args,
            self,
        )

    def glob_partitions(self, *args: str) -> "WorkingDirectory":
        """Gets the current WorkingDirectory + given file_glob as a string.

        Passing keys will additionally glob the value of those keys
        if they exist in the partitioning scheme.
        """
        return reduce(
            lambda prev, key: WorkingDirectory(prev.path / f"{key}=*"),
            args,
            self,
        )

    def file(self, file: str) -> str:
        """Gets the current WorkingDirectory + given file as a string."""
        return str(self.path / file)

    def directory(self, directory: str | None = None) -> str:
        """Gets the current WorkingDirectory + given directory as a string."""
        return str((self.path if directory is None else self.path / directory) / "_")[
            :-1
        ]

    def __str__(self) -> str:
        return str(self.path)
