"""Abstracts away file path operations."""

import datetime as dt
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
        self._path = path
        self._path.mkdir(mode=0o755, parents=True, exist_ok=True)

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
        return WorkingDirectory(self._path / segment)

    def push_partitions(self, *args: tuple[str, Any]) -> "WorkingDirectory":
        """Creates a new subdirectory of the WorkingDirectory.

        The subdirectory name will be based on the passed key/value pairs
        allowing for partitioning data based on its content.
        """
        return reduce(
            lambda prev, part: WorkingDirectory(prev._path / f"{part[0]}={part[1]!s}"),  # noqa: SLF001
            args,
            self,
        )

    def glob_partitions(self, *args: str) -> "WorkingDirectory":
        """Gets the current WorkingDirectory + given file_glob as a string.

        Passing keys will additionally glob the value of those keys
        if they exist in the partitioning scheme.
        """
        return reduce(
            lambda prev, key: WorkingDirectory(prev._path / f"{key}=*"),  # noqa: SLF001
            args,
            self,
        )

    def file(self, file: str) -> str:
        """Gets the current WorkingDirectory + given file as a string."""
        return str(self._path / file)

    def directory(self, directory: str | None = None) -> str:
        """Gets the current WorkingDirectory + given directory as a string."""
        return str((self._path if directory is None else self._path / directory) / "_")[
            :-1
        ]

    def __str__(self) -> str:
        return str(self._path)
