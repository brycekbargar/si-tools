from typing import Any
from pathlib import Path
import re


class WorkingDirectory:
    """Abstracts away file path operations.

    Probably should be instantiated using for_metaflow_run
    unless you know what you're doing.
    """

    def __init__(self, path: Path):
        """Creates a new WorkingDirectory."""

        self._path = path
        self._path.mkdir(mode=0o755, parents=True, exist_ok=True)

    @classmethod
    def for_metaflow_run(cls, base: Path, run_id: int) -> "WorkingDirectory":
        """Creates a new WorkingDirectory for a Metaflow run."""

        return cls(base / str(run_id))

    def push_segment(self, segment: str) -> "WorkingDirectory":
        """Creates a new subdirectory of the WorkingDirectory."""

        return WorkingDirectory(self._path / segment)

    def push_partitions(self, *args: tuple[str, Any]) -> "WorkingDirectory":
        """Creates a new subdirectory of the WorkingDirectory.

        The subdirectory name will be based on the passed key/value pairs
        allowing for partitioning data based on its content.
        """

        return WorkingDirectory(
            self._path / "&".join([f"{k}={str(v)}" for (k, v) in args])
        )

    def glob_keys(self, file_glob: str, *args: str) -> str:
        """Gets the current WorkingDirectory + given file_glob as a string.

        Passing keys will additionally glob the value of those keys in the partitioning scheme."""

        glob = str(self._path / file_glob)
        for k in args:
            glob = re.sub(rf"({k})=\w*", r"\1=*", glob)
        return glob

    def file(self, file: str) -> str:
        """Gets the current WorkingDirectory + given file as a string."""

        return str(self._path / file)

    def __str__(self):
        return str(self._path)
