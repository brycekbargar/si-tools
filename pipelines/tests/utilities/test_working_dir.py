from unittest.mock import patch, MagicMock
from pathlib import Path

from utilities.working_dir import WorkingDirectory as uut


class TestWorkingDirectory:
    @patch.object(Path, "mkdir")
    def test_dirs_are_created(self, mock_exists: MagicMock):
        uut(Path("path"))
        uut.for_metaflow_run(Path("base"), 123)

        assert mock_exists.call_count == 2

    @patch.object(Path, "mkdir")
    def test_segments(self, _: MagicMock):
        base = uut.for_metaflow_run(Path("/base"), 123)
        segment1 = base.push_segment("segment1")
        segment2 = base.push_segment("segment").push_segment("two")

        assert base.file("base.txt") == "/base/123/base.txt"
        assert segment1.file("segment1.txt") == "/base/123/segment1/segment1.txt"
        assert segment2.file("segment2.txt") == "/base/123/segment/two/segment2.txt"

    @patch.object(Path, "mkdir")
    def test_partitions(self, _: MagicMock):
        base = uut.for_metaflow_run(Path("/base"), 123)
        partition1 = base.push_partitions(("key1", "value1"))
        partition2 = base.push_partitions(("key1", "value1"), ("key2", "value2"))
        partition3 = partition1.push_partitions(("key2", "value2"))

        assert base.glob_keys("base.parquet") == "/base/123/base.parquet"
        assert (
            partition1.glob_keys("partition1.parquet")
            == "/base/123/key1=value1/partition1.parquet"
        )
        assert (
            partition2.glob_keys("partition2.parquet")
            == "/base/123/key1=value1&key2=value2/partition2.parquet"
        )
        assert (
            partition3.glob_keys("partition3.parquet")
            == "/base/123/key1=value1/key2=value2/partition3.parquet"
        )

        assert partition1.glob_keys("*.parquet", "key1") == "/base/123/key1=*/*.parquet"
        assert (
            partition2.glob_keys("*.parquet", "key2")
            == "/base/123/key1=value1&key2=*/*.parquet"
        )
        assert (
            partition3.glob_keys("*.parquet", "key1", "key2")
            == "/base/123/key1=*/key2=*/*.parquet"
        )
