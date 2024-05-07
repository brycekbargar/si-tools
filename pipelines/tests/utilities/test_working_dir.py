from pathlib import Path
from unittest.mock import MagicMock, patch

from utilities.working_dir import WorkingDirectory as uut


class TestWorkingDirectory:
    @patch("utilities.working_dir.mkdtemp")
    @patch.object(Path, "mkdir")
    def test_dirs_are_created(
        self,
        mock_mkdir: MagicMock,
        mkdtemp_mock: MagicMock,
    ) -> None:
        uut(Path("path"))
        uut.for_metaflow_run("base", 123)

        assert mock_mkdir.call_count == 2
        assert mkdtemp_mock.call_count == 1

    @patch("utilities.working_dir.mkdtemp")
    def test_segments(self, mkdtemp_mock: MagicMock) -> None:
        with patch.object(Path, "mkdir"):
            mkdtemp_mock.return_value = "/test"
            base = uut.for_metaflow_run("base", 123)

            segment1 = base.push_segment("segment1")
            segment2 = base.push_segment("segment").push_segment("two")

        assert base.file("base.txt") == "/test/base.txt"
        assert segment1.file("segment1.txt") == "/test/segment1/segment1.txt"
        assert segment2.file("segment2.txt") == "/test/segment/two/segment2.txt"

    @patch("utilities.working_dir.mkdtemp")
    def test_partitions(self, mkdtemp_mock: MagicMock) -> None:
        with patch.object(Path, "mkdir"):
            mkdtemp_mock.return_value = "/test"
            base = uut.for_metaflow_run("base", 123)

            partition1 = base.push_partitions(("key1", "value1"))
            partition2 = base.push_partitions(("key1", "value1"), ("key2", "value2"))
            partition3 = partition1.push_partitions(("key2", "value2"))

        assert base.glob_keys("base.parquet") == "/test/base.parquet"
        assert (
            partition1.glob_keys("partition1.parquet")
            == "/test/key1=value1/partition1.parquet"
        )
        assert (
            partition2.glob_keys("partition2.parquet")
            == "/test/key1=value1&key2=value2/partition2.parquet"
        )
        assert (
            partition3.glob_keys("partition3.parquet")
            == "/test/key1=value1/key2=value2/partition3.parquet"
        )

        assert partition1.glob_keys("*.parquet", "key1") == "/test/key1=*/*.parquet"
        assert (
            partition2.glob_keys("*.parquet", "key2")
            == "/test/key1=value1&key2=*/*.parquet"
        )
        assert (
            partition3.glob_keys("*.parquet", "key1", "key2")
            == "/test/key1=*/key2=*/*.parquet"
        )
