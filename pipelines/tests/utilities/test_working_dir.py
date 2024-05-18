from pathlib import Path
from unittest.mock import MagicMock, patch

from flows.utilities.working_dir import WorkingDirectory as uut


@patch("flows.utilities.working_dir.mkdtemp")
@patch.object(Path, "mkdir")
def test_dirs_are_created(
    mock_mkdir: MagicMock,
    mkdtemp_mock: MagicMock,
) -> None:
    uut(Path("path"))
    uut.for_metaflow_run("base", 123)

    assert mock_mkdir.call_count == 2
    assert mkdtemp_mock.call_count == 1


@patch("flows.utilities.working_dir.mkdtemp")
def test_push_segment(mkdtemp_mock: MagicMock) -> None:
    with patch.object(Path, "mkdir"):
        mkdtemp_mock.return_value = "/test"
        base = uut.for_metaflow_run("base", 123)

        segment1 = base.push_segment("segment1")
        segment2 = base.push_segment("segment2")

    assert base.directory() == "/test/"
    assert segment1.directory() == "/test/segment1/"
    assert segment2.directory() == "/test/segment2/"


@patch("flows.utilities.working_dir.mkdtemp")
def test_push_partitions(mkdtemp_mock: MagicMock) -> None:
    with patch.object(Path, "mkdir"):
        mkdtemp_mock.return_value = "/test"
        base = uut.for_metaflow_run("base", 123)

        partition1 = base.push_partitions(("key1", "value1"))
        partition2 = base.push_partitions(("key1", "value1"), ("key2", "value2"))
        partition3 = partition1.push_partitions(("key3", "value3"))

    assert base.directory("misc") == "/test/misc/"
    assert (
        partition1.file("partition1.parquet") == "/test/key1=value1/partition1.parquet"
    )
    assert (
        partition2.file("partition2.parquet")
        == "/test/key1=value1/key2=value2/partition2.parquet"
    )
    assert (
        partition3.file("partition3.parquet")
        == "/test/key1=value1/key3=value3/partition3.parquet"
    )


@patch("flows.utilities.working_dir.mkdtemp")
def test_glob_partitions(mkdtemp_mock: MagicMock) -> None:
    with patch.object(Path, "mkdir"):
        mkdtemp_mock.return_value = "/test"
        base = uut.for_metaflow_run("base", 123)

        partition1 = base.glob_partitions("key1")
        partition2 = base.glob_partitions("key1", "key2")
        partition3 = partition1.push_partitions(("key3", "value3"))
        partition4 = partition3.glob_partitions("key4")

    assert partition1.directory("misc") == "/test/key1=*/misc/"
    assert partition2.directory() == "/test/key1=*/key2=*/"
    assert partition3.directory() == "/test/key1=*/key3=value3/"
    assert partition4.directory() == "/test/key1=*/key3=value3/key4=*/"
