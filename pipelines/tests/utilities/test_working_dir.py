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

    assert base.path == Path("/test/")
    assert segment1.path == Path("/test/segment1/")
    assert segment2.path == Path("/test/segment2/")
