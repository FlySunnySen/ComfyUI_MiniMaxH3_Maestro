"""Segments-mode automatic mp4 merge: stream-copy concat of per-segment files."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

# ``director``/``lib`` use relative imports, so the plugin must be imported as
# ``ComfyUI_MiniMaxH3_Director.*`` with its parent dir on sys.path.
_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[1]
for _p in (str(_PLUGIN_ROOT.parent), str(_PLUGIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comfy.options

comfy.options.enable_args_parsing()
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")

from ComfyUI_MiniMaxH3_Director.lib.video_export import _ffmpeg_bin, write_frames_to_mp4
from ComfyUI_MiniMaxH3_Director.director.segment_mp4_export import merge_segment_mp4s

_FFMPEG = _ffmpeg_bin()


def make_plan(mode: str = "segments"):
    return SimpleNamespace(export_mode=mode)


def write_part(run_dir: Path, index: int, frames: int = 6, audio=None):
    from ComfyUI_MiniMaxH3_Director.director.segment_mp4_export import segment_mp4_path

    seg = SimpleNamespace(index=index)
    tensor = torch.full((frames, 16, 16, 3), float(index + 1) / 4.0)
    return write_frames_to_mp4(
        segment_mp4_path(run_dir, seg), tensor, fps=8.0, audio=audio
    )


@unittest.skipUnless(_FFMPEG, "ffmpeg required")
class MergeSegmentMp4sTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_merges_two_parts_in_index_order(self):
        write_part(self.run_dir, 1, frames=8)
        write_part(self.run_dir, 0, frames=6)
        merged = merge_segment_mp4s(self.run_dir, make_plan())
        self.assertIsNotNone(merged)
        merged_path = Path(merged)
        self.assertEqual(merged_path.name, "merged.mp4")
        self.assertGreater(merged_path.stat().st_size, 0)
        # Both source segments are kept.
        self.assertTrue((self.run_dir / "seg_0000.mp4").is_file())
        self.assertTrue((self.run_dir / "seg_0001.mp4").is_file())

    def test_merge_with_audio_tracks(self):
        audio = {
            "waveform": torch.zeros(1, 2, 8000),
            "sample_rate": 8000,
        }
        write_part(self.run_dir, 0, frames=8, audio=audio)
        write_part(self.run_dir, 1, frames=8, audio=audio)
        merged = merge_segment_mp4s(self.run_dir, make_plan())
        self.assertIsNotNone(merged)
        self.assertGreater(Path(merged).stat().st_size, 0)

    def test_fewer_than_two_parts_returns_none(self):
        write_part(self.run_dir, 0)
        self.assertIsNone(merge_segment_mp4s(self.run_dir, make_plan()))
        self.assertIsNone(merge_segment_mp4s(self.run_dir / "empty", make_plan()))

    def test_all_mode_returns_none(self):
        write_part(self.run_dir, 0)
        write_part(self.run_dir, 1)
        self.assertIsNone(merge_segment_mp4s(self.run_dir, make_plan(mode="all")))

    def test_ffmpeg_failure_cleans_up_and_returns_none(self):
        write_part(self.run_dir, 0)
        write_part(self.run_dir, 1)
        with mock.patch(
            "ComfyUI_MiniMaxH3_Director.director.segment_mp4_export.subprocess.run",
            return_value=mock.Mock(returncode=1, stderr=b"boom"),
        ):
            merged = merge_segment_mp4s(self.run_dir, make_plan())
        self.assertIsNone(merged)
        self.assertFalse((self.run_dir / "merged.mp4").exists())
        # Source segments untouched.
        self.assertTrue((self.run_dir / "seg_0000.mp4").is_file())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
