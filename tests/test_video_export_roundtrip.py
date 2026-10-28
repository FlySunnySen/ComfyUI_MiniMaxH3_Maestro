"""read_mp4_frames: decode an mp4 back to float RGB frames."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[1]
for _p in (str(_PLUGIN_ROOT.parent), str(_PLUGIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comfy.options

comfy.options.enable_args_parsing()
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")

from ComfyUI_MiniMaxH3_Director.lib.video_export import _ffmpeg_bin, read_mp4_frames, write_frames_to_mp4

_FFMPEG = _ffmpeg_bin()


@unittest.skipUnless(_FFMPEG, "ffmpeg required")
class ReadMp4FramesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "clip.mp4"

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _gradient(n, h, w):
        # Gray ramp avoids chroma-subsampling loss; time axis varies too.
        base = torch.linspace(0.2, 0.8, n).view(n, 1, 1, 1).expand(n, h, w, 3)
        return base.contiguous()

    def test_roundtrip_shape_and_values(self):
        n, h, w = 6, 16, 16
        src = self._gradient(n, h, w)
        write_frames_to_mp4(self.path, src, fps=8.0)
        out = read_mp4_frames(self.path, width=w, height=h)
        self.assertEqual(tuple(out.shape), (n, h, w, 3))
        mae = (out - src).abs().mean().item()
        self.assertLess(mae, 0.03, f"decoded frames deviate too much (mae={mae:.4f})")

    def test_scales_when_requested_size_differs(self):
        src = self._gradient(4, 16, 16)
        write_frames_to_mp4(self.path, src, fps=8.0)
        out = read_mp4_frames(self.path, width=8, height=8)
        self.assertEqual(tuple(out.shape), (4, 8, 8, 3))

    def test_missing_file_raises(self):
        with self.assertRaises(Exception):
            read_mp4_frames(self.path, width=16, height=16)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
