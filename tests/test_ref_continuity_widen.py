"""P0 rv2v/v2v reference widening: start_extra lookahead + source_clip fallback."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

# ``director`` uses ``..lib`` relative imports, so the plugin must be imported
# as ``ComfyUI_MiniMaxH3_Director.*`` with its parent dir on sys.path. The
# plugin root __init__ pulls in comfy — run with ComfyUI on PYTHONPATH.
_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[1]
for _p in (str(_PLUGIN_ROOT.parent), str(_PLUGIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# comfy's model_management probes CUDA at import time; force CPU so tests
# never need (or race with) the GPU ComfyUI is running on. comfy ignores argv
# unless args parsing is enabled first.
import comfy.options

comfy.options.enable_args_parsing()
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")

from ComfyUI_MiniMaxH3_Director.director.segment_runtime import (
    resolve_segment_raw_clip,
    resolve_segment_raw_clip_with_lookahead,
)


def make_plan(total: int, mode: str = "gen_image"):
    source_video = torch.arange(total * 3 * 8 * 8, dtype=torch.float32)
    source_video = source_video.reshape(total, 8, 8, 3)
    source_video[:] = torch.arange(total, dtype=torch.float32).view(total, 1, 1, 1)
    return SimpleNamespace(raw={"timelineMode": mode}, source_video=source_video)


def make_seg(start: int, end: int, *, source_clip=None, task_key="rv2v"):
    return SimpleNamespace(
        start_frame=start,
        end_frame=end,
        task_key=task_key,
        source_clip=source_clip,
    )


def frame_id(frames: torch.Tensor) -> torch.Tensor:
    """Each synthetic frame is constant and equals its own index."""
    return frames[:, 0, 0, 0]


class LookbehindTests(unittest.TestCase):
    def test_no_extra_matches_base_resolver(self):
        plan = make_plan(200)
        seg = make_seg(150, 200)
        base = resolve_segment_raw_clip(plan, seg)
        look = resolve_segment_raw_clip_with_lookahead(plan, seg)
        self.assertTrue(torch.equal(base, look))

    def test_start_extra_pulls_earlier_frames(self):
        plan = make_plan(200)
        seg = make_seg(150, 200)
        ext = resolve_segment_raw_clip_with_lookahead(plan, seg, start_extra=22)
        self.assertEqual(int(ext.shape[0]), 72)
        self.assertEqual(int(frame_id(ext)[0]), 128)
        self.assertEqual(int(frame_id(ext)[-1]), 199)

    def test_start_extra_clamps_at_zero(self):
        plan = make_plan(200)
        seg = make_seg(10, 60)
        ext = resolve_segment_raw_clip_with_lookahead(plan, seg, start_extra=22)
        self.assertEqual(int(ext.shape[0]), 60)
        self.assertEqual(int(frame_id(ext)[0]), 0)

    def test_start_and_end_extra(self):
        plan = make_plan(200)
        seg = make_seg(150, 180)
        ext = resolve_segment_raw_clip_with_lookahead(
            plan, seg, end_extra=8, start_extra=22
        )
        self.assertEqual(int(ext.shape[0]), 60)
        self.assertEqual(int(frame_id(ext)[0]), 128)
        self.assertEqual(int(frame_id(ext)[-1]), 187)

    def test_end_extra_clamps_at_total(self):
        plan = make_plan(200)
        seg = make_seg(150, 200)
        ext = resolve_segment_raw_clip_with_lookahead(plan, seg, end_extra=50)
        self.assertEqual(int(ext.shape[0]), 50)

    def test_source_clip_has_no_global_context(self):
        clip = torch.zeros(50, 8, 8, 3)
        plan = make_plan(200)
        seg = make_seg(150, 200, source_clip=clip)
        ext = resolve_segment_raw_clip_with_lookahead(plan, seg, start_extra=22)
        self.assertEqual(int(ext.shape[0]), 50)
        self.assertTrue(torch.equal(ext, clip))

    def test_non_gen_timeline_passes_widened_range_to_loader(self):
        plan = SimpleNamespace(raw={"timelineMode": "video"}, source_video=None)
        seg = make_seg(150, 180, task_key="v2v")
        decoded = torch.zeros(52, 8, 8, 3)
        with mock.patch(
            "ComfyUI_MiniMaxH3_Director.lib.video_io.logical_frame_count",
            return_value=200,
        ), mock.patch(
            "ComfyUI_MiniMaxH3_Director.director.segment_runtime.load_timeline_segment",
            return_value=decoded,
        ) as loader:
            ext = resolve_segment_raw_clip_with_lookahead(plan, seg, start_extra=22)
        loader.assert_called_once_with(plan.raw, 128, 180)
        self.assertTrue(torch.equal(ext, decoded))

    def test_non_gen_timeline_clamps_before_zero(self):
        plan = SimpleNamespace(raw={"timelineMode": "video"}, source_video=None)
        seg = make_seg(10, 60, task_key="v2v")
        decoded = torch.zeros(60, 8, 8, 3)
        with mock.patch(
            "ComfyUI_MiniMaxH3_Director.lib.video_io.logical_frame_count",
            return_value=200,
        ), mock.patch(
            "ComfyUI_MiniMaxH3_Director.director.segment_runtime.load_timeline_segment",
            return_value=decoded,
        ) as loader:
            resolve_segment_raw_clip_with_lookahead(plan, seg, start_extra=22)
        loader.assert_called_once_with(plan.raw, 0, 60)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
