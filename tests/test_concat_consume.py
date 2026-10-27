"""consume=True concat: identical output, chunk slots freed as they land."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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

from ComfyUI_MiniMaxH3_Director.director.segment_continuity import (
    _streaming_cat_frames,
    concat_continuous_chunks,
)


def make_plan(continuity: bool):
    return SimpleNamespace(continuity_enabled=continuity)


def make_chunks(*lengths, h=8, w=8):
    out = []
    for i, n in enumerate(lengths):
        c = torch.zeros(n, h, w, 3)
        c[:, :, :, :] = float(i) + 1.0
        out.append(c)
    return out


class ConcatConsumeTest(unittest.TestCase):
    def test_consume_matches_plain_multi_chunk(self):
        plan = make_plan(continuity=True)
        a = make_chunks(4, 5, 6)
        plain = concat_continuous_chunks(a, [], plan)
        b = make_chunks(4, 5, 6)
        consumed = concat_continuous_chunks(b, [], plan, consume=True)
        self.assertEqual(plain.shape, consumed.shape)
        self.assertTrue(torch.equal(plain, consumed))
        # All slots freed.
        self.assertTrue(all(c is None for c in b))

    def test_consume_matches_plain_no_continuity(self):
        plan = make_plan(continuity=False)
        a = make_chunks(3, 4)
        plain = concat_continuous_chunks(a, [], plan)
        b = make_chunks(3, 4)
        consumed = concat_continuous_chunks(b, [], plan, consume=True)
        self.assertTrue(torch.equal(plain, consumed))
        self.assertTrue(all(c is None for c in b))

    def test_consume_single_chunk(self):
        plan = make_plan(continuity=True)
        b = make_chunks(5)
        out = concat_continuous_chunks(b, [], plan, consume=True)
        self.assertEqual(int(out.shape[0]), 5)
        self.assertTrue(torch.equal(out, b[0]))

    def test_streaming_cat_mixed_shapes_pads(self):
        a = [torch.full((3, 4, 4, 3), 1.0), torch.full((2, 6, 6, 3), 2.0)]
        out = _streaming_cat_frames(a)
        self.assertEqual(tuple(out.shape), (5, 6, 6, 3))
        # Center-pad on both axes: a 1-px 0.5 frame around the 4x4 content.
        self.assertTrue(torch.equal(out[:3, 0], torch.full((3, 6, 3), 0.5)))
        self.assertTrue(torch.equal(out[:3, 5], torch.full((3, 6, 3), 0.5)))
        self.assertTrue(torch.equal(out[:3, 1:5, 0], torch.full((3, 4, 3), 0.5)))
        self.assertTrue(torch.equal(out[:3, 1:5, 5], torch.full((3, 4, 3), 0.5)))
        self.assertTrue(torch.equal(out[:3, 1:5, 1:5], torch.full((3, 4, 4, 3), 1.0)))
        self.assertTrue(torch.equal(out[3:], torch.full((2, 6, 6, 3), 2.0)))

    def test_consume_after_seam_fix_keeps_values_close(self):
        # With continuity on, seam fix blends edges — consume path must take
        # the same fix path (values identical to the non-consume run).
        plan = make_plan(continuity=True)
        a = make_chunks(4, 4, 4)
        plain = concat_continuous_chunks(a, [], plan)
        b = make_chunks(4, 4, 4)
        consumed = concat_continuous_chunks(b, [], plan, consume=True)
        self.assertTrue(torch.equal(plain, consumed))
        self.assertTrue(all(c is None for c in b))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
