"""Export pixel offload: round-trip fidelity and RAM-slot release."""
import sys
import tempfile
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

import comfy.options

comfy.options.enable_args_parsing()
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")

from ComfyUI_MiniMaxH3_Director.director import export_offload
from ComfyUI_MiniMaxH3_Director.director.export_offload import (
    load_offloaded_slots,
    new_offload_dir,
    offload_slots,
    remove_offload_dir,
)


def make_chunk(n, h=8, w=8, fill=1.0):
    return torch.full((n, h, w, 3), fill)


class OffloadRoundTripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_tensors_identical_and_shared(self):
        chunk = make_chunk(5, fill=0.25)
        pre = make_chunk(7, h=4, w=4, fill=0.5)
        slots = {"outputs": chunk, "pre_refine": pre, "pre_face": chunk, "refine_passes": []}
        offload_slots(self.root, 2, slots)
        loaded = load_offloaded_slots(self.root, 2)
        self.assertTrue(torch.equal(loaded["outputs"], chunk))
        self.assertTrue(torch.equal(loaded["pre_refine"], pre))
        # outputs/pre_face were the same object pre-save; pickle memo keeps that.
        self.assertIs(loaded["outputs"], loaded["pre_face"])

    def test_roundtrip_refine_passes(self):
        passes = [("p1", make_chunk(3, fill=0.75))]
        offload_slots(self.root, 0, {"outputs": make_chunk(2), "refine_passes": passes})
        loaded = load_offloaded_slots(self.root, 0)
        self.assertEqual([s for s, _ in loaded["refine_passes"]], ["p1"])
        self.assertTrue(torch.equal(loaded["refine_passes"][0][1], make_chunk(3, fill=0.75)))

    def test_remove_dir(self):
        offload_slots(self.root, 1, {"outputs": make_chunk(2)})
        remove_offload_dir(self.root)
        self.assertFalse(self.root.exists())
        # Idempotent.
        remove_offload_dir(self.root)
        remove_offload_dir(None)

    def test_new_offload_dir_skips_segments_mode(self):
        plan = SimpleNamespace(export_mode="segments")
        self.assertIsNone(new_offload_dir(plan))

    def test_new_offload_dir_all_mode_uses_output_root_and_cleans_stale(self):
        import folder_paths

        plan = SimpleNamespace(export_mode="all")
        stale = Path(folder_paths.get_output_directory()) / "minimax_director_offload"
        stale.mkdir(parents=True, exist_ok=True)
        old = stale / "20000101_000000"
        old.mkdir(exist_ok=True)
        root = new_offload_dir(plan)
        try:
            self.assertIsNotNone(root)
            self.assertEqual(root.parent.name, "minimax_director_offload")
            self.assertTrue(root.exists())
            self.assertFalse(old.exists(), "stale run dirs must be cleaned")
        finally:
            remove_offload_dir(root)


class OffloadFinishedSegmentPixelsTest(unittest.TestCase):
    def test_offload_pops_dicts_posters_slots(self):
        from ComfyUI_MiniMaxH3_Director.director.executor_core import (
            _offload_finished_segment_pixels,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = make_chunk(6, fill=0.5)
            pre = make_chunk(6, h=4, w=4, fill=0.25)
            completed_outputs = {1: chunk}
            completed_pre_refine = {1: pre}
            completed_pre_face = {1: chunk}
            completed_refine_passes = {1: []}
            seg = SimpleNamespace(index=1)
            segment_outputs = [make_chunk(1), chunk]
            output_chunks = [make_chunk(1), chunk]
            output_pre_chunks = [make_chunk(1), pre]
            output_pre_face_chunks = [make_chunk(1), chunk]
            output_segments = [SimpleNamespace(index=0), seg]
            progress_pos = {1: 1}

            stored = _offload_finished_segment_pixels(
                1,
                root,
                completed_outputs=completed_outputs,
                completed_pre_refine=completed_pre_refine,
                completed_pre_face=completed_pre_face,
                completed_refine_passes=completed_refine_passes,
                segment_outputs=segment_outputs,
                segment_pre_refine=[make_chunk(1), pre],
                segment_pre_face=None,
                progress_pos=progress_pos,
                output_chunks=output_chunks,
                output_pre_chunks=output_pre_chunks,
                output_pre_face_chunks=output_pre_face_chunks,
                output_segments=output_segments,
            )

            self.assertGreater(stored, 0)
            self.assertNotIn(1, completed_outputs)
            self.assertNotIn(1, completed_pre_refine)
            self.assertNotIn(1, completed_pre_face)
            self.assertNotIn(1, completed_refine_passes)
            # Run lists and all-mode output lists carry 1-frame posters now.
            self.assertEqual(int(segment_outputs[1].shape[0]), 1)
            self.assertEqual(int(output_chunks[1].shape[0]), 1)
            self.assertEqual(int(output_pre_chunks[1].shape[0]), 1)
            self.assertEqual(int(output_pre_face_chunks[1].shape[0]), 1)
            # Disk copy survives for the merge reload.
            loaded = load_offloaded_slots(root, 1)
            self.assertTrue(torch.equal(loaded["outputs"], chunk))
            self.assertTrue(torch.equal(loaded["pre_refine"], pre))

    def test_no_dir_is_noop(self):
        from ComfyUI_MiniMaxH3_Director.director.executor_core import (
            _offload_finished_segment_pixels,
        )

        completed_outputs = {1: make_chunk(4)}
        stored = _offload_finished_segment_pixels(
            1,
            None,
            completed_outputs=completed_outputs,
            completed_pre_refine={},
            completed_pre_face={},
            completed_refine_passes={},
            segment_outputs=[],
            segment_pre_refine=[],
            segment_pre_face=None,
            progress_pos={},
        )
        self.assertEqual(stored, 0.0)
        self.assertIn(1, completed_outputs)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
