"""Offload finished segments' export pixels to disk during 「全部导出」 runs.

The all-export merge needs every chunk in RAM only at the final concat. Keeping
them resident from the moment a segment finishes pushes the FaceRefine window
of the next segment past the cgroup line (2026-10-28 crash: seg 3 died in the
face-refine VAE decode with ~80GB RSS while seg 1+2 pixels sat unused). This
module moves those pixels to <output>/minimax_director_offload/<stamp>/ and
streams them back right before the merge.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.export_offload")


def new_offload_dir(plan) -> Path | None:
    """Create ``minimax_director_offload/<YYYYMMDD_HHMMSS>/`` for one execute.

    Only active in all-export mode (segments mode keeps pixels in RAM and
    releases them per-segment instead). Stale run dirs are removed — offloaded
    pixels never outlive the execute that wrote them.
    """
    if getattr(plan, "export_mode", "all") != "all":
        return None
    try:
        import folder_paths

        base = Path(folder_paths.get_output_directory()) / "minimax_director_offload"
        base.mkdir(parents=True, exist_ok=True)
        for stale in base.iterdir():
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = base / stamp
        if root.exists():
            for i in range(1, 1000):
                candidate = base / f"{stamp}_{i:03d}"
                if not candidate.exists():
                    root = candidate
                    break
        root.mkdir(parents=True, exist_ok=False)
        log.info("MiniMax H3 Director export offload dir: %s", root)
        return root
    except OSError as exc:
        log.warning("Export offload dir unavailable (%s); keeping pixels in RAM.", exc)
        return None


class InsufficientDiskSpace(RuntimeError):
    pass


def _slots_bytes(slots: dict[str, Any]) -> int:
    total = 0
    for v in slots.values():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
        elif isinstance(v, list):
            total += sum(
                t.numel() * t.element_size()
                for _, t in v
                if torch.is_tensor(t)
            )
    return total


def offload_slots(root: Path, seg_index: int, slots: dict[str, Any]) -> Path:
    need = _slots_bytes(slots)
    free = shutil.disk_usage(root).free
    if free < need * 1.3 + 1024**3:
        raise InsufficientDiskSpace(
            f"offload seg #{seg_index + 1} needs ~{need / 1024**3:.1f}GB, "
            f"only {free / 1024**3:.1f}GB free on {root}"
        )
    path = root / f"seg_{int(seg_index):04d}.pt"
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(slots, tmp)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def has_offload(root: Path | None, seg_index: int) -> bool:
    return root is not None and (root / f"seg_{int(seg_index):04d}.pt").exists()


def load_offloaded_slots(root: Path, seg_index: int) -> dict[str, Any]:
    return torch.load(
        root / f"seg_{int(seg_index):04d}.pt", map_location="cpu", weights_only=True
    )


def remove_offload_dir(root: Path | None) -> None:
    if root is None:
        return
    shutil.rmtree(root, ignore_errors=True)
