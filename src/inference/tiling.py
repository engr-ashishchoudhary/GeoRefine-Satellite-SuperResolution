"""
Full-scene inference tiling for GeoRefine.
Distinct from src/dataset/patches.py's compute_patch_grid(), which is for
*training* patch extraction and deliberately drops any remainder that does
not fit a full patch. Inference tiling has a different requirement: every
pixel of the input scene must appear in the output, so edge tiles here are
clipped to fit within the scene bounds rather than dropped.
Adjacent tiles overlap by `overlap` pixels so that blend_weight_mask() can
taper each tile's contribution to zero across that overlap region, avoiding
visible seams where tile outputs are combined (a standard "overlap-add"
approach, matching how the reference Real-ESRGAN inference script and most
tiled super-resolution pipelines handle scenes larger than a single forward
pass).
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
def compute_tile_grid(
    height: int, width: int, tile_size: int, overlap: int
) -> List[Tuple[int, int, int, int]]:
    """Return (row, col, tile_h, tile_w) tuples covering the full [0,height) x [0,width) extent.
    Tiles are spaced by (tile_size - overlap) so adjacent tiles overlap by
    `overlap` pixels. The last tile in each row/column is clipped (its
    tile_h or tile_w may be smaller than tile_size) so every pixel is
    covered - no remainder is ever dropped, unlike training-patch extraction.
    If the scene is smaller than tile_size in a dimension, a single tile
    covering that whole dimension is returned - tiling is a safety net for
    large scenes, not forced fragmentation of small ones.
    """
    if tile_size <= 0 or height <= 0 or width <= 0:
        return []
    step = max(tile_size - overlap, 1)
    def _starts(total: int) -> List[int]:
        if total <= tile_size:
            return [0]
        starts = list(range(0, total - tile_size + 1, step))
        if starts[-1] + tile_size < total:
            starts.append(total - tile_size)
        return starts
    row_starts = _starts(height)
    col_starts = _starts(width)
    tiles: List[Tuple[int, int, int, int]] = []
    for row in row_starts:
        tile_h = min(tile_size, height - row)
        for col in col_starts:
            tile_w = min(tile_size, width - col)
            tiles.append((row, col, tile_h, tile_w))
    return tiles
def blend_weight_mask(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """A (tile_h, tile_w) weight mask that linearly tapers to a small minimum
    across the first/last `overlap` pixels of each axis, and is 1.0 elsewhere.
    Used to weight each tile's contribution when accumulating into the full
    output canvas, so overlapping regions blend smoothly instead of showing
    a hard seam at the tile boundary. A small floor (not exactly 0) is used
    at the very edge to avoid a fully-zero-weight pixel in degenerate cases
    (e.g. a 1-pixel-wide tile), which would otherwise divide by zero during
    the accumulate/normalize step in scene_inference.py.
    """
    overlap = max(0, min(overlap, tile_h // 2, tile_w // 2))
    floor = 1e-3
    def _axis_weights(length: int) -> np.ndarray:
        weights = np.ones(length, dtype=np.float64)
        if overlap > 0:
            ramp = np.linspace(floor, 1.0, overlap, endpoint=False)
            weights[:overlap] = ramp
            weights[-overlap:] = ramp[::-1]
        return weights
    row_weights = _axis_weights(tile_h)
    col_weights = _axis_weights(tile_w)
    return np.outer(row_weights, col_weights)
