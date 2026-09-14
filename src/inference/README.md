# src/inference - Phase 5: Super-Resolution Inference
Produces full-scene SR output, distinct from Phase 3's patch generation
(which is for training-data preparation only).
## Tiling
A full Sentinel-2 scene can be far larger than a single forward pass
through the pretrained model should attempt on a normal laptop. This
module tiles the scene (config.yaml's `inference.tile_size` /
`tile_overlap`), runs super-resolution per tile, and combines tiles back
into the full-resolution output using overlap-add blending (see
tiling.py's blend_weight_mask()) so tile boundaries do not produce visible
seams.
If a scene is smaller than `tile_size`, it is processed as a single tile -
tiling is a safety net for large scenes, not forced fragmentation of small
ones.
## Output contract
`run_demo_output()` selects one scene (config.yaml's `inference.scene_id`,
or the first available scene) and writes it to the project's stable
demo_data/ contract, established in Phase 0:
    demo_data/input/scene.tif   - the scene's real LR band stack, unmodified
    demo_data/sr/scene_sr.tif   - the super-resolved output
`run_scene_inference()` is the general-purpose function for producing an
SR GeoTIFF at any output path, for any scene - not tied to the demo_data
contract. This is the function an eventual dashboard "live mode" or any
future batch-processing script should call directly.
## CLI
    python scripts/run_inference.py
Requires `data/processed/dataset_index.json` (Phase 1) and
`models/RealESRGAN_x4plus.pth` (see scripts/download_pretrained_model.py,
Phase 4).
## Known limitations
- Inference quality is not evaluated here - Phase 6 (Validation & Metrics)
  computes PSNR/SSIM/RMSE/SAM against real or synthetic reference data.
- Per-band pseudo-RGB processing (see src/model/README.md) applies per
  tile, same documented limitation as Phase 4.
- Only exercised against synthetic test GeoTIFFs so far - no real
  Sentinel-2 imagery has been run through full-scene inference yet.
