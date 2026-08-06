# 4x Upscaling Support — Design

**Issue:** [#1 support 4x upscaling](https://github.com/e-dream-ai/gpu-container-uprez/issues/1)
**Date:** 2026-08-02

## Problem

The container only supports 2x Real-ESRGAN upscaling. Every layer hard-codes this:
the Dockerfile downloads only `RealESRGAN_x2plus.pth`, `ModelLoader` builds
`RRDBNet(scale=2)` / `RealESRGANer(scale=2)`, and both `InputValidator` and the
handler's `INPUT_SCHEMA` reject any `upscale_factor` outside `[1, 2]`.

## Goal

Support `upscale_factor: 4` end-to-end, backed by `RealESRGAN_x4plus.pth`, while
keeping existing 1x (skip) and 2x behaviour unchanged. Supported set becomes
`{1, 2, 4}` — each backed by its native model (no arbitrary/outscale resizing).

## Changes by layer

### 1. Dockerfile

Download `RealESRGAN_x4plus.pth` into `/opt/models/realesrgan/` alongside the
existing x2plus weight:

```
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

### 2. `src/services/model_loader.py`

- Replace the single `'realesrgan'` path entry with a factor→file map:
  `{2: RealESRGAN_x2plus.pth, 4: RealESRGAN_x4plus.pth}`.
- `load_realesrgan_model(upscale_factor)` selects the weight file and builds
  `RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=factor)`
  and `RealESRGANer(scale=factor, ...)`. x2plus and x4plus share the same RRDBNet
  architecture — only `scale` and the weight file differ.
- Cache key stays `realesrgan_x{factor}` so 2x and 4x models can coexist in memory.
- Unsupported factors continue to fall back to 2x (defensive; validation already
  blocks them upstream).

### 3. `src/services/upscaler_service.py`

- Widen the two `!= 2 → fall back to 2` guards (in `upscale_frames` and
  `upscale_single_frame`) to accept `{2, 4}`.
- The batch path is already scale-generic: it crops output via
  `netscale = upsampler.scale` and pads to `_MOD_SCALE` (stays `2`). The x4plus
  RRDBNet has no input-divisibility constraint, so padding to an even size is
  harmless for 4x.
- `upscale_single_frame` uses `enhance(outscale=factor)`, which is already generic.

### 4. `src/utils/input_validator.py`

- `validate_upscale_factor`: accept `[1, 2, 4]`; update the error message.
- Add a warning for 4x about increased memory/processing time (mirrors the
  existing interpolation-factor warning pattern).

### 5. `src/handler.py`

- `INPUT_SCHEMA['upscale_factor']` constraint `x in [1, 2]` → `x in [1, 2, 4]`.
- Update the description string to mention 4x.

### 6. `README.md`

- Document `upscale_factor: 4`; list both models under "Models Used".

## Testing

No unit tests exist today (only the GPU-dependent `build-and-test.sh` integration
script). Add a focused `unittest` for the pure `InputValidator.validate_upscale_factor`:

- accepts `1`, `2`, `4` (and returns the coerced value)
- rejects `3`, `0`, `8`, and non-integer input

Written test-first (red → green). GPU-dependent model loading and the full
pipeline remain covered by the existing integration script on a GPU box.

## Out of scope

- 3x or arbitrary `outscale` resizing (adds a slower non-batch path for a factor
  nobody requested).
- Tile-size default tuning for 4x.
