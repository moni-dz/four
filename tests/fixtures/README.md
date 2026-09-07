# Decoder fixtures

Small inputs whose decoded pixels are pinned by `tests/decode_golden.rs`. Each is 64x48 and encodes
the same synthetic pattern: a horizontal red ramp, a vertical green ramp, an eight-pixel blue
checkerboard, and a saturated magenta patch that stresses chroma subsampling. `rgba8.png` adds a
horizontal alpha ramp.

Keep them small. The truncation test decodes every prefix of every fixture, so its cost grows with
the square of the fixture size.

## Regenerating

The pattern sources are not checked in; recreate them with the script in the history of this file's
introducing commit, or write any deterministic 64x48 pattern and re-record. Then:

```sh
ffmpeg -y -i pattern.ppm -c:v mjpeg -q:v 2 -pix_fmt yuvj420p baseline.jpg
ffmpeg -y -i pattern.ppm -c:v png -pix_fmt rgb24 rgb8.png
ffmpeg -y -i pattern.ppm -c:v png -pix_fmt rgb48be rgb16.png
ffmpeg -y -i pattern_alpha.pam -c:v png -pix_fmt rgba rgba8.png
ffmpeg -y -i pattern.ppm -vf palettegen=max_colors=64 palette.png
ffmpeg -y -i pattern.ppm -i palette.png -lavfi paletteuse -c:v png -pix_fmt pal8 palette8.png
ffmpeg -y -i pattern.ppm -c:v tiff -pix_fmt rgb24 rgb8.tiff
ffmpeg -y -i pattern.ppm -c:v libjxl -pix_fmt rgb24 rgb8.jxl
ffmpeg -y -framerate 5 -i frame%d.ppm -loop 0 animated.gif
```

Re-record the goldens with `FOUR_RECORD_GOLDEN=1 cargo test --test decode_golden -- --nocapture`
and paste the printed constants into `GOLDENS`. Regenerating fixtures changes every hash, so do it
only when adding coverage, never to make a failing test pass.

## Benchmark-sized fixtures

Three fixtures are larger than the rest, because the code they exercise does not run at 64x48:

- `busy.jpg` (384x384) has high-frequency content in every block, so no block takes the DC-only
  path that skips the inverse transform.
- `large.tiff` (640x480) is above `PARALLEL_PIXELS_MIN`, so it reaches the parallel normalization
  branch. Its coarse vertical banding keeps it compressible, so the file stays small.
- `screenshot.jxr` (3840x2160, ~12 MB) is a real Windows HDR screenshot, not synthetic — JPEG XR
  has no encoder anywhere in this toolchain (see "Missing coverage" below), so there is no small
  pattern to regenerate it from. It is exercised by `decodes_real_bgr101010_sample_pixels` in
  `crates/jpegxr/src/decode.rs` and by the `jpeg_xr_bgr101010` entry in `benches/formats.rs`, but is
  deliberately **not** added to `GOLDENS` or the truncation fuzzer in `tests/decode_golden.rs`: at
  12 MB it is ~450x the largest fixture those loops were sized for, and the truncation test's
  sampled offsets each run a real decode attempt, so its cost is not bounded the way it is for the
  KB-scale fixtures above.

The truncation test samples offsets rather than walking every one once a fixture exceeds
`EXHAUSTIVE_PREFIX` bytes, so adding a large fixture to `GOLDENS` does not make the suite quadratic
in fixture *count* — but each sampled offset is still one real decode attempt, so a fixture that is
orders of magnitude larger than the others still dominates the loop's wall time.

## Missing coverage

Two formats have no fixture here because nothing in this repository or in a normal toolchain can
encode them:

- **Progressive JPEG.** ffmpeg's `mjpeg` encoder emits baseline only.
- **Arithmetic-coded JPEG.** This is the reason `src/image_formats/jpeg/` exists at all — zune-jpeg
  does not implement arithmetic entropy coding. No Rust encoder produces it; `jpegtran -arithmetic`
  from libjpeg-turbo is the usual source.

Until those land, the code paths that handle them are covered only by unit tests. JPEG XR's
`BGR101010` profile is now covered by `screenshot.jxr` above; its `RGBA128Float` profile still has
no fixture and is covered only by the `#[ignore]`d `decodes_real_sample_pixels` test in
`crates/jpegxr/src/decode.rs`, which reads a local file named by the `JPEGXR_SAMPLE` environment
variable.

