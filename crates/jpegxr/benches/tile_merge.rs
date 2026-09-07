//! Compares two strategies for merging independently-decoded tiles into a shared image buffer:
//!
//! - `merge_via_local_buffer_then_copy`: decode each tile into its own heap buffer (in
//!   parallel), then `copy_from_slice` every row into the shared buffer afterward. This was
//!   `decode_tiles_into`'s original approach.
//! - `merge_via_direct_shared_write`: decode each tile directly into its (proven-disjoint)
//!   region of the shared buffer via a raw pointer, skipping the local buffer and the copy. This
//!   measurably won (see the doc comment on `decode_tiles_into`'s `TileWriter`), and is what
//!   `src/decode.rs` now does; kept here as the record of that decision.
//!
//! Both use a synthetic per-macroblock "decode" step (cheap, data-dependent arithmetic) instead
//! of real entropy decoding, so this isolates the cost of the merge strategy itself rather than
//! real decode work. Run with `cargo bench -p jpegxr --bench tile_merge`.
//!
//! Exploratory/measurement-only code: lint noise that would matter in shipped decoder code (raw
//! casts, unsafe without per-site `#[expect]`) isn't worth polishing here.
#![allow(
    unsafe_code,
    clippy::pedantic,
    reason = "exploratory measurement code, not shipped decoder logic; see the module doc comment"
)]

use std::hint::black_box;

use divan::Bencher;
use rayon::prelude::*;

fn main() {
    check_merge_strategies_agree();
    divan::main();
}

/// Runs both merge strategies once and confirms they produce the same image, so a benchmark run
/// itself catches a divergence between them (the two closures above are not wired into
/// `cargo test`, since `harness = false` here hands `main` to `divan` instead of libtest).
fn check_merge_strategies_agree() {
    let tiles = tile_grid(MACROBLOCK_WIDTH, MACROBLOCK_HEIGHT, TILES_ACROSS, TILES_DOWN);

    let mut via_copy = vec![0_i32; MACROBLOCK_WIDTH * MACROBLOCK_HEIGHT * STRIDE];
    for tile in &tiles {
        let mut local = vec![0_i32; tile.width * tile.height * STRIDE];
        for local_y in 0..tile.height {
            for local_x in 0..tile.width {
                let seed = (tile.top + local_y) * MACROBLOCK_WIDTH + tile.left + local_x;
                let start = (local_y * tile.width + local_x) * STRIDE;
                for (i, slot) in local[start..start + STRIDE].iter_mut().enumerate() {
                    *slot = synthetic_decode_value(seed * STRIDE + i);
                }
            }
        }
        let row_len = tile.width * STRIDE;
        for local_y in 0..tile.height {
            let global_row = (tile.top + local_y) * MACROBLOCK_WIDTH + tile.left;
            via_copy[global_row * STRIDE..][..row_len]
                .copy_from_slice(&local[local_y * row_len..][..row_len]);
        }
    }

    let mut via_direct = vec![0_i32; MACROBLOCK_WIDTH * MACROBLOCK_HEIGHT * STRIDE];
    let base = SharedBase(via_direct.as_mut_ptr());
    for tile in &tiles {
        let base = base.0;
        for local_y in 0..tile.height {
            let global_row = (tile.top + local_y) * MACROBLOCK_WIDTH + tile.left;
            for local_x in 0..tile.width {
                let seed = (tile.top + local_y) * MACROBLOCK_WIDTH + tile.left + local_x;
                let start = (global_row + local_x) * STRIDE;
                // SAFETY: single-threaded here, and `start..start + STRIDE` is in-bounds for the
                // same reason given in `merge_via_direct_shared_write`.
                unsafe {
                    for i in 0..STRIDE {
                        *base.add(start + i) = synthetic_decode_value(seed * STRIDE + i);
                    }
                }
            }
        }
    }

    assert_eq!(via_copy, via_direct, "merge strategies disagree");
}

#[derive(Clone, Copy)]
struct Tile {
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}

fn tile_grid(
    macroblock_width: usize,
    macroblock_height: usize,
    tiles_across: usize,
    tiles_down: usize,
) -> Vec<Tile> {
    let tile_width = macroblock_width / tiles_across;
    let tile_height = macroblock_height / tiles_down;
    let mut tiles = Vec::with_capacity(tiles_across * tiles_down);
    for tile_y in 0..tiles_down {
        for tile_x in 0..tiles_across {
            tiles.push(Tile {
                left: tile_x * tile_width,
                top: tile_y * tile_height,
                width: tile_width,
                height: tile_height,
            });
        }
    }
    tiles
}

/// Stand-in for real per-macroblock decode work: cheap but data-dependent so it can't be
/// constant-folded away, comparable order of magnitude to a coefficient dequantize+predict step.
#[inline]
fn synthetic_decode_value(seed: usize) -> i32 {
    let mut x = seed as u32;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x as i32
}

const MACROBLOCK_WIDTH: usize = 128;
const MACROBLOCK_HEIGHT: usize = 128;
const STRIDE: usize = 48; // components * 16, matching the lowpass path's per-macroblock width
const TILES_ACROSS: usize = 4;
const TILES_DOWN: usize = 4;

#[divan::bench]
fn merge_via_local_buffer_then_copy(bencher: Bencher<'_, '_>) {
    let tiles = tile_grid(MACROBLOCK_WIDTH, MACROBLOCK_HEIGHT, TILES_ACROSS, TILES_DOWN);
    let values = vec![0_i32; MACROBLOCK_WIDTH * MACROBLOCK_HEIGHT * STRIDE];

    bencher.bench_local(|| {
        let mut values = values.clone();

        let tile_values: Vec<Vec<i32>> = tiles
            .par_iter()
            .map(|tile| {
                let mut local = vec![0_i32; tile.width * tile.height * STRIDE];
                for local_y in 0..tile.height {
                    for local_x in 0..tile.width {
                        let seed = (tile.top + local_y) * MACROBLOCK_WIDTH + tile.left + local_x;
                        let start = (local_y * tile.width + local_x) * STRIDE;
                        for (i, slot) in local[start..start + STRIDE].iter_mut().enumerate() {
                            *slot = synthetic_decode_value(seed * STRIDE + i);
                        }
                    }
                }
                local
            })
            .collect();

        for (tile, local) in tiles.iter().zip(&tile_values) {
            let row_len = tile.width * STRIDE;
            for local_y in 0..tile.height {
                let global_row = (tile.top + local_y) * MACROBLOCK_WIDTH + tile.left;
                values[global_row * STRIDE..][..row_len]
                    .copy_from_slice(&local[local_y * row_len..][..row_len]);
            }
        }

        black_box(values);
    });
}

/// Wraps a raw pointer so it can cross the `rayon` closure boundary.
///
/// # Safety
///
/// Every caller must guarantee that concurrent uses of this pointer touch disjoint memory
/// ranges; this type only removes the compiler's (overly conservative, for this case) refusal to
/// share a `*mut` across threads, it does not itself establish disjointness.
struct SharedBase(*mut i32);

// SAFETY: see the type's doc comment; this benchmark's only use (below) writes disjoint tile
// regions computed from `tile_grid`, which never overlap.
unsafe impl Sync for SharedBase {}

impl SharedBase {
    /// Reads the pointer through a method call so a closure capturing `self.get()` captures the
    /// whole `SharedBase` (and its `Sync` impl), not just the `*mut i32` field — Rust's disjoint
    /// closure captures would otherwise capture the bare, non-`Sync` pointer field directly.
    fn get(&self) -> *mut i32 {
        self.0
    }
}

#[divan::bench]
fn merge_via_direct_shared_write(bencher: Bencher<'_, '_>) {
    let tiles = tile_grid(MACROBLOCK_WIDTH, MACROBLOCK_HEIGHT, TILES_ACROSS, TILES_DOWN);
    let values = vec![0_i32; MACROBLOCK_WIDTH * MACROBLOCK_HEIGHT * STRIDE];

    bencher.bench_local(|| {
        let mut values = values.clone();
        let base = SharedBase(values.as_mut_ptr());

        tiles.par_iter().for_each(|tile| {
            let base = base.get();
            for local_y in 0..tile.height {
                let global_row = (tile.top + local_y) * MACROBLOCK_WIDTH + tile.left;
                for local_x in 0..tile.width {
                    let seed = (tile.top + local_y) * MACROBLOCK_WIDTH + tile.left + local_x;
                    let start = (global_row + local_x) * STRIDE;

                    // SAFETY: `tile_grid` partitions the macroblock grid into disjoint
                    // (row, column) rectangles, so the `start..start + STRIDE` range this tile
                    // writes never overlaps the range any other tile (running concurrently on
                    // another thread) writes. Each index is in-bounds because `start + STRIDE`
                    // never exceeds `values.len()` for a tile fully inside the macroblock grid.
                    unsafe {
                        for i in 0..STRIDE {
                            *base.add(start + i) = synthetic_decode_value(seed * STRIDE + i);
                        }
                    }
                }
            }
        });

        black_box(values);
    });
}
