//! A reusable scratch arena for the decoder's intermediate coefficient and sample buffers.
//!
//! One decode of a 4K image allocates a few hundred buffers totalling about 200 MB, most of it a
//! handful of whole-image coefficient planes. From the global allocator each large plane arrives as
//! freshly mapped memory that faults in page by page and is unmapped again on drop. [`Arena`] wraps
//! a [`blink_alloc::SyncBlinkAlloc`] instead: allocation bumps a pointer in a shared chunk from any
//! thread, freeing a whole decode is a reset, and the reset keeps the last chunk for the next
//! decode.
//!
//! `SyncBlinkAlloc` sizes each new chunk from the chunks allocated before it since the last reset,
//! and a reset discards all but the last. With a few large buffers per decode that never settles:
//! every cycle starts small and maps fresh chunks for the big planes again. So the arena counts
//! each cycle's demand and, when it outgrows the chunk size, rebuilds the allocator with a chunk
//! size that holds the whole decode, after which one retained chunk serves every cycle.

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use blink_alloc::SyncBlinkAlloc;

/// A thread-safe bump scratch allocator reused across decodes.
///
/// Pass one to [`Decoder::decode_bgr101010_in`](crate::Decoder::decode_bgr101010_in) (or the RGBA
/// variants) to keep the decoder's intermediate buffers in memory that is already mapped. The
/// arena retains its last chunk between decodes; drop it to return the memory.
#[derive(Default)]
pub struct Arena {
    blink: SyncBlinkAlloc,
    /// Minimum size of a new chunk, grown to the largest demand seen so far.
    chunk_size: usize,
    /// Bytes requested since the last reset.
    demand: AtomicUsize,
}

/// Granularity and slack of the chunk size, covering alignment padding and chunk headers.
const CHUNK_SLACK: usize = 1 << 20;

impl Arena {
    /// Creates an empty arena; its first chunk is sized by the first allocation.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            blink: SyncBlinkAlloc::new(),
            chunk_size: 0,
            demand: AtomicUsize::new(0),
        }
    }

    /// Frees every allocation at once, keeping the last chunk for reuse.
    ///
    /// If the previous cycle asked for more than the chunk size, the allocator is rebuilt so its
    /// next chunk holds that much. Taking `&mut self` proves no allocation from the previous cycle
    /// is still alive.
    pub fn reset(&mut self) {
        let demand = std::mem::take(self.demand.get_mut());
        if demand > self.chunk_size {
            self.chunk_size = demand.next_multiple_of(CHUNK_SLACK) + CHUNK_SLACK;
            // Dropping the old allocator returns all of its chunks.
            self.blink = SyncBlinkAlloc::with_chunk_size_in(self.chunk_size, Global);
        } else {
            self.blink.reset();
        }
    }

    /// Returns the minimum size of the chunks the arena allocates.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arena").finish_non_exhaustive()
    }
}

#[expect(
    unsafe_code,
    reason = "forwards the allocator contract to blink-alloc's `Allocator for SyncBlinkAlloc`"
)]
// SAFETY: every call forwards unchanged to `SyncBlinkAlloc`, which is `Sync` and serves concurrent
// requests itself; the demand counter is a relaxed atomic that only sizes future chunks. Its blocks are only freed by `reset` or drop, both of which need `&mut Arena`
// and so outlive every `&Arena` a collection holds.
unsafe impl Allocator for &Arena {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.demand.fetch_add(layout.size(), Ordering::Relaxed);
        self.blink.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller's contract for `ptr` and `layout` carries over unchanged.
        unsafe { Allocator::deallocate(&self.blink, ptr, layout) };
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.demand.fetch_add(new_layout.size(), Ordering::Relaxed);
        // SAFETY: the caller's contract for the block carries over unchanged.
        unsafe { Allocator::grow(&self.blink, ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller's contract for the block carries over unchanged.
        unsafe { Allocator::shrink(&self.blink, ptr, old_layout, new_layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::Arena;

    #[test]
    fn reset_reuses_the_retained_chunk() {
        let mut arena = Arena::new();
        let allocate = |arena: &Arena| {
            let values: Vec<u32, _> = Vec::with_capacity_in(1 << 16, arena);
            values.as_ptr()
        };

        // The first reset sizes the chunk to the first cycle's demand; later cycles reuse it.
        allocate(&arena);
        arena.reset();
        let first = allocate(&arena);
        arena.reset();
        assert_eq!(allocate(&arena), first);
    }

    #[test]
    fn reset_grows_the_chunk_to_hold_a_whole_cycle() {
        let mut arena = Arena::new();
        for _ in 0..2 {
            let first: Vec<u8, _> = Vec::with_capacity_in(3 << 20, &arena);
            let second: Vec<u8, _> = Vec::with_capacity_in(5 << 20, &arena);
            drop((first, second));
            arena.reset();
        }
        assert!(arena.chunk_size() >= 8 << 20);

        // Both buffers now land in the one retained chunk, back to back.
        let first: Vec<u8, _> = Vec::with_capacity_in(3 << 20, &arena);
        let second: Vec<u8, _> = Vec::with_capacity_in(5 << 20, &arena);
        let distance = first.as_ptr().addr().abs_diff(second.as_ptr().addr());
        assert!(distance < arena.chunk_size());
    }

    #[test]
    fn allocates_from_many_threads_at_once() {
        let arena = Arena::new();
        std::thread::scope(|scope| {
            for thread in 0..4_u32 {
                let arena = &arena;
                scope.spawn(move || {
                    let mut values = Vec::new_in(arena);
                    values.extend((0..10_000).map(|value| value * thread));
                    assert!(values.iter().copied().eq((0..10_000).map(|v| v * thread)));
                });
            }
        });
    }

    #[test]
    fn zeroed_allocations_are_cleared_after_reuse() {
        let mut arena = Arena::new();
        let mut dirty: Vec<u8, _> = Vec::with_capacity_in(4096, &arena);
        dirty.resize(4096, u8::MAX);
        drop(dirty);

        arena.reset();
        let zeroed = Box::<[u8], _>::new_zeroed_slice_in(4096, &arena);
        // SAFETY: zero is a valid `u8`.
        #[expect(unsafe_code, reason = "reads back a zeroed allocation")]
        let zeroed = unsafe { zeroed.assume_init() };
        assert!(zeroed.iter().all(|&byte| byte == 0));
    }
}
