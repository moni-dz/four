//! Coalesces image-decode requests so only the latest queued job runs after the active one.

use std::path::Path;
use std::sync::Arc;

use super::HDROptions;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LoadRequest(pub(super) u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LoadPurpose {
    Image,
    HdrMetrics,
    HDROptions,
}

#[derive(Debug)]
pub(super) struct DecodeJob<T> {
    pub(super) request: LoadRequest,
    pub(super) payload: T,
}

#[derive(Debug)]
pub(super) struct DecodePayload {
    pub(super) hdr_options: HDROptions,
    pub(super) include_hdr_metrics: bool,
    pub(super) path: Arc<Path>,
    pub(super) purpose: LoadPurpose,
}

// Decoders are synchronous, so an active job must finish. Keeping only the latest waiting job
// bounds decode work to one allocation-heavy operation at a time without misrepresenting it as
// cancellable.
#[derive(Debug)]
pub(super) struct LatestLoadCoordinator<T> {
    active: Option<LoadRequest>,
    queued: Option<DecodeJob<T>>,
}

impl<T> LatestLoadCoordinator<T> {
    pub(super) fn new() -> Self {
        Self {
            active: None,
            queued: None,
        }
    }

    pub(super) fn submit(&mut self, job: DecodeJob<T>) -> Option<DecodeJob<T>> {
        if self.active.is_none() {
            self.active = Some(job.request);
            return Some(job);
        }

        self.queued = Some(job);
        None
    }

    pub(super) fn complete(&mut self, request: LoadRequest) -> Option<DecodeJob<T>> {
        let active = self
            .active
            .take()
            .expect("a decode completed while the load coordinator was idle");
        assert_eq!(
            active, request,
            "a decode other than the active request completed"
        );

        let next = self.queued.take();
        self.active = next.as_ref().map(|job| job.request);
        next
    }

    pub(super) fn discard_queued(&mut self) {
        self.queued = None;
    }
}
