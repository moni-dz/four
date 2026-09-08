//! Estimates `MaxCLL`, white points, and display HDR metrics from decoded source pixels.

use std::num::NonZeroUsize;

use rayon::prelude::*;
use tonemapping::{
    ColorChannel as ToneColorChannel, LinearRGB, LuminanceWhitePoint, LuminanceWhitePointEstimator,
    MaxCLLEstimator, MaxCLLMode, WhitePoint,
};

use super::pixel::PixelLayout;
use super::{
    Error, HDR_BATCH_PIXELS, JPEGXRColorChannel, JPEGXRError, PARALLEL_PIXELS_MIN,
    PARALLEL_PIXELS_PER_JOB, Result, SC_RGB_REFERENCE_WHITE_NITS, error,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct MaxCLL {
    pub(super) relative_light_level: f32,
    pub(super) channel: JPEGXRColorChannel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct HDRMetrics {
    pub(super) max_cll: MaxCLL,
    pub(super) max_cll_mode: MaxCLLMode,
    pub(super) luminance_white_point: Option<LuminanceWhitePoint>,
    pub(super) max_luminance_nits: f32,
    pub(super) average_luminance_nits: f32,
    pub(super) min_luminance_nits: f32,
    pub(super) rec709_percentage: f32,
    pub(super) dci_p3_percentage: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct HDRAnalysis {
    pub(super) max_cll: Option<MaxCLL>,
    pub(super) luminance_white_point: Option<LuminanceWhitePoint>,
    pub(super) hdr_metrics: Option<HDRMetrics>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct HDRPixelSelection {
    count: usize,
    exclude_fully_transparent: bool,
}

impl HDRPixelSelection {
    pub(super) fn new(source: &[u8], row_stride: usize, layout: PixelLayout) -> Result<Self> {
        let pixels_per_row = row_stride / layout.bytes_per_pixel;
        let row_count = source.len() / row_stride;
        let source_pixel_count = pixels_per_row
            .checked_mul(row_count)
            .expect("validated JPEG XR pixel count fits usize");

        invariant!(source_pixel_count > 0);

        let visible_alpha_pixels = if layout.has_alpha {
            visible_alpha_pixel_count(source, row_stride, layout)?
        } else {
            source_pixel_count
        };

        let exclude_fully_transparent = layout.has_alpha && visible_alpha_pixels > 0;

        let count = if exclude_fully_transparent {
            visible_alpha_pixels
        } else {
            source_pixel_count
        };

        invariant!(count > 0);
        invariant!(count <= source_pixel_count);

        Ok(Self {
            count,
            exclude_fully_transparent,
        })
    }
}

impl HDRMetrics {
    #[cfg(test)]
    pub(super) fn estimate(
        source: &[u8],
        row_stride: usize,
        layout: PixelLayout,
        max_cll_mode: MaxCLLMode,
        estimate_luminance_white_point: bool,
    ) -> Result<Self> {
        HDRAnalysis::estimate(
            source,
            row_stride,
            layout,
            max_cll_mode,
            AnalysisScope {
                estimate_max_cll: true,
                estimate_luminance_white_point,
                collect_hdr_metrics: true,
            },
        )
        .map(|analysis| {
            analysis
                .hdr_metrics
                .expect("full HDR analysis produces display metrics")
        })
    }
}

/// Which measurements an HDR analysis pass collects.
#[derive(Clone, Copy, Debug)]
pub(super) struct AnalysisScope {
    pub(super) estimate_max_cll: bool,
    pub(super) estimate_luminance_white_point: bool,
    pub(super) collect_hdr_metrics: bool,
}

/// The parameters that decide which measurements an analysis pass collects.
#[derive(Clone, Copy, Debug)]
pub(super) struct AnalysisRequest {
    pub(super) selection: HDRPixelSelection,
    pub(super) pixel_count: usize,
    pub(super) max_cll_mode: MaxCLLMode,
    pub(super) estimate_max_cll: bool,
    pub(super) estimate_luminance_white_point: bool,
    pub(super) collect_hdr_metrics: bool,
}

/// The measurements one worker gathers from its share of the image.
///
/// Fields merge associatively, allowing row groups to run in parallel.
#[derive(Debug)]
pub(super) struct AnalysisTotals {
    pub(super) accumulator: Option<HDRMetricAccumulator>,
    pub(super) max_cll_estimator: Option<MaxCLLEstimator>,
    pub(super) luminance_white_point_estimator: Option<LuminanceWhitePointEstimator>,
    pub(super) max_cll_batch: Option<Vec<LinearRGB>>,
}

impl AnalysisTotals {
    pub(super) fn new(request: &AnalysisRequest) -> Self {
        Self {
            accumulator: request.collect_hdr_metrics.then(HDRMetricAccumulator::new),
            // Every partial estimator declares the whole image, so the merged observation count
            // matches what `finish` requires and the retained sample counts agree.
            max_cll_estimator: request.estimate_max_cll.then(|| {
                MaxCLLEstimator::with_mode(
                    NonZeroUsize::new(request.pixel_count)
                        .expect("HDR analysis includes at least one pixel"),
                    request.max_cll_mode,
                )
            }),
            luminance_white_point_estimator: request.estimate_luminance_white_point.then(|| {
                LuminanceWhitePointEstimator::new(
                    NonZeroUsize::new(request.pixel_count)
                        .expect("HDR analysis includes at least one pixel"),
                )
            }),
            max_cll_batch: request
                .estimate_max_cll
                .then(|| Vec::with_capacity(HDR_BATCH_PIXELS.min(request.pixel_count))),
        }
    }

    pub(super) fn observe_slab(
        &mut self,
        source: &[u8],
        row_stride: usize,
        layout: PixelLayout,
        request: &AnalysisRequest,
    ) -> Result<()> {
        visit_pixels(source, row_stride, layout, |color, alpha| {
            if request.selection.exclude_fully_transparent && alpha == 0.0 {
                return;
            }

            if let Some(accumulator) = &mut self.accumulator {
                accumulator.observe(color);
            }

            let color = LinearRGB::new(color);
            if let Some(estimator) = &mut self.luminance_white_point_estimator {
                estimator.observe(color);
            }

            if let Some(batch) = &mut self.max_cll_batch {
                batch.push(color);
                if batch.len() == HDR_BATCH_PIXELS {
                    self.max_cll_estimator
                        .as_mut()
                        .expect("a MaxCLL batch has an estimator")
                        .observe_many(batch);
                    batch.clear();
                }
            }
        })
    }

    fn merge(&mut self, other: Self) {
        if let (Some(accumulator), Some(other)) = (&mut self.accumulator, other.accumulator) {
            accumulator.merge(other);
        }

        // Drain the other worker's pending batch into its own estimator before merging, so no
        // observation is lost and the counts still add up.
        let mut other_max_cll = other.max_cll_estimator;
        if let (Some(estimator), Some(batch)) =
            (other_max_cll.as_mut(), other.max_cll_batch.as_ref())
        {
            estimator.observe_many(batch);
        }
        if let (Some(estimator), Some(other)) = (&mut self.max_cll_estimator, other_max_cll) {
            estimator.merge(other);
        }

        if let (Some(estimator), Some(other)) = (
            &mut self.luminance_white_point_estimator,
            other.luminance_white_point_estimator,
        ) {
            estimator.merge(other);
        }
    }
}

impl HDRAnalysis {
    pub(super) fn estimate(
        source: &[u8],
        row_stride: usize,
        layout: PixelLayout,
        max_cll_mode: MaxCLLMode,
        scope: AnalysisScope,
    ) -> Result<Self> {
        invariant!(layout.encoding.is_hdr());
        invariant!(row_stride >= layout.bytes_per_pixel);
        invariant!(!scope.collect_hdr_metrics || scope.estimate_max_cll);

        let selection = HDRPixelSelection::new(source, row_stride, layout)?;
        let pixel_count = selection.count;

        let request = AnalysisRequest {
            selection,
            pixel_count,
            max_cll_mode,
            estimate_max_cll: scope.estimate_max_cll,
            estimate_luminance_white_point: scope.estimate_luminance_white_point,
            collect_hdr_metrics: scope.collect_hdr_metrics,
        };

        let totals = compute_totals(source, row_stride, layout, &request)?;

        let AnalysisTotals {
            accumulator,
            mut max_cll_estimator,
            luminance_white_point_estimator,
            max_cll_batch,
        } = totals;

        if let Some(batch) = &max_cll_batch {
            max_cll_estimator
                .as_mut()
                .expect("a MaxCLL batch has an estimator")
                .observe_many(batch);
        }

        if let Some(accumulator) = &accumulator {
            invariant_eq!(
                usize::try_from(accumulator.pixel_count)
                    .expect("the bounded JPEG XR pixel count fits usize"),
                pixel_count
            );
        }

        let max_cll = max_cll_estimator.map(finish_max_cll);
        let luminance_white_point = luminance_white_point_estimator.and_then(|estimator| {
            estimator
                .finish()
                .expect("HDR analysis observes exactly its declared pixel count")
        });
        let hdr_metrics = accumulator.map(|accumulator| {
            accumulator.finish(
                max_cll.expect("display HDR metrics include MaxCLL"),
                max_cll_mode,
                luminance_white_point,
            )
        });

        Ok(Self {
            max_cll,
            luminance_white_point,
            hdr_metrics,
        })
    }
}

/// Gathers mergeable HDR measurements over parallel row groups.
///
/// Jobs are sized in pixels for comparable row-group sizes across formats.
fn compute_totals(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
    request: &AnalysisRequest,
) -> Result<AnalysisTotals> {
    let width = row_stride / layout.bytes_per_pixel;
    let rows_per_job = PARALLEL_PIXELS_PER_JOB.div_ceil(width);

    // `visit_pixels` below scans every source pixel regardless of transparency, so the
    // serial/parallel split must be sized off the pixels actually scanned, not
    // `request.pixel_count` (which, for images with transparency, counts only the visible pixels
    // the estimators retain). Otherwise a large mostly-transparent image runs its full scan
    // serially.
    let row_count = source.len() / row_stride;
    let scanned_pixel_count = width * row_count;

    if scanned_pixel_count < PARALLEL_PIXELS_MIN {
        let mut totals = AnalysisTotals::new(request);
        totals.observe_slab(source, row_stride, layout, request)?;
        return Ok(totals);
    }

    // `try_fold` reuses one `AnalysisTotals` (and its full-image-sized percentile heaps) across
    // every chunk rayon assigns to the same split, instead of allocating a fresh one per chunk:
    // a `map` here would size those heaps for the whole image on every one of the ~1000 chunks a
    // large image produces.
    source
        .par_chunks(rows_per_job * row_stride)
        .try_fold(
            || AnalysisTotals::new(request),
            |mut totals, slab| {
                totals.observe_slab(slab, row_stride, layout, request)?;
                Ok::<AnalysisTotals, Error>(totals)
            },
        )
        .try_reduce(
            || AnalysisTotals::new(request),
            |mut left, right| {
                left.merge(right);
                Ok::<AnalysisTotals, Error>(left)
            },
        )
}

pub(super) fn finish_max_cll(estimator: MaxCLLEstimator) -> MaxCLL {
    let estimate = estimator
        .finish()
        .expect("the HDR analysis pass visits the measured number of pixels");
    let relative_light_level = estimate.level();

    invariant!(relative_light_level.is_finite());
    invariant!(relative_light_level >= 0.0);

    MaxCLL {
        relative_light_level,
        channel: jpeg_xr_color_channel(estimate.channel()),
    }
}

impl MaxCLL {
    pub(super) fn relative_light_level(self) -> f32 {
        invariant!(self.relative_light_level.is_finite());
        invariant!(self.relative_light_level >= 0.0);
        self.relative_light_level
    }

    pub(super) fn nits(self) -> f32 {
        nonnegative_f64_to_f32(
            f64::from(self.relative_light_level) * f64::from(SC_RGB_REFERENCE_WHITE_NITS),
        )
    }
}

pub(super) fn hdr_white_point(max_cll: MaxCLL) -> WhitePoint {
    WhitePoint::new(max_cll.relative_light_level().max(1.0))
        .expect("MaxCLL floored at display white is a positive finite white point")
}

pub(super) fn display_white_point() -> WhitePoint {
    WhitePoint::new(1.0).expect("display white is a positive finite white point")
}

pub(super) fn hdr_luminance_white_point(white_point: LuminanceWhitePoint) -> LuminanceWhitePoint {
    let luminance = white_point.luminance().max(1.0);
    LuminanceWhitePoint::new(luminance)
        .expect("p99.99 luminance floored at display white is a positive finite white point")
}

pub(super) fn display_luminance_white_point() -> LuminanceWhitePoint {
    LuminanceWhitePoint::new(1.0).expect("display white is a positive finite luminance white point")
}

#[derive(Clone, Copy, Debug)]
pub(super) struct HDRMetricAccumulator {
    pixel_count: u64,
    luminance_sum_nits: f64,
    max_luminance_nits: f64,
    min_luminance_nits: f64,
    rec709_pixels: u64,
    dci_p3_pixels: u64,
}

impl HDRMetricAccumulator {
    const fn new() -> Self {
        Self {
            pixel_count: 0,
            luminance_sum_nits: 0.0,
            max_luminance_nits: f64::NEG_INFINITY,
            min_luminance_nits: f64::INFINITY,
            rec709_pixels: 0,
            dci_p3_pixels: 0,
        }
    }

    fn observe(&mut self, color: [f32; 3]) {
        let color = color.map(sanitize_metric_sample);

        // Rec. 709 luma weights (ITU-R BT.709-6 section 3.2), applied to linear scRGB components
        // to get relative luminance, then scaled to absolute nits by the scRGB reference white.
        let luminance = (0.212_6 * color[0] + 0.715_2 * color[1] + 0.072_2 * color[2]).max(0.0)
            * f64::from(SC_RGB_REFERENCE_WHITE_NITS);

        self.pixel_count += 1;
        self.luminance_sum_nits += luminance;
        self.max_luminance_nits = self.max_luminance_nits.max(luminance);
        self.min_luminance_nits = self.min_luminance_nits.min(luminance);

        match gamut_membership(color) {
            GamutMembership::Rec709 => self.rec709_pixels += 1,
            GamutMembership::DisplayP3Only => self.dci_p3_pixels += 1,
            GamutMembership::OutsideDisplayP3 => {}
        }
    }

    /// Folds `other` into these metrics.
    ///
    /// Counts and sums add; extrema take the wider bound. Average luminance can vary by a rounding
    /// step with worker count.
    fn merge(&mut self, other: Self) {
        self.pixel_count += other.pixel_count;
        self.luminance_sum_nits += other.luminance_sum_nits;
        self.max_luminance_nits = self.max_luminance_nits.max(other.max_luminance_nits);
        self.min_luminance_nits = self.min_luminance_nits.min(other.min_luminance_nits);
        self.rec709_pixels += other.rec709_pixels;
        self.dci_p3_pixels += other.dci_p3_pixels;
    }

    pub(super) fn finish(
        self,
        max_cll: MaxCLL,
        max_cll_mode: MaxCLLMode,
        luminance_white_point: Option<LuminanceWhitePoint>,
    ) -> HDRMetrics {
        invariant!(self.pixel_count > 0);
        invariant!(self.max_luminance_nits.is_finite());
        invariant!(self.min_luminance_nits.is_finite());

        let pixel_count = f64::from(
            u32::try_from(self.pixel_count)
                .expect("the bounded JPEG XR pixel count fits u32 metadata arithmetic"),
        );

        HDRMetrics {
            max_cll,
            max_cll_mode,
            luminance_white_point,
            max_luminance_nits: nonnegative_f64_to_f32(self.max_luminance_nits),
            average_luminance_nits: nonnegative_f64_to_f32(self.luminance_sum_nits / pixel_count),
            min_luminance_nits: nonnegative_f64_to_f32(self.min_luminance_nits),
            rec709_percentage: percentage(self.rec709_pixels, self.pixel_count),
            dci_p3_percentage: percentage(self.dci_p3_pixels, self.pixel_count),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GamutMembership {
    Rec709,
    DisplayP3Only,
    OutsideDisplayP3,
}

fn gamut_membership(color: [f64; 3]) -> GamutMembership {
    let scale = color.iter().copied().map(f64::abs).fold(1.0_f64, f64::max);
    let epsilon = 1.0e-6 * scale;

    if color.iter().all(|channel| *channel >= -epsilon) {
        return GamutMembership::Rec709;
    }

    // Converts Rec. 709 linear RGB (already established as out-of-gamut above) to Display-P3
    // linear RGB via the direct Rec. 709 -> Display-P3 primaries matrix (both D65 white points,
    // so no chromatic adaptation step is needed). Nonnegative components here mean the color is
    // representable in Display-P3 even though it fell outside Rec. 709.
    let display_p3 = [
        0.822_592_87 * color[0] + 0.177_533_95 * color[1],
        0.033_199_51 * color[0] + 0.966_783_50 * color[1],
        0.017_085_35 * color[0] + 0.072_395_72 * color[1] + 0.910_301_48 * color[2],
    ];

    if display_p3.iter().all(|channel| *channel >= -epsilon) {
        GamutMembership::DisplayP3Only
    } else {
        GamutMembership::OutsideDisplayP3
    }
}

fn sanitize_metric_sample(value: f32) -> f64 {
    if value.is_nan() {
        0.0
    } else if value == f32::INFINITY {
        f64::from(f32::MAX)
    } else if value == f32::NEG_INFINITY {
        -f64::from(f32::MAX)
    } else {
        f64::from(value)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "bounded metadata values are intentionally returned as the decoder's f32 scalar type"
)]
fn nonnegative_f64_to_f32(value: f64) -> f32 {
    invariant!(value.is_finite());
    invariant!(value >= 0.0);
    value.min(f64::from(f32::MAX)) as f32
}

fn percentage(part: u64, total: u64) -> f32 {
    invariant!(part <= total);
    invariant!(total > 0);

    let part = u32::try_from(part).expect("the bounded JPEG XR pixel count fits u32");
    let total = u32::try_from(total).expect("the bounded JPEG XR pixel count fits u32");
    percentage_from_u32(part, total)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a bounded 0..=100 metadata percentage is intentionally stored as f32"
)]
fn percentage_from_u32(part: u32, total: u32) -> f32 {
    (f64::from(part) * 100.0 / f64::from(total)) as f32
}

const fn jpeg_xr_color_channel(channel: ToneColorChannel) -> JPEGXRColorChannel {
    match channel {
        ToneColorChannel::Red => JPEGXRColorChannel::Red,
        ToneColorChannel::Green => JPEGXRColorChannel::Green,
        ToneColorChannel::Blue => JPEGXRColorChannel::Blue,
    }
}

pub(super) fn visit_pixels(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
    mut visitor: impl FnMut([f32; 3], f32),
) -> Result<()> {
    invariant!(row_stride >= layout.bytes_per_pixel);

    let mut rows = source.chunks_exact(row_stride);
    for row in &mut rows {
        let mut pixels = row.chunks_exact(layout.bytes_per_pixel);

        for pixel in &mut pixels {
            let (color, alpha) = layout.read_pixel(pixel)?;
            visitor(color, alpha);
        }

        if !pixels.remainder().is_empty() {
            return Err(error(JPEGXRError::Output(
                "JPEG XR row contains a partial pixel",
            )));
        }
    }

    if !rows.remainder().is_empty() {
        return Err(error(JPEGXRError::Output(
            "JPEG XR source buffer contains a partial row",
        )));
    }

    Ok(())
}

/// Counts source pixels with nonzero alpha over parallel row groups.
fn visible_alpha_pixel_count(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
) -> Result<usize> {
    invariant!(layout.has_alpha);

    let width = row_stride / layout.bytes_per_pixel;
    let row_count = source.len() / row_stride;
    let scanned_pixel_count = width * row_count;

    if scanned_pixel_count < PARALLEL_PIXELS_MIN {
        return visible_alpha_pixel_count_slab(source, row_stride, layout);
    }

    let rows_per_job = PARALLEL_PIXELS_PER_JOB.div_ceil(width);
    source
        .par_chunks(rows_per_job * row_stride)
        .try_fold(
            || 0_usize,
            |count, slab| {
                Ok::<usize, Error>(
                    count + visible_alpha_pixel_count_slab(slab, row_stride, layout)?,
                )
            },
        )
        .try_reduce(|| 0_usize, |left, right| Ok(left + right))
}

fn visible_alpha_pixel_count_slab(
    source: &[u8],
    row_stride: usize,
    layout: PixelLayout,
) -> Result<usize> {
    let mut visible_pixels = 0_usize;

    visit_pixels(source, row_stride, layout, |_color, alpha| {
        if alpha > 0.0 {
            visible_pixels = visible_pixels
                .checked_add(1)
                .expect("validated JPEG XR pixel count fits usize");
        }
    })?;

    Ok(visible_pixels)
}
