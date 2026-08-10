//! Owns frame geometry, bounded coefficient storage, and pixel materialization.

use exn::OptionExt;
use rayon::prelude::*;

use super::{
    BLOCK_SIDE, COMPONENTS_MAX, DIMENSION_MAX, DecodedImage, Error, JPEGError, JPEGLimit,
    JPEGTableKind, PIXELS_MAX, PROGRESSIVE_COEFFICIENT_BYTES_MAX, Result, divide_ceil, error, idct,
};

const PARALLEL_PIXELS_MIN: usize = 256 * 1024;
const PARALLEL_PIXELS_PER_JOB: usize = 64 * 1024;
const PARALLEL_BLOCKS_MIN: usize = 4 * 1024;
const PARALLEL_BLOCKS_PER_JOB: usize = 1_024;

#[derive(Clone, Copy)]
pub(super) enum ColorTransform {
    YCbCr,
    Rgb,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CodingProcess {
    Sequential,
    Progressive,
}

pub(super) struct FrameComponent {
    pub(super) identifier: u8,
    pub(super) horizontal_sampling: u8,
    pub(super) vertical_sampling: u8,
    pub(super) quantization_table: usize,
    pub(super) plane_width: u32,
    pub(super) plane: Vec<u8>,
    pub(super) block_columns: u32,
    pub(super) block_rows: u32,
    pub(super) data_block_columns: u32,
    pub(super) data_block_rows: u32,
    pub(super) coefficients: Vec<[i32; 64]>,
}

pub(super) struct Frame {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mcu_columns: u32,
    pub(super) mcu_rows: u32,
    pub(super) max_horizontal_sampling: u8,
    pub(super) max_vertical_sampling: u8,
    pub(super) components: Vec<FrameComponent>,
    pub(super) process: CodingProcess,
}

impl Frame {
    pub(super) fn new(
        width: u32,
        height: u32,
        mut components: Vec<FrameComponent>,
        process: CodingProcess,
    ) -> Result<Self> {
        invariant!(!components.is_empty());
        invariant!(components.len() <= COMPONENTS_MAX);

        let max_horizontal_sampling = components
            .iter()
            .map(|component| component.horizontal_sampling)
            .max()
            .expect("frame has components");

        let max_vertical_sampling = components
            .iter()
            .map(|component| component.vertical_sampling)
            .max()
            .expect("frame has components");

        let mcu_width = u32::from(max_horizontal_sampling) * BLOCK_SIDE;
        let mcu_height = u32::from(max_vertical_sampling) * BLOCK_SIDE;
        let mcu_columns = divide_ceil(width, mcu_width);
        let mcu_rows = divide_ceil(height, mcu_height);

        let storage = ComponentStorageLayout {
            width,
            height,
            mcu_columns,
            mcu_rows,
            max_horizontal_sampling,
            max_vertical_sampling,
            process,
        };

        validate_progressive_storage(&components, &storage)?;

        components
            .iter_mut()
            .try_for_each(|component| allocate_component_storage(component, &storage))?;

        Ok(Self {
            width,
            height,
            mcu_columns,
            mcu_rows,
            max_horizontal_sampling,
            max_vertical_sampling,
            components,
            process,
        })
    }

    pub(super) fn materialize_progressive(
        &mut self,
        quantization_snapshots: &[Option<[u16; 64]>; COMPONENTS_MAX],
    ) -> Result<()> {
        invariant_eq!(self.process, CodingProcess::Progressive);
        invariant!(!self.components.is_empty());

        for (component_index, component) in self.components.iter_mut().enumerate() {
            let quantization = quantization_snapshots[component_index].ok_or_raise(|| {
                JPEGError::Table(
                    JPEGTableKind::Quantization,
                    "progressive component is missing its quantization table snapshot",
                )
            })?;

            let block_count = component
                .block_columns
                .checked_mul(component.block_rows)
                .ok_or_raise(|| {
                    JPEGError::ArithmeticOverflow("component block count overflowed")
                })?;

            if component.coefficients.len() != block_count as usize {
                return Err(error(JPEGError::Frame(
                    "progressive coefficient plane has an invalid size",
                )));
            }

            let block_count =
                usize::try_from(block_count).expect("the bounded component block count fits usize");
            if block_count >= PARALLEL_BLOCKS_MIN {
                materialize_component_parallel(component, &quantization)?;
            } else {
                materialize_component_sequential(component, &quantization)?;
            }

            component.coefficients = Vec::new();
        }

        Ok(())
    }

    pub(super) fn into_image(self, transform: ColorTransform) -> DecodedImage {
        invariant!(self.components.len() == 1 || self.components.len() == 3);
        invariant!(u64::from(self.width) * u64::from(self.height) <= PIXELS_MAX);

        let byte_count = u64::from(self.width) * u64::from(self.height) * 4;
        let capacity = usize::try_from(byte_count)
            .expect("the decoded pixel limit fits every supported pointer width");
        let pixel_count = capacity / 4;
        let rgba = if pixel_count >= PARALLEL_PIXELS_MIN {
            self.rgba_pixels_parallel(transform, capacity)
        } else {
            self.rgba_pixels_sequential(transform, capacity)
        };

        invariant_eq!(rgba.len(), capacity);
        DecodedImage::new(self.width, self.height, rgba)
    }

    fn rgba_pixels_sequential(&self, transform: ColorTransform, capacity: usize) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(capacity);
        for y in 0..self.height {
            for x in 0..self.width {
                rgba.extend_from_slice(&self.rgba_pixel(x, y, transform));
            }
        }
        rgba
    }

    fn rgba_pixels_parallel(&self, transform: ColorTransform, capacity: usize) -> Vec<u8> {
        let width = usize::try_from(self.width).expect("validated JPEG width fits usize");
        let mut rgba = vec![0; capacity];
        rgba.par_chunks_exact_mut(4)
            .with_min_len(PARALLEL_PIXELS_PER_JOB)
            .enumerate()
            .for_each(|(index, target)| {
                let x = u32::try_from(index % width).expect("JPEG pixel x fits u32");
                let y = u32::try_from(index / width).expect("JPEG pixel y fits u32");
                target.copy_from_slice(&self.rgba_pixel(x, y, transform));
            });
        rgba
    }

    fn rgba_pixel(&self, x: u32, y: u32, transform: ColorTransform) -> [u8; 4] {
        let first = self.sample(0, x, y);
        if self.components.len() == 1 {
            [first, first, first, 255]
        } else {
            let second = self.sample(1, x, y);
            let third = self.sample(2, x, y);
            convert_color(first, second, third, transform)
        }
    }

    pub(super) fn sample(&self, component_index: usize, x: u32, y: u32) -> u8 {
        invariant!(component_index < self.components.len());
        invariant!(x < self.width);
        invariant!(y < self.height);

        let component = &self.components[component_index];
        let sample_x =
            x * u32::from(component.horizontal_sampling) / u32::from(self.max_horizontal_sampling);
        let sample_y =
            y * u32::from(component.vertical_sampling) / u32::from(self.max_vertical_sampling);
        let index = u64::from(sample_y) * u64::from(component.plane_width) + u64::from(sample_x);
        let index = usize::try_from(index).expect("the bounded component plane index fits usize");
        component.plane[index]
    }
}

fn materialize_component_sequential(
    component: &mut FrameComponent,
    quantization: &[u16; 64],
) -> Result<()> {
    for block_index_usize in 0..component.coefficients.len() {
        let block_index =
            u32::try_from(block_index_usize).expect("the bounded component block count fits u32");
        let quantized = component.coefficients[block_index_usize];
        let coefficients = dequantize_block(&quantized, quantization)?;
        let samples = idct::inverse(&coefficients);
        let block_x = block_index % component.block_columns;
        let block_y = block_index / component.block_columns;
        write_block(component, block_x, block_y, &samples);
    }
    Ok(())
}

fn materialize_component_parallel(
    component: &mut FrameComponent,
    quantization: &[u16; 64],
) -> Result<()> {
    let block_columns = usize::try_from(component.block_columns)
        .expect("the bounded component block width fits usize");
    let plane_width = usize::try_from(component.plane_width)
        .expect("the bounded component plane width fits usize");
    let block_side = usize::try_from(BLOCK_SIDE).expect("JPEG block side fits usize");
    let plane_bytes_per_block_row = plane_width * block_side;
    let block_rows_per_job = PARALLEL_BLOCKS_PER_JOB.div_ceil(block_columns);
    let coefficients = &component.coefficients;

    component
        .plane
        .par_chunks_mut(plane_bytes_per_block_row)
        .zip(coefficients.par_chunks(block_columns))
        .with_min_len(block_rows_per_job)
        .try_for_each(|(plane, coefficient_row)| {
            for (block_x, quantized) in coefficient_row.iter().enumerate() {
                let coefficients = dequantize_block(quantized, quantization)?;
                let samples = idct::inverse(&coefficients);
                write_block_row(plane, plane_width, block_x, &samples);
            }
            Ok::<(), Error>(())
        })
}

fn write_block_row(plane: &mut [u8], plane_width: usize, block_x: usize, samples: &[u8; 64]) {
    let block_side = usize::try_from(BLOCK_SIDE).expect("JPEG block side fits usize");
    let pixel_x = block_x * block_side;
    for y in 0..block_side {
        let target_start = y * plane_width + pixel_x;
        let source_start = y * block_side;
        plane[target_start..target_start + block_side]
            .copy_from_slice(&samples[source_start..source_start + block_side]);
    }
}

pub(super) fn dequantize_block(values: &[i32; 64], quantization: &[u16; 64]) -> Result<[i32; 64]> {
    invariant!(quantization.iter().all(|value| *value > 0));
    invariant!(values.iter().all(|value| value.checked_abs().is_some()));

    let mut coefficients = [0_i32; 64];
    for index in 0..64 {
        coefficients[index] = values[index]
            .checked_mul(i32::from(quantization[index]))
            .ok_or_raise(|| {
                JPEGError::ArithmeticOverflow("dequantized progressive coefficient overflowed")
            })?;
    }
    Ok(coefficients)
}

pub(super) fn write_block(
    component: &mut FrameComponent,
    block_x: u32,
    block_y: u32,
    samples: &[u8; 64],
) {
    invariant!(component.plane_width > 0);
    invariant!(!component.plane.is_empty());

    let pixel_x = block_x * BLOCK_SIDE;
    let pixel_y = block_y * BLOCK_SIDE;
    for y in 0..BLOCK_SIDE {
        let target_start =
            u64::from(pixel_y + y) * u64::from(component.plane_width) + u64::from(pixel_x);
        let target_end = target_start + u64::from(BLOCK_SIDE);
        let source_start =
            usize::try_from(y * BLOCK_SIDE).expect("an eight-row JPEG block always fits usize");
        let source_end = source_start + BLOCK_SIDE as usize;
        let target_start =
            usize::try_from(target_start).expect("the bounded component plane offset fits usize");
        let target_end =
            usize::try_from(target_end).expect("the bounded component plane offset fits usize");
        component.plane[target_start..target_end]
            .copy_from_slice(&samples[source_start..source_end]);
    }
}

struct ComponentStorageLayout {
    width: u32,
    height: u32,
    mcu_columns: u32,
    mcu_rows: u32,
    max_horizontal_sampling: u8,
    max_vertical_sampling: u8,
    process: CodingProcess,
}

fn validate_progressive_storage(
    components: &[FrameComponent],
    layout: &ComponentStorageLayout,
) -> Result<()> {
    invariant!(!components.is_empty());
    invariant!(components.len() <= COMPONENTS_MAX);

    if layout.process == CodingProcess::Sequential {
        return Ok(());
    }
    let byte_count = components.iter().try_fold(0_u64, |byte_count, component| {
        let blocks = u64::from(layout.mcu_columns)
            * u64::from(component.horizontal_sampling)
            * u64::from(layout.mcu_rows)
            * u64::from(component.vertical_sampling);
        let component_bytes = blocks * 64 * size_of::<i32>() as u64;

        byte_count.checked_add(component_bytes).ok_or_raise(|| {
            JPEGError::ArithmeticOverflow("progressive coefficient storage overflowed")
        })
    })?;

    if byte_count > PROGRESSIVE_COEFFICIENT_BYTES_MAX {
        return Err(error(JPEGError::LimitExceeded(
            JPEGLimit::ProgressiveCoefficientBytes(PROGRESSIVE_COEFFICIENT_BYTES_MAX),
        )));
    }
    Ok(())
}

fn allocate_component_storage(
    component: &mut FrameComponent,
    layout: &ComponentStorageLayout,
) -> Result<()> {
    invariant!(component.plane.is_empty());
    invariant!(component.horizontal_sampling > 0);

    let block_columns = layout
        .mcu_columns
        .checked_mul(u32::from(component.horizontal_sampling))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component block width overflowed"))?;
    let block_rows = layout
        .mcu_rows
        .checked_mul(u32::from(component.vertical_sampling))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component block height overflowed"))?;
    let plane_width = block_columns
        .checked_mul(BLOCK_SIDE)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component plane width overflowed"))?;
    let plane_height = block_rows
        .checked_mul(BLOCK_SIDE)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component plane height overflowed"))?;
    let sample_count = plane_width
        .checked_mul(plane_height)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component plane size overflowed"))?;
    let data_width = layout
        .width
        .checked_mul(u32::from(component.horizontal_sampling))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component data width overflowed"))?;
    let data_height = layout
        .height
        .checked_mul(u32::from(component.vertical_sampling))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("component data height overflowed"))?;
    let data_block_columns = divide_ceil(
        data_width,
        u32::from(layout.max_horizontal_sampling) * BLOCK_SIDE,
    );
    let data_block_rows = divide_ceil(
        data_height,
        u32::from(layout.max_vertical_sampling) * BLOCK_SIDE,
    );

    component.plane_width = plane_width;
    component.plane = vec![0; sample_count as usize];
    component.block_columns = block_columns;
    component.block_rows = block_rows;
    component.data_block_columns = data_block_columns;
    component.data_block_rows = data_block_rows;
    if layout.process == CodingProcess::Progressive {
        let block_count = block_columns.checked_mul(block_rows).ok_or_raise(|| {
            JPEGError::ArithmeticOverflow("component coefficient count overflowed")
        })?;
        component.coefficients = vec![[0; 64]; block_count as usize];
    }
    invariant_eq!(component.plane.len(), sample_count as usize);
    Ok(())
}

fn convert_color(first: u8, second: u8, third: u8, transform: ColorTransform) -> [u8; 4] {
    match transform {
        ColorTransform::Rgb => [first, second, third, 255],
        ColorTransform::YCbCr => {
            let luminance = f32::from(first);
            let blue_difference = f32::from(second) - 128.0;
            let red_difference = f32::from(third) - 128.0;
            let red = luminance + 1.402 * red_difference;
            let green = luminance - 0.344_136 * blue_difference - 0.714_136 * red_difference;
            let blue = luminance + 1.772 * blue_difference;
            [clamp_color(red), clamp_color(green), clamp_color(blue), 255]
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "YCbCr conversion values are rounded and clamped to u8 before conversion"
)]
fn clamp_color(value: f32) -> u8 {
    invariant!(value.is_finite());
    value.round().clamp(f32::from(u8::MIN), f32::from(u8::MAX)) as u8
}

pub(super) fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(error(JPEGError::Frame("JPEG dimensions must be nonzero")));
    }

    if width > DIMENSION_MAX || height > DIMENSION_MAX {
        return Err(error(JPEGError::LimitExceeded(JPEGLimit::Dimensions(
            DIMENSION_MAX,
        ))));
    }

    if u64::from(width) * u64::from(height) > PIXELS_MAX {
        return Err(error(JPEGError::LimitExceeded(JPEGLimit::Pixels(
            PIXELS_MAX,
        ))));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component(width: u32, height: u32, value: impl Fn(usize) -> u8) -> FrameComponent {
        let sample_count = usize::try_from(u64::from(width) * u64::from(height))
            .expect("test sample count fits usize");
        FrameComponent {
            identifier: 1,
            horizontal_sampling: 1,
            vertical_sampling: 1,
            quantization_table: 0,
            plane_width: width,
            plane: (0..sample_count).map(value).collect(),
            block_columns: width.div_ceil(BLOCK_SIDE),
            block_rows: height.div_ceil(BLOCK_SIDE),
            data_block_columns: width.div_ceil(BLOCK_SIDE),
            data_block_rows: height.div_ceil(BLOCK_SIDE),
            coefficients: Vec::new(),
        }
    }

    fn frame(width: u32, height: u32, components: Vec<FrameComponent>) -> Frame {
        Frame {
            width,
            height,
            mcu_columns: width.div_ceil(BLOCK_SIDE),
            mcu_rows: height.div_ceil(BLOCK_SIDE),
            max_horizontal_sampling: 1,
            max_vertical_sampling: 1,
            components,
            process: CodingProcess::Sequential,
        }
    }

    fn progressive_component(width: u32, height: u32, coefficients: [i32; 64]) -> FrameComponent {
        let mut component = component(width, height, |_| 0);
        let block_count = usize::try_from(component.block_columns * component.block_rows)
            .expect("test block count fits usize");
        component.coefficients = vec![coefficients; block_count];
        component
    }

    #[test]
    fn parallel_pixel_materialization_preserves_grayscale_order() {
        let width = 512;
        let height = 512;
        let component = component(width, height, |index| {
            u8::try_from(index % 256).expect("test sample fits u8")
        });
        let expected: Vec<_> = component
            .plane
            .iter()
            .copied()
            .flat_map(|sample| [sample, sample, sample, 255])
            .collect();

        let image = frame(width, height, vec![component]).into_image(ColorTransform::Rgb);

        assert_eq!(image.rgba8(), expected);
    }

    #[test]
    fn parallel_progressive_materialization_writes_complete_block_rows() {
        let width = 512;
        let height = 512;
        let mut component = progressive_component(width, height, [0; 64]);
        for (index, coefficients) in component.coefficients.iter_mut().enumerate() {
            coefficients[0] = i32::try_from(index % 16 * 8).expect("test coefficient fits i32");
        }
        let block_columns = component.block_columns;
        let mut frame = frame(width, height, vec![component]);
        frame.process = CodingProcess::Progressive;
        let mut quantization = [None; COMPONENTS_MAX];
        quantization[0] = Some([1; 64]);

        frame.materialize_progressive(&quantization).unwrap();

        assert_eq!(frame.components[0].coefficients.len(), 0);
        for y in 0..height {
            for x in 0..width {
                let block_index = y / BLOCK_SIDE * block_columns + x / BLOCK_SIDE;
                let expected = 128 + u8::try_from(block_index % 16).expect("test sample fits u8");
                let index = usize::try_from(y * width + x).expect("test sample index fits usize");
                assert_eq!(frame.components[0].plane[index], expected);
            }
        }
    }
}
