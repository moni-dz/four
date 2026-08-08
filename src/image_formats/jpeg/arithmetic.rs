mod decoder;

use super::{
    COMPONENTS_MAX, Frame, FrameComponent, JPEGError, Result, ScanHeader, ZIGZAG_TO_NATURAL,
    dequantize_block, error, idct, write_block,
};
use decoder::Decoder;
use exn::OptionExt;

pub(super) const TABLES_MAX: usize = 4;
const DC_STATISTICS_COUNT: usize = 64;
const AC_STATISTICS_COUNT: usize = 256;
const FIXED_STATE_INDEX: u8 = 113;

#[derive(Clone, Copy)]
pub(super) struct DCConditioning {
    pub(super) lower: u8,
    pub(super) upper: u8,
}

#[derive(Clone, Copy)]
pub(super) struct ConditioningTables {
    pub(super) dc: [DCConditioning; TABLES_MAX],
    pub(super) ac: [u8; TABLES_MAX],
}

impl ConditioningTables {
    pub(super) const fn defaults() -> Self {
        let tables = Self {
            dc: [DCConditioning { lower: 0, upper: 1 }; TABLES_MAX],
            ac: [5; TABLES_MAX],
        };
        invariant!(tables.dc[0].lower <= tables.dc[0].upper);
        invariant!(tables.ac[0] >= 1);
        tables
    }
}

pub(super) struct SequentialPlan {
    pub(super) frame_index: usize,
    pub(super) horizontal_sampling: u8,
    pub(super) vertical_sampling: u8,
    pub(super) quantization: [u16; 64],
    pub(super) dc_table: usize,
    pub(super) ac_table: usize,
}

pub(super) struct ProgressivePlan {
    pub(super) frame_index: usize,
    pub(super) horizontal_sampling: u8,
    pub(super) vertical_sampling: u8,
    pub(super) dc_table: usize,
    pub(super) ac_table: usize,
}

pub(super) fn decode_sequential(
    entropy: &[u8],
    frame: &mut Frame,
    plans: &[SequentialPlan],
    conditioning: &ConditioningTables,
    restart_interval: u32,
) -> Result<(usize, u8)> {
    invariant!(!plans.is_empty());
    invariant!(plans.len() <= COMPONENTS_MAX);

    let (mcu_columns, mcu_rows) = sequential_scan_dimensions(frame, plans);
    let mcu_count = mcu_columns
        .checked_mul(mcu_rows)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("arithmetic MCU count overflowed"))?;

    let mut state = ScanState::new(entropy);
    let mut restart_index = 0_u8;

    for mcu_index in 0..mcu_count {
        let mcu_x = mcu_index % mcu_columns;
        let mcu_y = mcu_index / mcu_columns;
        decode_sequential_mcu(&mut state, frame, plans, conditioning, mcu_x, mcu_y)?;

        let completed = mcu_index + 1;
        if restart_interval > 0 && completed < mcu_count && completed % restart_interval == 0 {
            state.restart(0xd0 + restart_index)?;
            restart_index = (restart_index + 1) & 7;
        }
    }

    state.decoder.finish()
}

fn sequential_scan_dimensions(frame: &Frame, plans: &[SequentialPlan]) -> (u32, u32) {
    invariant!(!plans.is_empty());
    invariant!(plans.len() <= COMPONENTS_MAX);

    if plans.len() > 1 {
        (frame.mcu_columns, frame.mcu_rows)
    } else {
        let component = &frame.components[plans[0].frame_index];
        (component.data_block_columns, component.data_block_rows)
    }
}

fn decode_sequential_mcu(
    state: &mut ScanState<'_>,
    frame: &mut Frame,
    plans: &[SequentialPlan],
    conditioning: &ConditioningTables,
    mcu_x: u32,
    mcu_y: u32,
) -> Result<()> {
    invariant!(!plans.is_empty());
    invariant!(plans.len() <= COMPONENTS_MAX);

    if plans.len() == 1 {
        let plan = &plans[0];
        let samples = decode_sequential_block(state, plan, conditioning, 0)?;
        let component = &mut frame.components[plan.frame_index];
        write_block(component, mcu_x, mcu_y, &samples);
        return Ok(());
    }

    for (predictor_index, plan) in plans.iter().enumerate() {
        for block_y in 0..plan.vertical_sampling {
            for block_x in 0..plan.horizontal_sampling {
                let samples = decode_sequential_block(state, plan, conditioning, predictor_index)?;
                let component = &mut frame.components[plan.frame_index];
                let x = mcu_x * u32::from(plan.horizontal_sampling) + u32::from(block_x);
                let y = mcu_y * u32::from(plan.vertical_sampling) + u32::from(block_y);
                write_block(component, x, y, &samples);
            }
        }
    }

    Ok(())
}

fn decode_sequential_block(
    state: &mut ScanState<'_>,
    plan: &SequentialPlan,
    conditioning: &ConditioningTables,
    predictor_index: usize,
) -> Result<[u8; 64]> {
    invariant!(predictor_index < COMPONENTS_MAX);
    invariant!(plan.quantization.iter().all(|value| *value > 0));

    let difference = state.decode_dc_difference(
        plan.dc_table,
        predictor_index,
        conditioning.dc[plan.dc_table],
    )?;

    let predictor = wrapping_predictor(state.dc_predictors[predictor_index], difference);
    state.dc_predictors[predictor_index] = predictor;

    let mut quantized = [0_i32; 64];
    quantized[0] = i32::from(predictor.cast_signed());

    decode_ac_band(
        state,
        &mut quantized,
        plan.ac_table,
        conditioning.ac[plan.ac_table],
        1,
        63,
        0,
    )?;

    let coefficients = dequantize_block(&quantized, &plan.quantization)?;
    // Keep the SIMD transform local: arithmetic state orders block decoding, and scheduling an
    // individual 8x8 IDCT through Rayon costs more than the transform itself.
    Ok(idct::inverse(&coefficients))
}

pub(super) fn decode_progressive(
    entropy: &[u8],
    frame: &mut Frame,
    plans: &[ProgressivePlan],
    scan: &ScanHeader,
    conditioning: &ConditioningTables,
    restart_interval: u32,
) -> Result<(usize, u8)> {
    invariant_eq!(plans.len(), scan.components.len());
    invariant!(!plans.is_empty());

    let (mcu_columns, mcu_rows) = progressive_scan_dimensions(frame, plans);

    let mcu_count = mcu_columns
        .checked_mul(mcu_rows)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("arithmetic MCU count overflowed"))?;

    let mut state = ScanState::new(entropy);
    let mut restart_index = 0_u8;

    for mcu_index in 0..mcu_count {
        let mcu_x = mcu_index % mcu_columns;
        let mcu_y = mcu_index / mcu_columns;
        decode_progressive_mcu(&mut state, frame, plans, scan, conditioning, mcu_x, mcu_y)?;

        let completed = mcu_index + 1;

        if restart_interval > 0 && completed < mcu_count && completed % restart_interval == 0 {
            state.restart(0xd0 + restart_index)?;
            restart_index = (restart_index + 1) & 7;
        }
    }

    state.decoder.finish()
}

fn progressive_scan_dimensions(frame: &Frame, plans: &[ProgressivePlan]) -> (u32, u32) {
    invariant!(!plans.is_empty());
    invariant!(plans.len() <= COMPONENTS_MAX);

    if plans.len() > 1 {
        (frame.mcu_columns, frame.mcu_rows)
    } else {
        let component = &frame.components[plans[0].frame_index];
        (component.data_block_columns, component.data_block_rows)
    }
}

fn decode_progressive_mcu(
    state: &mut ScanState<'_>,
    frame: &mut Frame,
    plans: &[ProgressivePlan],
    scan: &ScanHeader,
    conditioning: &ConditioningTables,
    mcu_x: u32,
    mcu_y: u32,
) -> Result<()> {
    invariant!(!plans.is_empty());
    invariant!(plans.len() <= COMPONENTS_MAX);

    if plans.len() == 1 {
        let plan = &plans[0];
        let component = &mut frame.components[plan.frame_index];
        let index = coefficient_index(component, mcu_x, mcu_y)?;
        return decode_progressive_block(
            state,
            &mut component.coefficients[index],
            plan,
            scan,
            conditioning,
            0,
        );
    }
    decode_progressive_interleaved(state, frame, plans, scan, conditioning, mcu_x, mcu_y)
}

fn decode_progressive_interleaved(
    state: &mut ScanState<'_>,
    frame: &mut Frame,
    plans: &[ProgressivePlan],
    scan: &ScanHeader,
    conditioning: &ConditioningTables,
    mcu_x: u32,
    mcu_y: u32,
) -> Result<()> {
    invariant!(plans.len() > 1);
    invariant!(plans.len() <= COMPONENTS_MAX);

    for (predictor_index, plan) in plans.iter().enumerate() {
        let component = &mut frame.components[plan.frame_index];
        for block_y in 0..plan.vertical_sampling {
            for block_x in 0..plan.horizontal_sampling {
                let x = mcu_x * u32::from(plan.horizontal_sampling) + u32::from(block_x);
                let y = mcu_y * u32::from(plan.vertical_sampling) + u32::from(block_y);
                let index = coefficient_index(component, x, y)?;
                decode_progressive_block(
                    state,
                    &mut component.coefficients[index],
                    plan,
                    scan,
                    conditioning,
                    predictor_index,
                )?;
            }
        }
    }
    Ok(())
}

fn coefficient_index(component: &FrameComponent, block_x: u32, block_y: u32) -> Result<usize> {
    invariant!(component.block_columns > 0);
    invariant!(component.block_rows > 0);

    if block_x >= component.block_columns || block_y >= component.block_rows {
        return Err(error(JPEGError::Scan(
            "arithmetic progressive block coordinate is out of range",
        )));
    }

    let index = block_y
        .checked_mul(component.block_columns)
        .and_then(|value| value.checked_add(block_x))
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("arithmetic block index overflowed"))?;
    if index as usize >= component.coefficients.len() {
        return Err(error(JPEGError::Scan(
            "arithmetic progressive coefficient block is missing",
        )));
    }

    Ok(index as usize)
}

fn decode_progressive_block(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    plan: &ProgressivePlan,
    scan: &ScanHeader,
    conditioning: &ConditioningTables,
    predictor_index: usize,
) -> Result<()> {
    invariant!(scan.spectral_start <= scan.spectral_end);
    invariant!(predictor_index < COMPONENTS_MAX);

    if scan.spectral_start == 0 {
        if scan.successive_high == 0 {
            decode_dc_first(
                state,
                coefficients,
                plan,
                scan,
                conditioning,
                predictor_index,
            )
        } else {
            decode_dc_refinement(state, coefficients, scan.successive_low)
        }
    } else if scan.successive_high == 0 {
        decode_ac_band(
            state,
            coefficients,
            plan.ac_table,
            conditioning.ac[plan.ac_table],
            scan.spectral_start,
            scan.spectral_end,
            scan.successive_low,
        )
    } else {
        decode_ac_refinement(state, coefficients, plan.ac_table, scan)
    }
}

fn decode_dc_first(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    plan: &ProgressivePlan,
    scan: &ScanHeader,
    conditioning: &ConditioningTables,
    predictor_index: usize,
) -> Result<()> {
    invariant_eq!(scan.spectral_start, 0);
    invariant_eq!(scan.successive_high, 0);

    let difference = state.decode_dc_difference(
        plan.dc_table,
        predictor_index,
        conditioning.dc[plan.dc_table],
    )?;

    let predictor = wrapping_predictor(state.dc_predictors[predictor_index], difference);

    state.dc_predictors[predictor_index] = predictor;
    coefficients[0] = scale(i32::from(predictor.cast_signed()), scan.successive_low)?;

    Ok(())
}

fn decode_dc_refinement(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    successive_low: u8,
) -> Result<()> {
    invariant!(successive_low <= 13);
    invariant!(coefficients[0].checked_abs().is_some());

    if state.decode_fixed()? != 0 {
        coefficients[0] |= 1_i32 << successive_low;
    }

    Ok(())
}

fn decode_ac_band(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    table: usize,
    conditioning_index: u8,
    spectral_start: u8,
    spectral_end: u8,
    successive_low: u8,
) -> Result<()> {
    invariant!(spectral_start > 0);
    invariant!(spectral_start <= spectral_end);

    let mut spectral = spectral_start;
    while spectral <= spectral_end {
        let mut context = 3 * (usize::from(spectral) - 1);

        if state.decode_ac(table, context)? != 0 {
            break;
        }

        while state.decode_ac(table, context + 1)? == 0 {
            spectral += 1;
            context += 3;
            if spectral > spectral_end {
                return Err(error(JPEGError::Entropy(
                    "arithmetic AC zero run extends past its band",
                )));
            }
        }

        let value = decode_ac_value(state, table, context + 2, spectral, conditioning_index)?;

        coefficients[ZIGZAG_TO_NATURAL[usize::from(spectral)]] = scale(value, successive_low)?;
        spectral += 1;
    }

    Ok(())
}

fn decode_ac_refinement(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    table: usize,
    scan: &ScanHeader,
) -> Result<()> {
    invariant!(scan.spectral_start > 0);
    invariant!(scan.successive_high > 0);

    let correction = 1_i32 << scan.successive_low;
    let mut previous_end = scan.spectral_end;

    while previous_end > 0 {
        if coefficients[ZIGZAG_TO_NATURAL[usize::from(previous_end)]] != 0 {
            break;
        }
        previous_end -= 1;
    }

    let mut spectral = scan.spectral_start;

    while spectral <= scan.spectral_end {
        let context = 3 * (usize::from(spectral) - 1);
        if spectral > previous_end && state.decode_ac(table, context)? != 0 {
            break;
        }
        spectral = refine_coefficient_run(
            state,
            coefficients,
            table,
            context,
            spectral,
            scan.spectral_end,
            correction,
        )?;
    }

    Ok(())
}

fn refine_coefficient_run(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    table: usize,
    mut context: usize,
    mut spectral: u8,
    spectral_end: u8,
    correction: i32,
) -> Result<u8> {
    invariant!(spectral > 0);
    invariant!(spectral <= spectral_end);

    loop {
        let coefficient = &mut coefficients[ZIGZAG_TO_NATURAL[usize::from(spectral)]];

        if *coefficient != 0 {
            if state.decode_ac(table, context + 2)? != 0 {
                refine_nonzero(coefficient, correction)?;
            }
            return Ok(spectral + 1);
        }

        if state.decode_ac(table, context + 1)? != 0 {
            *coefficient = if state.decode_fixed()? != 0 {
                -correction
            } else {
                correction
            };
            return Ok(spectral + 1);
        }

        spectral += 1;
        context += 3;

        if spectral > spectral_end {
            return Err(error(JPEGError::Entropy(
                "arithmetic AC refinement run extends past its band",
            )));
        }
    }
}

fn refine_nonzero(coefficient: &mut i32, correction: i32) -> Result<()> {
    invariant!(*coefficient != 0);
    invariant!(correction > 0);

    let delta = if *coefficient < 0 {
        -correction
    } else {
        correction
    };

    *coefficient = coefficient
        .checked_add(delta)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("arithmetic AC refinement overflowed"))?;

    Ok(())
}

fn decode_ac_value(
    state: &mut ScanState<'_>,
    table: usize,
    mut context: usize,
    spectral: u8,
    conditioning_index: u8,
) -> Result<i32> {
    invariant!(spectral > 0);
    invariant!((1..=63).contains(&conditioning_index));

    let sign = state.decode_fixed()?;
    let mut magnitude = u16::from(state.decode_ac(table, context)?);

    if magnitude != 0 && state.decode_ac(table, context)? != 0 {
        magnitude <<= 1;
        context = if spectral <= conditioning_index {
            189
        } else {
            217
        };
        (magnitude, context) = decode_magnitude_category_ac(state, table, context, magnitude)?;
    }

    let value = decode_magnitude_bits_ac(state, table, context + 14, magnitude)?;
    let signed = i32::from(value) + 1;

    if sign == 0 { Ok(signed) } else { Ok(-signed) }
}

fn decode_magnitude_category_ac(
    state: &mut ScanState<'_>,
    table: usize,
    mut context: usize,
    mut magnitude: u16,
) -> Result<(u16, usize)> {
    invariant_eq!(magnitude, 2);
    invariant!(context == 189 || context == 217);

    let mut decisions = 0_u8;

    while state.decode_ac(table, context)? != 0 {
        magnitude <<= 1;
        decisions += 1;
        if magnitude == 0x8000 || decisions > 13 {
            return Err(error(JPEGError::Entropy(
                "arithmetic AC coefficient magnitude overflowed",
            )));
        }
        context += 1;
    }

    Ok((magnitude, context))
}

fn decode_magnitude_bits_ac(
    state: &mut ScanState<'_>,
    table: usize,
    context: usize,
    mut magnitude: u16,
) -> Result<u16> {
    invariant!(context < AC_STATISTICS_COUNT);
    invariant!(magnitude < 0x8000);

    let mut value = magnitude;
    while magnitude > 1 {
        magnitude >>= 1;
        if state.decode_ac(table, context)? != 0 {
            value |= magnitude;
        }
    }

    Ok(value)
}

fn scale(value: i32, successive_low: u8) -> Result<i32> {
    invariant!(successive_low <= 13);
    invariant!(value.checked_abs().is_some());

    value
        .checked_mul(1_i32 << successive_low)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("arithmetic coefficient overflowed"))
}

// Arithmetic JPEG defines the DC predictor as a wrapping 16-bit signed quantity. Keeping its bits
// in u16 makes the wrap explicit while this conversion preserves negative differences exactly.
fn wrapping_predictor(predictor: u16, difference: i32) -> u16 {
    invariant!((-32_768..=32_768).contains(&difference));

    let difference_bits = u16::try_from(difference.rem_euclid(1 << 16))
        .expect("a value reduced modulo 2^16 always fits u16");

    predictor.wrapping_add(difference_bits)
}

struct ScanState<'a> {
    decoder: Decoder<'a>,
    dc_statistics: [[u8; DC_STATISTICS_COUNT]; TABLES_MAX],
    ac_statistics: [[u8; AC_STATISTICS_COUNT]; TABLES_MAX],
    dc_predictors: [u16; COMPONENTS_MAX],
    dc_contexts: [u8; COMPONENTS_MAX],
    fixed_state: u8,
}

impl<'a> ScanState<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        invariant!(isize::try_from(bytes.len()).is_ok());
        let state = Self {
            decoder: Decoder::new(bytes),
            dc_statistics: [[0; DC_STATISTICS_COUNT]; TABLES_MAX],
            ac_statistics: [[0; AC_STATISTICS_COUNT]; TABLES_MAX],
            dc_predictors: [0; COMPONENTS_MAX],
            dc_contexts: [0; COMPONENTS_MAX],
            fixed_state: FIXED_STATE_INDEX,
        };
        invariant_eq!(state.fixed_state, FIXED_STATE_INDEX);
        invariant!(
            state
                .dc_statistics
                .iter()
                .flatten()
                .all(|value| *value == 0)
        );
        state
    }

    fn decode_dc(&mut self, table: usize, context: usize) -> Result<u8> {
        invariant!(table < TABLES_MAX);
        invariant!(context < DC_STATISTICS_COUNT);
        self.decoder.decode(&mut self.dc_statistics[table][context])
    }

    fn decode_ac(&mut self, table: usize, context: usize) -> Result<u8> {
        invariant!(table < TABLES_MAX);
        invariant!(context < AC_STATISTICS_COUNT);
        self.decoder.decode(&mut self.ac_statistics[table][context])
    }

    fn decode_fixed(&mut self) -> Result<u8> {
        invariant_eq!(self.fixed_state, FIXED_STATE_INDEX);
        let decision = self.decoder.decode(&mut self.fixed_state)?;
        invariant_eq!(self.fixed_state, FIXED_STATE_INDEX);
        Ok(decision)
    }

    fn decode_dc_difference(
        &mut self,
        table: usize,
        predictor_index: usize,
        conditioning: DCConditioning,
    ) -> Result<i32> {
        invariant!(table < TABLES_MAX);
        invariant!(predictor_index < COMPONENTS_MAX);

        let context_base = usize::from(self.dc_contexts[predictor_index]);
        if self.decode_dc(table, context_base)? == 0 {
            self.dc_contexts[predictor_index] = 0;
            return Ok(0);
        }
        let sign = self.decode_dc(table, context_base + 1)?;
        let mut context = context_base + 2 + usize::from(sign);
        let mut magnitude = u16::from(self.decode_dc(table, context)?);
        if magnitude != 0 {
            context = 20;
            (magnitude, context) = self.decode_dc_magnitude_category(table, context, magnitude)?;
        }
        self.dc_contexts[predictor_index] = dc_context(magnitude, sign, conditioning);
        let value = self.decode_dc_magnitude_bits(table, context + 14, magnitude)?;
        let signed = i32::from(value) + 1;
        if sign == 0 { Ok(signed) } else { Ok(-signed) }
    }

    fn decode_dc_magnitude_category(
        &mut self,
        table: usize,
        mut context: usize,
        mut magnitude: u16,
    ) -> Result<(u16, usize)> {
        invariant_eq!(magnitude, 1);
        invariant_eq!(context, 20);

        let mut decisions = 0_u8;
        while self.decode_dc(table, context)? != 0 {
            magnitude <<= 1;
            decisions += 1;
            if magnitude == 0x8000 || decisions > 14 {
                return Err(error(JPEGError::Entropy(
                    "arithmetic DC coefficient magnitude overflowed",
                )));
            }
            context += 1;
        }
        Ok((magnitude, context))
    }

    fn decode_dc_magnitude_bits(
        &mut self,
        table: usize,
        context: usize,
        mut magnitude: u16,
    ) -> Result<u16> {
        invariant!(context < DC_STATISTICS_COUNT);
        invariant!(magnitude < 0x8000);

        let mut value = magnitude;
        while magnitude > 1 {
            magnitude >>= 1;
            if self.decode_dc(table, context)? != 0 {
                value |= magnitude;
            }
        }
        Ok(value)
    }

    fn restart(&mut self, expected: u8) -> Result<()> {
        invariant!((0xd0..=0xd7).contains(&expected));
        invariant_eq!(self.fixed_state, FIXED_STATE_INDEX);

        self.decoder.restart(expected)?;
        self.dc_statistics.fill([0; DC_STATISTICS_COUNT]);
        self.ac_statistics.fill([0; AC_STATISTICS_COUNT]);
        self.dc_predictors.fill(0);
        self.dc_contexts.fill(0);
        invariant!(self.dc_statistics.iter().flatten().all(|value| *value == 0));
        Ok(())
    }
}

fn dc_context(magnitude: u16, sign: u8, conditioning: DCConditioning) -> u8 {
    invariant!(sign <= 1);
    invariant!(conditioning.lower <= conditioning.upper);

    let lower = (1_u32 << conditioning.lower) >> 1;
    let upper = (1_u32 << conditioning.upper) >> 1;
    let sign_offset = sign * 4;
    if u32::from(magnitude) < lower {
        0
    } else if u32::from(magnitude) > upper {
        12 + sign_offset
    } else {
        4 + sign_offset
    }
}
