use super::{
    COMPONENTS_MAX, Frame, FrameComponent, JPEGError, Result, ScanHeader, ZIGZAG_TO_NATURAL,
    dequantize_block, error, idct, write_block,
};
use exn::OptionExt;

const ARITHMETIC_TABLES_MAX: usize = 4;
const DC_STATISTICS_COUNT: usize = 64;
const AC_STATISTICS_COUNT: usize = 256;
const FIXED_STATE_INDEX: u8 = 113;
const RENORMALIZATION_ITERATIONS_MAX: u8 = 17;

const _: () = {
    assert!(ARITHMETIC_TABLES_MAX == 4);
    assert!(DC_STATISTICS_COUNT >= 49);
    assert!(AC_STATISTICS_COUNT >= 245);
    assert!(PROBABILITY_ESTIMATES.len() == FIXED_STATE_INDEX as usize + 1);
};

#[derive(Clone, Copy)]
pub(super) struct DCConditioning {
    pub(super) lower: u8,
    pub(super) upper: u8,
}

#[derive(Clone, Copy)]
pub(super) struct ConditioningTables {
    pub(super) dc: [DCConditioning; ARITHMETIC_TABLES_MAX],
    pub(super) ac: [u8; ARITHMETIC_TABLES_MAX],
}

impl ConditioningTables {
    pub(super) const fn defaults() -> Self {
        let tables = Self {
            dc: [DCConditioning { lower: 0, upper: 1 }; ARITHMETIC_TABLES_MAX],
            ac: [5; ARITHMETIC_TABLES_MAX],
        };
        assert!(tables.dc[0].lower <= tables.dc[0].upper);
        assert!(tables.ac[0] >= 1);
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
    assert!(!plans.is_empty());
    assert!(plans.len() <= COMPONENTS_MAX);

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
    assert!(!plans.is_empty());
    assert!(plans.len() <= COMPONENTS_MAX);

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
    assert!(!plans.is_empty());
    assert!(plans.len() <= COMPONENTS_MAX);

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
    assert!(predictor_index < COMPONENTS_MAX);
    assert!(plan.quantization.iter().all(|value| *value > 0));

    let difference = state.decode_dc_difference(
        plan.dc_table,
        predictor_index,
        conditioning.dc[plan.dc_table],
    )?;
    let predictor = state.dc_predictors[predictor_index].wrapping_add(difference as u16);
    state.dc_predictors[predictor_index] = predictor;
    let mut quantized = [0_i32; 64];
    quantized[0] = i32::from(predictor as i16);
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
    assert_eq!(plans.len(), scan.components.len());
    assert!(!plans.is_empty());

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
    assert!(!plans.is_empty());
    assert!(plans.len() <= COMPONENTS_MAX);

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
    assert!(!plans.is_empty());
    assert!(plans.len() <= COMPONENTS_MAX);

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
    assert!(plans.len() > 1);
    assert!(plans.len() <= COMPONENTS_MAX);

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
    assert!(component.block_columns > 0);
    assert!(component.block_rows > 0);

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
    assert!(scan.spectral_start <= scan.spectral_end);
    assert!(predictor_index < COMPONENTS_MAX);

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
    assert_eq!(scan.spectral_start, 0);
    assert_eq!(scan.successive_high, 0);

    let difference = state.decode_dc_difference(
        plan.dc_table,
        predictor_index,
        conditioning.dc[plan.dc_table],
    )?;
    let predictor = state.dc_predictors[predictor_index].wrapping_add(difference as u16);
    state.dc_predictors[predictor_index] = predictor;
    coefficients[0] = scale(i32::from(predictor as i16), scan.successive_low)?;
    Ok(())
}

fn decode_dc_refinement(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    successive_low: u8,
) -> Result<()> {
    assert!(successive_low <= 13);
    assert!(coefficients[0].checked_abs().is_some());

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
    assert!(spectral_start > 0);
    assert!(spectral_start <= spectral_end);

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
    assert!(scan.spectral_start > 0);
    assert!(scan.successive_high > 0);

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
    assert!(spectral > 0);
    assert!(spectral <= spectral_end);

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
    assert!(*coefficient != 0);
    assert!(correction > 0);

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
    assert!(spectral > 0);
    assert!((1..=63).contains(&conditioning_index));

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
    assert_eq!(magnitude, 2);
    assert!(context == 189 || context == 217);

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
    assert!(context < AC_STATISTICS_COUNT);
    assert!(magnitude < 0x8000);

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
    assert!(successive_low <= 13);
    assert!(value.checked_abs().is_some());
    value
        .checked_mul(1_i32 << successive_low)
        .ok_or_raise(|| JPEGError::ArithmeticOverflow("arithmetic coefficient overflowed"))
}

struct ScanState<'a> {
    decoder: Decoder<'a>,
    dc_statistics: [[u8; DC_STATISTICS_COUNT]; ARITHMETIC_TABLES_MAX],
    ac_statistics: [[u8; AC_STATISTICS_COUNT]; ARITHMETIC_TABLES_MAX],
    dc_predictors: [u16; COMPONENTS_MAX],
    dc_contexts: [u8; COMPONENTS_MAX],
    fixed_state: u8,
}

impl<'a> ScanState<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() <= isize::MAX as usize);
        let state = Self {
            decoder: Decoder::new(bytes),
            dc_statistics: [[0; DC_STATISTICS_COUNT]; ARITHMETIC_TABLES_MAX],
            ac_statistics: [[0; AC_STATISTICS_COUNT]; ARITHMETIC_TABLES_MAX],
            dc_predictors: [0; COMPONENTS_MAX],
            dc_contexts: [0; COMPONENTS_MAX],
            fixed_state: FIXED_STATE_INDEX,
        };
        assert_eq!(state.fixed_state, FIXED_STATE_INDEX);
        assert!(
            state
                .dc_statistics
                .iter()
                .flatten()
                .all(|value| *value == 0)
        );
        state
    }

    fn decode_dc(&mut self, table: usize, context: usize) -> Result<u8> {
        assert!(table < ARITHMETIC_TABLES_MAX);
        assert!(context < DC_STATISTICS_COUNT);
        self.decoder.decode(&mut self.dc_statistics[table][context])
    }

    fn decode_ac(&mut self, table: usize, context: usize) -> Result<u8> {
        assert!(table < ARITHMETIC_TABLES_MAX);
        assert!(context < AC_STATISTICS_COUNT);
        self.decoder.decode(&mut self.ac_statistics[table][context])
    }

    fn decode_fixed(&mut self) -> Result<u8> {
        assert_eq!(self.fixed_state, FIXED_STATE_INDEX);
        let decision = self.decoder.decode(&mut self.fixed_state)?;
        assert_eq!(self.fixed_state, FIXED_STATE_INDEX);
        Ok(decision)
    }

    fn decode_dc_difference(
        &mut self,
        table: usize,
        predictor_index: usize,
        conditioning: DCConditioning,
    ) -> Result<i32> {
        assert!(table < ARITHMETIC_TABLES_MAX);
        assert!(predictor_index < COMPONENTS_MAX);

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
        assert_eq!(magnitude, 1);
        assert_eq!(context, 20);

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
        assert!(context < DC_STATISTICS_COUNT);
        assert!(magnitude < 0x8000);

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
        assert!((0xd0..=0xd7).contains(&expected));
        assert_eq!(self.fixed_state, FIXED_STATE_INDEX);

        self.decoder.restart(expected)?;
        self.dc_statistics.fill([0; DC_STATISTICS_COUNT]);
        self.ac_statistics.fill([0; AC_STATISTICS_COUNT]);
        self.dc_predictors.fill(0);
        self.dc_contexts.fill(0);
        assert!(self.dc_statistics.iter().flatten().all(|value| *value == 0));
        Ok(())
    }
}

fn dc_context(magnitude: u16, sign: u8, conditioning: DCConditioning) -> u8 {
    assert!(sign <= 1);
    assert!(conditioning.lower <= conditioning.upper);

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

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
    code: u32,
    interval: u32,
    bit_count: i8,
    pending_marker: Option<u8>,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        assert!(bytes.len() <= isize::MAX as usize);
        let decoder = Self {
            bytes,
            offset: 0,
            code: 0,
            interval: 0,
            bit_count: -16,
            pending_marker: None,
        };
        assert_eq!(decoder.offset, 0);
        assert_eq!(decoder.bit_count, -16);
        decoder
    }

    fn decode(&mut self, state: &mut u8) -> Result<u8> {
        assert!(usize::from(*state & 0x7f) < PROBABILITY_ESTIMATES.len());
        assert!(*state >> 7 <= 1);

        self.renormalize()?;
        let mps = *state >> 7;
        let estimate = PROBABILITY_ESTIMATES[usize::from(*state & 0x7f)];
        self.interval -= u32::from(estimate.value);
        let mps_interval = self.interval;
        let boundary = self.interval << self.bit_count;
        let decision = if self.code >= boundary {
            self.code -= boundary;
            self.decode_lps_path(state, estimate, mps, mps_interval)
        } else if self.interval < 0x8000 {
            self.decode_mps_exchange(state, estimate, mps)
        } else {
            mps
        };
        assert!(decision <= 1);
        Ok(decision)
    }

    fn decode_lps_path(
        &mut self,
        state: &mut u8,
        estimate: ProbabilityEstimate,
        mps: u8,
        mps_interval: u32,
    ) -> u8 {
        assert!(mps <= 1);
        assert!(mps_interval < 0x1_0000);

        self.interval = u32::from(estimate.value);
        if mps_interval < u32::from(estimate.value) {
            self.update_mps(state, estimate);
            mps
        } else {
            self.update_lps(state, estimate);
            mps ^ 1
        }
    }

    fn decode_mps_exchange(
        &mut self,
        state: &mut u8,
        estimate: ProbabilityEstimate,
        mps: u8,
    ) -> u8 {
        assert!(self.interval < 0x8000);
        assert!(mps <= 1);

        if self.interval < u32::from(estimate.value) {
            self.update_lps(state, estimate);
            mps ^ 1
        } else {
            self.update_mps(state, estimate);
            mps
        }
    }

    fn update_lps(&self, state: &mut u8, estimate: ProbabilityEstimate) {
        let mut mps = *state >> 7;
        assert!(mps <= 1);
        assert!(usize::from(estimate.next_lps) < PROBABILITY_ESTIMATES.len());

        if estimate.switch_mps {
            mps ^= 1;
        }
        *state = (mps << 7) | estimate.next_lps;
    }

    fn update_mps(&self, state: &mut u8, estimate: ProbabilityEstimate) {
        let mps = *state >> 7;
        assert!(mps <= 1);
        assert!(usize::from(estimate.next_mps) < PROBABILITY_ESTIMATES.len());
        *state = (mps << 7) | estimate.next_mps;
    }

    fn renormalize(&mut self) -> Result<()> {
        assert!(self.interval <= 0x1_0000);
        assert!((-16..=7).contains(&self.bit_count));

        let mut iterations = 0_u8;
        while self.interval < 0x8000 {
            iterations += 1;
            if iterations > RENORMALIZATION_ITERATIONS_MAX {
                return Err(error(JPEGError::Entropy(
                    "arithmetic decoder renormalization exceeded its bound",
                )));
            }
            self.bit_count -= 1;
            if self.bit_count < 0 {
                self.byte_in()?;
            }
            self.interval <<= 1;
        }
        assert!(self.interval >= 0x8000);
        Ok(())
    }

    fn byte_in(&mut self) -> Result<()> {
        assert!(self.bit_count < 0);
        assert!(self.offset <= self.bytes.len());

        let byte = self.next_entropy_byte()?;
        self.code = (self.code << 8) | u32::from(byte);
        self.bit_count += 8;
        if self.bit_count < 0 {
            self.bit_count += 1;
            if self.bit_count == 0 {
                self.interval = 0x8000;
            }
        }
        Ok(())
    }

    fn next_entropy_byte(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.pending_marker.is_none() || self.offset >= 2);

        if self.pending_marker.is_some() {
            return Ok(0);
        }
        let byte = self.raw_byte()?;
        if byte != 0xff {
            return Ok(byte);
        }
        let marker = self.marker_suffix()?;
        if marker == 0 {
            return Ok(0xff);
        }
        self.pending_marker = Some(marker);
        Ok(0)
    }

    fn marker_suffix(&mut self) -> Result<u8> {
        assert!(self.offset > 0);
        assert_eq!(self.bytes[self.offset - 1], 0xff);

        let mut marker = self.raw_byte()?;
        while marker == 0xff {
            marker = self.raw_byte()?;
        }
        Ok(marker)
    }

    fn restart(&mut self, expected: u8) -> Result<()> {
        assert!((0xd0..=0xd7).contains(&expected));
        assert!(self.offset <= self.bytes.len());

        let marker = self.take_marker()?;
        if marker != expected {
            return Err(error(JPEGError::RestartMarkerMismatch {
                expected,
                found: marker,
            }));
        }
        self.code = 0;
        self.interval = 0;
        self.bit_count = -16;
        self.pending_marker = None;
        Ok(())
    }

    fn finish(mut self) -> Result<(usize, u8)> {
        assert!(self.offset <= self.bytes.len());
        assert!((-16..=7).contains(&self.bit_count));

        let marker = self.take_marker()?;
        assert!(self.offset <= self.bytes.len());
        Ok((self.offset, marker))
    }

    fn take_marker(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.pending_marker.is_none() || self.offset >= 2);

        if let Some(marker) = self.pending_marker.take() {
            return Ok(marker);
        }
        while self.offset < self.bytes.len() {
            let byte = self.raw_byte()?;
            if byte != 0xff {
                continue;
            }
            let marker = self.marker_suffix()?;
            if marker != 0 {
                return Ok(marker);
            }
        }
        Err(error(JPEGError::UnexpectedEnd(
            "arithmetic entropy data is missing its terminating marker",
        )))
    }

    fn raw_byte(&mut self) -> Result<u8> {
        assert!(self.offset <= self.bytes.len());
        assert!(self.bytes.len() <= isize::MAX as usize);

        let byte = self.bytes.get(self.offset).copied().ok_or_raise(|| {
            JPEGError::UnexpectedEnd("unexpected end of arithmetic entropy data")
        })?;
        self.offset += 1;
        Ok(byte)
    }
}

#[derive(Clone, Copy)]
struct ProbabilityEstimate {
    value: u16,
    next_lps: u8,
    next_mps: u8,
    switch_mps: bool,
}

const fn probability(
    value: u16,
    next_lps: u8,
    next_mps: u8,
    switch_mps: bool,
) -> ProbabilityEstimate {
    ProbabilityEstimate {
        value,
        next_lps,
        next_mps,
        switch_mps,
    }
}

// ITU-T T.81 Table D.3. Keeping the transition table literal avoids arithmetic or indexing that
// could turn a corrupt probability state into an out-of-bounds transition in the decoder hot path.
const PROBABILITY_ESTIMATES: [ProbabilityEstimate; 114] = [
    probability(0x5a1d, 1, 1, true),
    probability(0x2586, 14, 2, false),
    probability(0x1114, 16, 3, false),
    probability(0x080b, 18, 4, false),
    probability(0x03d8, 20, 5, false),
    probability(0x01da, 23, 6, false),
    probability(0x00e5, 25, 7, false),
    probability(0x006f, 28, 8, false),
    probability(0x0036, 30, 9, false),
    probability(0x001a, 33, 10, false),
    probability(0x000d, 35, 11, false),
    probability(0x0006, 9, 12, false),
    probability(0x0003, 10, 13, false),
    probability(0x0001, 12, 13, false),
    probability(0x5a7f, 15, 15, true),
    probability(0x3f25, 36, 16, false),
    probability(0x2cf2, 38, 17, false),
    probability(0x207c, 39, 18, false),
    probability(0x17b9, 40, 19, false),
    probability(0x1182, 42, 20, false),
    probability(0x0cef, 43, 21, false),
    probability(0x09a1, 45, 22, false),
    probability(0x072f, 46, 23, false),
    probability(0x055c, 48, 24, false),
    probability(0x0406, 49, 25, false),
    probability(0x0303, 51, 26, false),
    probability(0x0240, 52, 27, false),
    probability(0x01b1, 54, 28, false),
    probability(0x0144, 56, 29, false),
    probability(0x00f5, 57, 30, false),
    probability(0x00b7, 59, 31, false),
    probability(0x008a, 60, 32, false),
    probability(0x0068, 62, 33, false),
    probability(0x004e, 63, 34, false),
    probability(0x003b, 32, 35, false),
    probability(0x002c, 33, 9, false),
    probability(0x5ae1, 37, 37, true),
    probability(0x484c, 64, 38, false),
    probability(0x3a0d, 65, 39, false),
    probability(0x2ef1, 67, 40, false),
    probability(0x261f, 68, 41, false),
    probability(0x1f33, 69, 42, false),
    probability(0x19a8, 70, 43, false),
    probability(0x1518, 72, 44, false),
    probability(0x1177, 73, 45, false),
    probability(0x0e74, 74, 46, false),
    probability(0x0bfb, 75, 47, false),
    probability(0x09f8, 77, 48, false),
    probability(0x0861, 78, 49, false),
    probability(0x0706, 79, 50, false),
    probability(0x05cd, 48, 51, false),
    probability(0x04de, 50, 52, false),
    probability(0x040f, 50, 53, false),
    probability(0x0363, 51, 54, false),
    probability(0x02d4, 52, 55, false),
    probability(0x025c, 53, 56, false),
    probability(0x01f8, 54, 57, false),
    probability(0x01a4, 55, 58, false),
    probability(0x0160, 56, 59, false),
    probability(0x0125, 57, 60, false),
    probability(0x00f6, 58, 61, false),
    probability(0x00cb, 59, 62, false),
    probability(0x00ab, 61, 63, false),
    probability(0x008f, 61, 32, false),
    probability(0x5b12, 65, 65, true),
    probability(0x4d04, 80, 66, false),
    probability(0x412c, 81, 67, false),
    probability(0x37d8, 82, 68, false),
    probability(0x2fe8, 83, 69, false),
    probability(0x293c, 84, 70, false),
    probability(0x2379, 86, 71, false),
    probability(0x1edf, 87, 72, false),
    probability(0x1aa9, 87, 73, false),
    probability(0x174e, 72, 74, false),
    probability(0x1424, 72, 75, false),
    probability(0x119c, 74, 76, false),
    probability(0x0f6b, 74, 77, false),
    probability(0x0d51, 75, 78, false),
    probability(0x0bb6, 77, 79, false),
    probability(0x0a40, 77, 48, false),
    probability(0x5832, 80, 81, true),
    probability(0x4d1c, 88, 82, false),
    probability(0x438e, 89, 83, false),
    probability(0x3bdd, 90, 84, false),
    probability(0x34ee, 91, 85, false),
    probability(0x2eae, 92, 86, false),
    probability(0x299a, 93, 87, false),
    probability(0x2516, 86, 71, false),
    probability(0x5570, 88, 89, true),
    probability(0x4ca9, 95, 90, false),
    probability(0x44d9, 96, 91, false),
    probability(0x3e22, 97, 92, false),
    probability(0x3824, 99, 93, false),
    probability(0x32b4, 99, 94, false),
    probability(0x2e17, 93, 86, false),
    probability(0x56a8, 95, 96, true),
    probability(0x4f46, 101, 97, false),
    probability(0x47e5, 102, 98, false),
    probability(0x41cf, 103, 99, false),
    probability(0x3c3d, 104, 100, false),
    probability(0x375e, 99, 93, false),
    probability(0x5231, 105, 102, false),
    probability(0x4c0f, 106, 103, false),
    probability(0x4639, 107, 104, false),
    probability(0x415e, 103, 99, false),
    probability(0x5627, 105, 106, true),
    probability(0x50e7, 108, 107, false),
    probability(0x4b85, 109, 103, false),
    probability(0x5597, 110, 109, false),
    probability(0x504f, 111, 107, false),
    probability(0x5a10, 110, 111, true),
    probability(0x5522, 112, 109, false),
    probability(0x59eb, 112, 111, true),
    probability(0x5a1d, 113, 113, false),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_standard_arithmetic_test_sequence() {
        // ITU-T T.81 K.4.1 supplies both this entropy segment and the 256 binary decisions. The
        // appended EOI marker also exercises the arithmetic decoder's normative zero-fill path.
        let entropy = [
            0x65, 0x5b, 0x51, 0x44, 0xf7, 0x96, 0x9d, 0x51, 0x78, 0x55, 0xbf, 0xff, 0x00, 0xfc,
            0x51, 0x84, 0xc7, 0xce, 0xf9, 0x39, 0x00, 0x28, 0x7d, 0x46, 0x70, 0x8e, 0xcb, 0xc0,
            0xf6, 0xff, 0xd9, 0x00,
        ];
        let expected = [
            0x00, 0x02, 0x00, 0x51, 0x00, 0x00, 0x00, 0xc0, 0x03, 0x52, 0x87, 0x2a, 0xaa, 0xaa,
            0xaa, 0xaa, 0x82, 0xc0, 0x20, 0x00, 0xfc, 0xd7, 0x9e, 0xf6, 0x74, 0xea, 0xab, 0xf7,
            0x69, 0x7e, 0xe7, 0x4c,
        ];
        let mut decoder = Decoder::new(&entropy);
        let mut state = 0_u8;
        for bit_index in 0..256 {
            let expected_bit = (expected[bit_index / 8] >> (7 - bit_index % 8)) & 1;
            assert_eq!(decoder.decode(&mut state).unwrap(), expected_bit);
        }
        assert_eq!(decoder.finish().unwrap(), (31, 0xd9));
    }
}
