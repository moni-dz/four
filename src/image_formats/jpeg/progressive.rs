use super::reader::BitReader;
use super::{
    COMPONENTS_MAX, Error, Frame, FrameComponent, ProgressivePlan, Result, ScanHeader,
    ZIGZAG_TO_NATURAL, receive_extend,
};

struct ScanState<'a> {
    reader: BitReader<'a>,
    dc_predictors: [i32; COMPONENTS_MAX],
    eob_run: u32,
}

pub(super) fn decode_scan(
    entropy: &[u8],
    frame: &mut Frame,
    plans: &[ProgressivePlan],
    scan: &ScanHeader,
    restart_interval: u32,
) -> Result<(usize, u8)> {
    assert_eq!(plans.len(), scan.components.len());
    assert!(!plans.is_empty());

    let (mcu_columns, mcu_rows) = scan_dimensions(frame, plans);
    let mcu_count = mcu_columns
        .checked_mul(mcu_rows)
        .ok_or_else(|| Error::new("progressive MCU count overflowed"))?;
    let mut state = ScanState {
        reader: BitReader::new(entropy),
        dc_predictors: [0; COMPONENTS_MAX],
        eob_run: 0,
    };
    let mut restart_index = 0_u8;
    for mcu_index in 0..mcu_count {
        let mcu_x = mcu_index % mcu_columns;
        let mcu_y = mcu_index / mcu_columns;
        decode_mcu(&mut state, frame, plans, scan, mcu_x, mcu_y)?;

        let completed = mcu_index + 1;
        if restart_interval > 0 && completed < mcu_count && completed % restart_interval == 0 {
            state.reader.restart(0xd0 + restart_index)?;
            state.dc_predictors.fill(0);
            state.eob_run = 0;
            restart_index = (restart_index + 1) & 7;
        }
    }
    if state.eob_run != 0 {
        return Err(Error::new("progressive EOB run extends past the scan"));
    }
    state.reader.finish()
}

fn scan_dimensions(frame: &Frame, plans: &[ProgressivePlan]) -> (u32, u32) {
    assert!(!plans.is_empty());
    assert!(plans.len() <= COMPONENTS_MAX);

    if plans.len() > 1 {
        (frame.mcu_columns, frame.mcu_rows)
    } else {
        let component = &frame.components[plans[0].frame_index];
        (component.data_block_columns, component.data_block_rows)
    }
}

fn decode_mcu(
    state: &mut ScanState<'_>,
    frame: &mut Frame,
    plans: &[ProgressivePlan],
    scan: &ScanHeader,
    mcu_x: u32,
    mcu_y: u32,
) -> Result<()> {
    assert!(!plans.is_empty());
    assert!(plans.len() <= COMPONENTS_MAX);

    if plans.len() == 1 {
        let plan = &plans[0];
        let component = &mut frame.components[plan.frame_index];
        let index = coefficient_index(component, mcu_x, mcu_y)?;
        return decode_block(state, &mut component.coefficients[index], plan, scan, 0);
    }

    for (scan_index, plan) in plans.iter().enumerate() {
        let component = &mut frame.components[plan.frame_index];
        for block_y in 0..plan.vertical_sampling {
            for block_x in 0..plan.horizontal_sampling {
                let x = mcu_x * u32::from(plan.horizontal_sampling) + u32::from(block_x);
                let y = mcu_y * u32::from(plan.vertical_sampling) + u32::from(block_y);
                let index = coefficient_index(component, x, y)?;
                decode_block(
                    state,
                    &mut component.coefficients[index],
                    plan,
                    scan,
                    scan_index,
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
        return Err(Error::new("progressive block coordinate is out of range"));
    }
    let index = block_y
        .checked_mul(component.block_columns)
        .and_then(|value| value.checked_add(block_x))
        .ok_or_else(|| Error::new("progressive block index overflowed"))?;
    if index as usize >= component.coefficients.len() {
        return Err(Error::new("progressive coefficient block is missing"));
    }
    Ok(index as usize)
}

fn decode_block(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    plan: &ProgressivePlan,
    scan: &ScanHeader,
    predictor_index: usize,
) -> Result<()> {
    assert!(predictor_index < COMPONENTS_MAX);
    assert!(scan.spectral_start <= scan.spectral_end);

    if scan.spectral_start == 0 {
        if scan.successive_high == 0 {
            decode_dc_first(
                state,
                coefficients,
                plan,
                scan.successive_low,
                predictor_index,
            )
        } else {
            decode_dc_refinement(state, coefficients, scan.successive_low)
        }
    } else if scan.successive_high == 0 {
        decode_ac_first(state, coefficients, plan, scan)
    } else {
        decode_ac_refinement(state, coefficients, plan, scan)
    }
}

fn decode_dc_first(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    plan: &ProgressivePlan,
    successive_low: u8,
    predictor_index: usize,
) -> Result<()> {
    assert!(successive_low <= 13);
    assert!(predictor_index < COMPONENTS_MAX);

    let table = plan
        .dc
        .as_ref()
        .ok_or_else(|| Error::new("progressive DC Huffman table is missing"))?;
    let category = table.decode(&mut state.reader)?;
    if category > 11 {
        return Err(Error::new(
            "progressive DC coefficient category exceeds 11 bits",
        ));
    }
    let difference = receive_extend(&mut state.reader, category)?;
    let predictor = state.dc_predictors[predictor_index]
        .checked_add(difference)
        .ok_or_else(|| Error::new("progressive DC predictor overflowed"))?;
    state.dc_predictors[predictor_index] = predictor;
    coefficients[0] = scale_coefficient(predictor, successive_low)?;
    Ok(())
}

fn decode_dc_refinement(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    successive_low: u8,
) -> Result<()> {
    assert!(successive_low <= 13);
    assert!(coefficients[0].checked_abs().is_some());

    if state.reader.read_bits(1)? != 0 {
        coefficients[0] |= 1_i32 << successive_low;
    }
    Ok(())
}

fn decode_ac_first(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    plan: &ProgressivePlan,
    scan: &ScanHeader,
) -> Result<()> {
    assert!(scan.spectral_start > 0);
    assert!(scan.successive_high == 0);

    if state.eob_run > 0 {
        state.eob_run -= 1;
        return Ok(());
    }
    let table = plan
        .ac
        .as_ref()
        .ok_or_else(|| Error::new("progressive AC Huffman table is missing"))?;
    let mut spectral = scan.spectral_start;
    while spectral <= scan.spectral_end {
        let symbol = table.decode(&mut state.reader)?;
        let zero_run = symbol >> 4;
        let category = symbol & 0x0f;
        if category > 0 {
            spectral = spectral
                .checked_add(zero_run)
                .ok_or_else(|| Error::new("progressive AC run overflowed"))?;
            if spectral > scan.spectral_end {
                return Err(Error::new("progressive AC run extends past its band"));
            }
            let value = receive_extend(&mut state.reader, category)?;
            coefficients[ZIGZAG_TO_NATURAL[usize::from(spectral)]] =
                scale_coefficient(value, scan.successive_low)?;
            spectral += 1;
        } else if zero_run == 15 {
            if u16::from(spectral) + 15 > u16::from(scan.spectral_end) {
                return Err(Error::new("progressive ZRL extends past its band"));
            }
            spectral += 16;
        } else {
            state.eob_run = read_eob_run(&mut state.reader, zero_run)? - 1;
            break;
        }
    }
    Ok(())
}

fn decode_ac_refinement(
    state: &mut ScanState<'_>,
    coefficients: &mut [i32; 64],
    plan: &ProgressivePlan,
    scan: &ScanHeader,
) -> Result<()> {
    assert!(scan.spectral_start > 0);
    assert!(scan.successive_high > 0);

    let table = plan
        .ac
        .as_ref()
        .ok_or_else(|| Error::new("progressive AC Huffman table is missing"))?;
    let correction = 1_i32 << scan.successive_low;
    let mut spectral = scan.spectral_start;
    if state.eob_run == 0 {
        while spectral <= scan.spectral_end {
            let symbol = table.decode(&mut state.reader)?;
            let mut zero_run = i32::from(symbol >> 4);
            let category = symbol & 0x0f;
            let new_value = refinement_value(&mut state.reader, category, correction)?;
            if category == 0 && zero_run != 15 {
                state.eob_run = read_eob_run(&mut state.reader, zero_run as u8)?;
                break;
            }
            advance_refinement(
                &mut state.reader,
                coefficients,
                scan.spectral_end,
                correction,
                &mut spectral,
                &mut zero_run,
            )?;
            if new_value != 0 {
                coefficients[ZIGZAG_TO_NATURAL[usize::from(spectral)]] = new_value;
            }
            spectral = spectral
                .checked_add(1)
                .ok_or_else(|| Error::new("progressive refinement index overflowed"))?;
        }
    }
    if state.eob_run > 0 {
        refine_remaining(
            &mut state.reader,
            coefficients,
            spectral,
            scan.spectral_end,
            correction,
        )?;
        state.eob_run -= 1;
    }
    Ok(())
}

fn refinement_value(reader: &mut BitReader<'_>, category: u8, correction: i32) -> Result<i32> {
    assert!(category <= 10);
    assert!(correction > 0);
    assert_eq!(correction.count_ones(), 1);

    if category == 0 {
        return Ok(0);
    }
    if category != 1 {
        return Err(Error::new("progressive AC refinement category must be one"));
    }
    if reader.read_bits(1)? != 0 {
        Ok(correction)
    } else {
        Ok(-correction)
    }
}

fn advance_refinement(
    reader: &mut BitReader<'_>,
    coefficients: &mut [i32; 64],
    spectral_end: u8,
    correction: i32,
    spectral: &mut u8,
    zero_run: &mut i32,
) -> Result<()> {
    assert!(correction > 0);
    assert_eq!(correction.count_ones(), 1);
    assert!(*zero_run >= 0);

    loop {
        if *spectral > spectral_end {
            return Err(Error::new(
                "progressive refinement run extends past its band",
            ));
        }
        let coefficient = &mut coefficients[ZIGZAG_TO_NATURAL[usize::from(*spectral)]];
        if *coefficient != 0 {
            refine_nonzero(reader, coefficient, correction)?;
        } else {
            *zero_run -= 1;
            if *zero_run < 0 {
                break;
            }
        }
        *spectral += 1;
    }
    Ok(())
}

fn refine_remaining(
    reader: &mut BitReader<'_>,
    coefficients: &mut [i32; 64],
    spectral_start: u8,
    spectral_end: u8,
    correction: i32,
) -> Result<()> {
    assert!(spectral_start <= spectral_end.saturating_add(1));
    assert!(correction > 0);
    assert_eq!(correction.count_ones(), 1);

    for spectral in spectral_start..=spectral_end {
        let coefficient = &mut coefficients[ZIGZAG_TO_NATURAL[usize::from(spectral)]];
        if *coefficient != 0 {
            refine_nonzero(reader, coefficient, correction)?;
        }
    }
    Ok(())
}

fn refine_nonzero(
    reader: &mut BitReader<'_>,
    coefficient: &mut i32,
    correction: i32,
) -> Result<()> {
    assert!(*coefficient != 0);
    assert!(correction > 0);
    assert_eq!(correction.count_ones(), 1);

    if reader.read_bits(1)? == 0 || (*coefficient & correction) != 0 {
        return Ok(());
    }
    let delta = if *coefficient > 0 {
        correction
    } else {
        -correction
    };
    *coefficient = coefficient
        .checked_add(delta)
        .ok_or_else(|| Error::new("progressive AC refinement overflowed"))?;
    Ok(())
}

fn read_eob_run(reader: &mut BitReader<'_>, additional_bits: u8) -> Result<u32> {
    assert!(additional_bits <= 14);
    assert!(u32::from(additional_bits) < u32::BITS);

    let base = 1_u32 << additional_bits;
    let suffix = u32::from(reader.read_bits(additional_bits)?);
    base.checked_add(suffix)
        .ok_or_else(|| Error::new("progressive EOB run overflowed"))
}

fn scale_coefficient(value: i32, successive_low: u8) -> Result<i32> {
    assert!(successive_low <= 13);
    assert!(value.checked_abs().is_some());

    value
        .checked_mul(1_i32 << successive_low)
        .ok_or_else(|| Error::new("progressive coefficient overflowed"))
}
