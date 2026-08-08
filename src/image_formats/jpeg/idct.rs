const BLOCK_SIDE: usize = 8;
const LEVEL_SHIFT: f32 = 128.0;

// C(u) * cos((2x + 1)u*pi/16), stored once so decoding does no trigonometry per block.
const BASIS: [[f32; BLOCK_SIDE]; BLOCK_SIDE] = [
    [
        0.70710677,
        0.98078525,
        0.923_879_5,
        0.831_469_6,
        0.70710677,
        0.55557024,
        0.38268343,
        0.19509032,
    ],
    [
        0.70710677,
        0.831_469_6,
        0.38268343,
        -0.19509032,
        -0.70710677,
        -0.98078525,
        -0.923_879_5,
        -0.55557024,
    ],
    [
        0.70710677,
        0.55557024,
        -0.38268343,
        -0.98078525,
        -0.70710677,
        0.19509032,
        0.923_879_5,
        0.831_469_6,
    ],
    [
        0.70710677,
        0.19509032,
        -0.923_879_5,
        -0.55557024,
        0.70710677,
        0.831_469_6,
        -0.38268343,
        -0.98078525,
    ],
    [
        0.70710677,
        -0.19509032,
        -0.923_879_5,
        0.55557024,
        0.70710677,
        -0.831_469_6,
        -0.38268343,
        0.98078525,
    ],
    [
        0.70710677,
        -0.55557024,
        -0.38268343,
        0.98078525,
        -0.70710677,
        -0.19509032,
        0.923_879_5,
        -0.831_469_6,
    ],
    [
        0.70710677,
        -0.831_469_6,
        0.38268343,
        0.19509032,
        -0.70710677,
        0.98078525,
        -0.923_879_5,
        0.55557024,
    ],
    [
        0.70710677,
        -0.98078525,
        0.923_879_5,
        -0.831_469_6,
        0.70710677,
        -0.55557024,
        0.38268343,
        -0.19509032,
    ],
];

pub(super) fn inverse(coefficients: &[i32; 64]) -> [u8; 64] {
    assert_eq!(coefficients.len(), BLOCK_SIDE * BLOCK_SIDE);
    assert!(
        coefficients
            .iter()
            .all(|value| value.checked_abs().is_some())
    );

    if coefficients[1..].iter().all(|value| *value == 0) {
        let sample = clamp_sample(coefficients[0] as f32 / 8.0 + LEVEL_SHIFT);
        return [sample; 64];
    }

    let mut intermediate = [[0.0_f32; BLOCK_SIDE]; BLOCK_SIDE];
    for (vertical_frequency, intermediate_row) in intermediate.iter_mut().enumerate() {
        for (x, target) in intermediate_row.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            let coefficient_start = vertical_frequency * BLOCK_SIDE;
            let coefficient_row = &coefficients[coefficient_start..coefficient_start + BLOCK_SIDE];
            for (coefficient, basis) in coefficient_row.iter().zip(BASIS[x].iter()) {
                sum += *coefficient as f32 * basis;
            }
            *target = sum;
        }
    }

    let mut samples = [0_u8; 64];
    let (sample_rows, sample_remainder) = samples.as_chunks_mut::<BLOCK_SIDE>();
    assert!(sample_remainder.is_empty());
    for (sample_row, basis_row) in sample_rows.iter_mut().zip(BASIS.iter()) {
        for (x, sample) in sample_row.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for (intermediate_row, basis) in intermediate.iter().zip(basis_row.iter()) {
                sum += intermediate_row[x] * basis;
            }
            *sample = clamp_sample(sum / 4.0 + LEVEL_SHIFT);
        }
    }
    samples
}

fn clamp_sample(value: f32) -> u8 {
    assert!(value.is_finite());
    value.round().clamp(f32::from(u8::MIN), f32::from(u8::MAX)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_coefficient_produces_a_flat_block() {
        let mut coefficients = [0; 64];
        coefficients[0] = 80;
        let samples = inverse(&coefficients);

        assert_eq!(samples, [138; 64]);
        assert!(samples.windows(2).all(|window| window[0] == window[1]));
    }
}
