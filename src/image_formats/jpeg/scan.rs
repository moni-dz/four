//! Parses arithmetic JPEG frame components and scan headers.

use exn::OptionExt;

use super::{
    COMPONENTS_MAX, CodingProcess, Frame, FrameComponent, JPEGError, JPEGLimit, JPEGTableKind,
    QUANTIZATION_TABLES_MAX, Result, arithmetic, error, reader::Reader,
};

pub(super) struct ScanComponent {
    pub(super) frame_index: usize,
    pub(super) dc_table: usize,
    pub(super) ac_table: usize,
}

pub(super) struct ScanHeader {
    pub(super) components: Vec<ScanComponent>,
    pub(super) spectral_start: u8,
    pub(super) spectral_end: u8,
    pub(super) successive_high: u8,
    pub(super) successive_low: u8,
}

pub(super) fn parse_frame_components(
    segment: &mut Reader<'_>,
    component_count: usize,
) -> Result<Vec<FrameComponent>> {
    invariant!(component_count == 1 || component_count == 3);
    invariant!(component_count <= COMPONENTS_MAX);

    let mut components = Vec::with_capacity(component_count);
    let mut blocks_per_mcu = 0_u32;
    for _ in 0..component_count {
        let identifier = segment.read_u8()?;
        if components
            .iter()
            .any(|component: &FrameComponent| component.identifier == identifier)
        {
            return Err(error(JPEGError::Frame(
                "frame contains duplicate component identifiers",
            )));
        }

        let sampling = segment.read_u8()?;
        let horizontal_sampling = sampling >> 4;
        let vertical_sampling = sampling & 0x0f;
        if !(1..=4).contains(&horizontal_sampling) || !(1..=4).contains(&vertical_sampling) {
            return Err(error(JPEGError::Frame(
                "component sampling factor is outside 1 through 4",
            )));
        }

        blocks_per_mcu += u32::from(horizontal_sampling) * u32::from(vertical_sampling);
        
        let quantization_table = usize::from(segment.read_u8()?);
        if quantization_table >= QUANTIZATION_TABLES_MAX {
            return Err(error(JPEGError::Table(
                JPEGTableKind::Quantization,
                "component quantization table index is out of range",
            )));
        }

        components.push(FrameComponent {
            identifier,
            horizontal_sampling,
            vertical_sampling,
            quantization_table,
            plane_width: 0,
            plane: Vec::new(),
            block_columns: 0,
            block_rows: 0,
            data_block_columns: 0,
            data_block_rows: 0,
            coefficients: Vec::new(),
        });
    }

    if blocks_per_mcu > 10 {
        return Err(error(JPEGError::LimitExceeded(JPEGLimit::FrameDataUnits(
            10,
        ))));
    }

    Ok(components)
}

pub(super) fn parse_scan_header(segment: &mut Reader<'_>, frame: &Frame) -> Result<ScanHeader> {
    invariant!(!frame.components.is_empty());
    invariant!(frame.components.len() <= COMPONENTS_MAX);

    let component_count = usize::from(segment.read_u8()?);
    if component_count == 0 || component_count > frame.components.len() {
        return Err(error(JPEGError::Scan(
            "scan component count is out of range",
        )));
    }

    let mut components = Vec::with_capacity(component_count);
    for _ in 0..component_count {
        let identifier = segment.read_u8()?;
        let frame_index = frame
            .components
            .iter()
            .position(|component| component.identifier == identifier)
            .ok_or_raise(|| JPEGError::Scan("scan references an unknown frame component"))?;

        if components
            .iter()
            .any(|component: &ScanComponent| component.frame_index == frame_index)
        {
            return Err(error(JPEGError::Scan(
                "scan contains a duplicate component",
            )));
        }

        let selectors = segment.read_u8()?;
        let dc_table = usize::from(selectors >> 4);
        let ac_table = usize::from(selectors & 0x0f);
        if dc_table >= arithmetic::TABLES_MAX || ac_table >= arithmetic::TABLES_MAX {
            return Err(error(JPEGError::Scan(
                "scan entropy table selector is out of range",
            )));
        }

        components.push(ScanComponent {
            frame_index,
            dc_table,
            ac_table,
        });
    }

    let spectral_start = segment.read_u8()?;
    let spectral_end = segment.read_u8()?;
    let approximation = segment.read_u8()?;
    if segment.remaining() != 0 {
        return Err(error(JPEGError::Segment("SOS segment has trailing bytes")));
    }

    let header = ScanHeader {
        components,
        spectral_start,
        spectral_end,
        successive_high: approximation >> 4,
        successive_low: approximation & 0x0f,
    };
    
    validate_scan_header(&header, frame.process)?;

    Ok(header)
}

fn validate_scan_header(scan: &ScanHeader, process: CodingProcess) -> Result<()> {
    invariant!(!scan.components.is_empty());
    invariant!(scan.components.len() <= COMPONENTS_MAX);

    if process == CodingProcess::Sequential {
        let is_sequential = scan.spectral_start == 0
            && scan.spectral_end == 63
            && scan.successive_high == 0
            && scan.successive_low == 0;
        
        if !is_sequential {
            return Err(error(JPEGError::Scan(
                "sequential JPEG scan parameters are invalid",
            )));
        }
        
        return Ok(());
    }
    
    if scan.spectral_start > scan.spectral_end || scan.spectral_end >= 64 {
        return Err(error(JPEGError::Scan(
            "progressive spectral selection is out of range",
        )));
    }
    
    if scan.spectral_start == 0 && scan.spectral_end != 0 {
        return Err(error(JPEGError::Scan(
            "a progressive DC scan must end at coefficient zero",
        )));
    }
    
    if scan.spectral_start > 0 && scan.components.len() != 1 {
        return Err(error(JPEGError::Scan(
            "a progressive AC scan must contain one component",
        )));
    }
    
    if scan.successive_high > 0 && scan.successive_low + 1 != scan.successive_high {
        return Err(error(JPEGError::Scan(
            "progressive refinement must advance exactly one bit",
        )));
    }
    
    if scan.successive_low > 13 {
        return Err(error(JPEGError::Scan(
            "progressive approximation bit exceeds 13",
        )));
    }
    
    Ok(())
}
