// This 16x16 baseline JPEG has red, green, blue, and white quadrants. Keeping it textual avoids a
// mysterious binary fixture while still exercising marker parsing, 4:2:0 sampling, and the IDCT.
const BASELINE_COLOR_JPEG_BASE64: &str = concat!(
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsK",
    "CwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQU",
    "FBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAAQABADASIAAhEB",
    "AxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9",
    "AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6",
    "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ip",
    "qrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEB",
    "AQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB",
    "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RV",
    "VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6",
    "wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD50r7Gr8vK",
    "/pbo8TPAz/Vz6n/wpe09p7T/AJdctuXk/wCnrve4vGHM/wDiNP1D3Pqn1T2vX2vP7X2flT5e",
    "X2fne/S2v//Z",
);

pub(super) fn baseline_color_jpeg() -> Vec<u8> {
    assert_eq!(BASELINE_COLOR_JPEG_BASE64.len() % 4, 0);
    assert!(!BASELINE_COLOR_JPEG_BASE64.is_empty());

    let mut decoded = Vec::with_capacity(BASELINE_COLOR_JPEG_BASE64.len() / 4 * 3);
    let (groups, remainder) = BASELINE_COLOR_JPEG_BASE64.as_bytes().as_chunks::<4>();
    assert!(remainder.is_empty());
    for group in groups {
        let first = u32::from(base64_value(group[0]));
        let second = u32::from(base64_value(group[1]));
        let third = u32::from(base64_value(group[2]));
        let fourth = u32::from(base64_value(group[3]));
        let bits = first << 18 | second << 12 | third << 6 | fourth;
        decoded.push((bits >> 16) as u8);
        if group[2] != b'=' {
            decoded.push((bits >> 8) as u8);
        }
        if group[3] != b'=' {
            decoded.push(bits as u8);
        }
    }
    assert_eq!(&decoded[..2], &[0xff, 0xd8]);
    assert_eq!(&decoded[decoded.len() - 2..], &[0xff, 0xd9]);
    decoded
}

fn base64_value(byte: u8) -> u8 {
    assert!(byte.is_ascii());
    assert!(byte != b'\n');

    match byte {
        b'A'..=b'Z' => byte - b'A',
        b'a'..=b'z' => byte - b'a' + 26,
        b'0'..=b'9' => byte - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        b'=' => 0,
        _ => panic!("invalid base64 test fixture"),
    }
}
