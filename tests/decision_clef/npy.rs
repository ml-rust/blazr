//! A minimal `.npy` reader for the Clef fixtures: format 1.0 to 3.0, C
//! order, little-endian `u2` data holding BF16 bits, read as F32.

use std::path::Path;

/// One array: its shape and its values as F32.
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

fn fail(path: &Path, why: &str) -> ! {
    panic!("{}: {why}", path.display())
}

/// Reads the BF16 array at `path`, panicking with the path and the reason
/// when the file is not one.
pub fn read_bf16(path: &Path) -> Npy {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        fail(path, "no .npy magic");
    }
    let (header_len, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 if bytes.len() >= 12 => {
            let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (n as usize, 12)
        }
        v => fail(path, &format!("format version {v} is not 1, 2 or 3")),
    };
    let end = start + header_len;
    if bytes.len() < end {
        fail(path, "the header runs past the file");
    }
    let header = String::from_utf8_lossy(&bytes[start..end]).to_string();
    if !header.contains("'descr': '<u2'") {
        fail(path, &format!("not BF16 bits as '<u2': {header}"));
    }
    if !header.contains("'fortran_order': False") {
        fail(path, "Fortran order is not handled");
    }
    let at = header
        .find("'shape': (")
        .unwrap_or_else(|| fail(path, &format!("no shape in {header}")));
    let rest = &header[at + "'shape': (".len()..];
    let close = rest
        .find(')')
        .unwrap_or_else(|| fail(path, &format!("bad shape in {header}")));
    let shape: Vec<usize> = rest[..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .unwrap_or_else(|_| fail(path, &format!("bad dim {s}")))
        })
        .collect();
    let count: usize = shape.iter().product();
    let body = &bytes[end..];
    if body.len() != count * 2 {
        fail(path, "the payload size disagrees with the shape");
    }
    let (words, _) = body.as_chunks::<2>();
    let data = words
        .iter()
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
        .collect();
    Npy { shape, data }
}
