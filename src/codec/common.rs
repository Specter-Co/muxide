//! Common bitstream utilities for start code scanning.
//!
//! Provides shared infrastructure for parsing Annex B formatted bitstreams
//! (H.264, H.265) which use start code delimiters.

/// Iterator over NAL units in an Annex B bitstream.
///
/// Annex B uses start codes (`0x00 0x00 0x01` or `0x00 0x00 0x00 0x01`) to
/// delimit NAL units. This iterator yields each NAL unit's payload without
/// the start code prefix.
///
/// # Example
///
/// ```
/// use muxide::codec::AnnexBNalIter;
///
/// // Two NAL units with 4-byte start codes
/// let data = [
///     0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f,  // SPS
///     0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x3c, 0x80,  // PPS
/// ];
///
/// let nals: Vec<_> = AnnexBNalIter::new(&data).collect();
/// assert_eq!(nals.len(), 2);
/// assert_eq!(nals[0][0] & 0x1f, 7);  // SPS NAL type
/// assert_eq!(nals[1][0] & 0x1f, 8);  // PPS NAL type
/// ```
pub struct AnnexBNalIter<'a> {
    data: &'a [u8],
    /// Pending start code for the current NAL; primed on construction and
    /// refreshed at the end of each `next` so a single scan locates the boundary
    /// for the next iteration's NAL end.
    next_start: Option<(usize, usize)>,
}

impl<'a> AnnexBNalIter<'a> {
    /// Create a new iterator over NAL units in the given Annex B data.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            next_start: find_start_code(data, 0),
        }
    }
}

impl<'a> Iterator for AnnexBNalIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let (start_code_pos, start_code_len) = self.next_start.take()?;
        let nal_start = start_code_pos + start_code_len;

        let next = find_start_code(self.data, nal_start);
        let nal_end = next.map_or(self.data.len(), |(p, _)| p);
        self.next_start = next;

        Some(&self.data[nal_start..nal_end])
    }
}

/// Find the next Annex B start code in the data starting from `from`.
///
/// Returns the position and length of the start code:
/// - 4-byte: `0x00 0x00 0x00 0x01` (length = 4)
/// - 3-byte: `0x00 0x00 0x01` (length = 3)
///
/// 4-byte start codes are checked first to avoid matching `0x00 0x00 0x01`
/// within a `0x00 0x00 0x00 0x01` sequence.
///
/// # Returns
///
/// - `Some((position, length))` if a start code is found
/// - `None` if no start code exists from `from` onwards
pub fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    if from >= data.len() {
        return None;
    }
    // SIMD scan for the 3-byte tail; promote to 4-byte when preceded by 0x00.
    let pos = memchr::memmem::find(&data[from..], &[0x00, 0x00, 0x01])? + from;
    if pos > from && data[pos - 1] == 0x00 {
        Some((pos - 1, 4))
    } else {
        Some((pos, 3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_start_code_4byte() {
        let data = [0x00, 0x00, 0x00, 0x01, 0x67];
        assert_eq!(find_start_code(&data, 0), Some((0, 4)));
    }

    #[test]
    fn test_find_start_code_3byte() {
        let data = [0x00, 0x00, 0x01, 0x67];
        assert_eq!(find_start_code(&data, 0), Some((0, 3)));
    }

    #[test]
    fn test_find_start_code_offset() {
        let data = [0xAB, 0xCD, 0x00, 0x00, 0x00, 0x01, 0x67];
        assert_eq!(find_start_code(&data, 0), Some((2, 4)));
        // When starting from position 3, we find the start code at position 3
        // (this is the middle of a 4-byte start code, but also valid as 3-byte)
        assert_eq!(find_start_code(&data, 3), Some((3, 3)));
        assert_eq!(find_start_code(&data, 6), None);
    }

    #[test]
    fn test_find_start_code_none() {
        let data = [0x00, 0x00, 0x02, 0x67];
        assert_eq!(find_start_code(&data, 0), None);
    }

    #[test]
    fn test_annexb_iter_multiple_nals() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, // SPS (type 7)
            0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, // PPS (type 8)
            0x00, 0x00, 0x00, 0x01, 0x65, 0x88, // IDR (type 5)
        ];

        let nals: Vec<_> = AnnexBNalIter::new(&data).collect();
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0], &[0x67, 0x42]);
        assert_eq!(nals[1], &[0x68, 0xCE]);
        assert_eq!(nals[2], &[0x65, 0x88]);
    }

    #[test]
    fn test_annexb_iter_empty() {
        let data: [u8; 0] = [];
        let nals: Vec<_> = AnnexBNalIter::new(&data).collect();
        assert!(nals.is_empty());
    }

    #[test]
    fn test_annexb_iter_no_start_code() {
        let data = [0x67, 0x42, 0x00, 0x1f];
        let nals: Vec<_> = AnnexBNalIter::new(&data).collect();
        assert!(nals.is_empty());
    }

    #[test]
    fn test_annexb_iter_mixed_start_codes() {
        // Mix of 3-byte and 4-byte start codes
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, // 4-byte
            0x00, 0x00, 0x01, 0x68, 0xCE, // 3-byte
        ];

        let nals: Vec<_> = AnnexBNalIter::new(&data).collect();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &[0x67, 0x42]);
        assert_eq!(nals[1], &[0x68, 0xCE]);
    }
}
