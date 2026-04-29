//! Fragmented MP4 (fMP4) support for streaming applications.
//!
//! Fragmented MP4 splits the container into an init segment (ftyp + moov)
//! and media segments (moof + mdat). This is essential for:
//! - DASH streaming
//! - HLS with fMP4
//! - Low-latency live streaming
//!
//! # Example
//!
//! ```
//! use muxide::api::VideoCodec;
//! use muxide::fragmented::{FragmentConfig, FragmentedMuxer, SampleSpec};
//!
//! let config = FragmentConfig {
//!     codec: VideoCodec::H264,
//!     width: 1920,
//!     height: 1080,
//!     timescale: 90_000,
//!     sps: vec![0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11],
//!     pps: vec![0x68, 0xce, 0x38, 0x80],
//!     ..Default::default()
//! };
//!
//! let muxer = FragmentedMuxer::new(config);
//!
//! let mut out = Vec::new();
//! muxer.write_init(&mut out);
//! // send `out` to client; reuse the buffer when ready.
//!
//! let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xaa, 0xbb];
//! let samples = [SampleSpec { frame: &frame, pts: 0, dts: 0, is_sync: true }];
//! out.clear();
//! muxer.write_fragment(&mut out, 1, 0, &samples).unwrap();
//! ```

use crate::api::VideoCodec;
use crate::codec::av1::extract_av1_config;
use crate::codec::h264::annexb_to_avcc_into;
use crate::codec::h265::{
    hevc_annexb_to_hvcc_into, sps_general_level_idc, sps_general_profile_idc,
    sps_general_profile_space, sps_general_tier_flag,
};

/// Errors that can occur during fragmented MP4 muxing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FragmentedError {
    #[error("DTS values must be non-decreasing: prev={prev_dts}, curr={curr_dts}")]
    NonMonotonicDts { prev_dts: u64, curr_dts: u64 },
}

/// Configuration for fragmented MP4 output.
#[derive(Debug, Clone)]
pub struct FragmentConfig {
    /// Video codec; selects sample-entry box, conversion path, and required parameter sets.
    pub codec: VideoCodec,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Media timescale (typically 90000 for video).
    pub timescale: u32,
    /// SPS NAL unit (H.264 / H.265 required for init segment).
    pub sps: Vec<u8>,
    /// PPS NAL unit (H.264 / H.265 required for init segment).
    pub pps: Vec<u8>,
    /// VPS NAL unit (H.265 required for init segment).
    pub vps: Option<Vec<u8>>,
    /// Sequence Header OBU (AV1 required for init segment).
    pub av1_sequence_header: Option<Vec<u8>>,
    /// VP9 configuration (extracted from first keyframe).
    pub vp9_config: Option<crate::codec::vp9::Vp9Config>,
}

impl Default for FragmentConfig {
    fn default() -> Self {
        // Note: This default provides example SPS/PPS for testing.
        // In production, you must provide actual SPS/PPS from your encoder.
        Self {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            timescale: 90000,
            sps: vec![0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11],
            pps: vec![0x68, 0xce, 0x38, 0x80],
            vps: None,
            av1_sequence_header: None,
            vp9_config: None,
        }
    }
}

/// One sample to write into a fragment. `frame` is in the codec's wire format
/// (Annex B for H.264/H.265, OBU for AV1/VP9); the muxer converts as it writes.
#[derive(Debug, Clone, Copy)]
pub struct SampleSpec<'a> {
    pub frame: &'a [u8],
    pub pts: u64,
    pub dts: u64,
    pub is_sync: bool,
}

/// Per-sample duration in trun: gap to next sample, or for the last sample
/// mirror the previous gap. Single-sample fragments fall back to 3000 ticks.
fn sample_duration(samples: &[SampleSpec<'_>], i: usize) -> u32 {
    if i + 1 < samples.len() {
        samples[i + 1].dts.saturating_sub(samples[i].dts) as u32
    } else if i > 0 {
        samples[i].dts.saturating_sub(samples[i - 1].dts) as u32
    } else {
        3000
    }
}

/// The `base_media_decode_time` the next fragment should carry to continue
/// the timeline that `write_fragment(samples)` wrote.
pub fn next_base_media_decode_time(samples: &[SampleSpec<'_>]) -> Option<u64> {
    let last_idx = samples.len().checked_sub(1)?;
    Some(samples[last_idx].dts + sample_duration(samples, last_idx) as u64)
}

/// Fragmented MP4 muxer. Per-fragment counters are caller-managed so the muxer
/// stays immutable across fragments and is safe to share.
#[derive(Debug, Clone)]
pub struct FragmentedMuxer {
    config: FragmentConfig,
}

impl FragmentedMuxer {
    pub fn new(config: FragmentConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &FragmentConfig {
        &self.config
    }

    /// Append the init segment (ftyp + moov) to `out`.
    pub fn write_init(&self, out: &mut Vec<u8>) {
        write_ftyp(out);
        write_moov(out, &self.config);
    }

    /// Append one media fragment (moof + mdat) to `out`. Sample bytes are
    /// converted into the mdat in a single pass; per-sample sizes are
    /// patched into the trun region as they're discovered.
    pub fn write_fragment(
        &self,
        out: &mut Vec<u8>,
        sequence_number: u32,
        base_media_decode_time: u64,
        samples: &[SampleSpec<'_>],
    ) -> Result<(), FragmentedError> {
        if samples.is_empty() {
            return Ok(());
        }

        for w in samples.windows(2) {
            if w[1].dts < w[0].dts {
                return Err(FragmentedError::NonMonotonicDts {
                    prev_dts: w[0].dts,
                    curr_dts: w[1].dts,
                });
            }
        }

        let n = samples.len();
        let moof_size = moof_size_for(n);
        let moof_start = out.len();
        let data_offset = (moof_size + MDAT_HEADER_SIZE) as u32;

        let frame_total: usize = samples.iter().map(|s| s.frame.len()).sum();
        out.reserve(moof_size + MDAT_HEADER_SIZE + frame_total + 4 * n);
        out.resize(moof_start + moof_size, 0);

        // Write the moof up front — every field except per-sample sizes is
        // computable from `samples`. Sizes are patched inline below.
        write_moof_skeleton(
            &mut out[moof_start..moof_start + moof_size],
            sequence_number,
            base_media_decode_time,
            data_offset,
            samples,
        );
        let trun_samples_off = moof_start + trun_samples_offset_in_moof();

        let mdat_header_start = out.len();
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(b"mdat");

        for (i, s) in samples.iter().enumerate() {
            let pre = out.len();
            convert_into(self.config.codec, s.frame, out);
            let size = (out.len() - pre) as u32;
            let size_off = trun_samples_off + i * TRUN_PER_SAMPLE + 4;
            out[size_off..size_off + 4].copy_from_slice(&size.to_be_bytes());
        }

        let mdat_size = (out.len() - mdat_header_start) as u32;
        out[mdat_header_start..mdat_header_start + 4].copy_from_slice(&mdat_size.to_be_bytes());

        Ok(())
    }
}

// ============================================================================
// Codec conversion into output
// ============================================================================

fn convert_into(codec: VideoCodec, frame: &[u8], out: &mut Vec<u8>) {
    match codec {
        VideoCodec::H264 => annexb_to_avcc_into(frame, out),
        VideoCodec::H265 => hevc_annexb_to_hvcc_into(frame, out),
        VideoCodec::Av1 | VideoCodec::Vp9 => out.extend_from_slice(frame),
    }
}

// ============================================================================
// moof: closed-form sizes from sample count.
// ============================================================================

const MFHD_SIZE: usize = 16;
const TFHD_SIZE: usize = 16;
const TFDT_SIZE: usize = 20;
const TRUN_HEADER_SIZE: usize = 20;
const TRUN_PER_SAMPLE: usize = 16;
const TRAF_HEADER_SIZE: usize = 8;
const MOOF_HEADER_SIZE: usize = 8;
const MDAT_HEADER_SIZE: usize = 8;

fn moof_size_for(sample_count: usize) -> usize {
    MOOF_HEADER_SIZE
        + MFHD_SIZE
        + TRAF_HEADER_SIZE
        + TFHD_SIZE
        + TFDT_SIZE
        + TRUN_HEADER_SIZE
        + TRUN_PER_SAMPLE * sample_count
}

/// Byte offset of the first per-sample trun entry within a moof.
fn trun_samples_offset_in_moof() -> usize {
    MOOF_HEADER_SIZE + MFHD_SIZE + TRAF_HEADER_SIZE + TFHD_SIZE + TFDT_SIZE + TRUN_HEADER_SIZE
}

/// Write the moof in full except for per-sample sizes, which are patched
/// into the trun region as samples are appended to mdat.
fn write_moof_skeleton(
    moof: &mut [u8],
    sequence_number: u32,
    base_media_decode_time: u64,
    data_offset: u32,
    samples: &[SampleSpec<'_>],
) {
    let total = moof.len();
    debug_assert_eq!(total, moof_size_for(samples.len()));

    moof[0..4].copy_from_slice(&(total as u32).to_be_bytes());
    moof[4..8].copy_from_slice(b"moof");

    let mut p = MOOF_HEADER_SIZE;

    // mfhd
    moof[p..p + 4].copy_from_slice(&(MFHD_SIZE as u32).to_be_bytes());
    moof[p + 4..p + 8].copy_from_slice(b"mfhd");
    moof[p + 8..p + 12].copy_from_slice(&0u32.to_be_bytes()); // version + flags
    moof[p + 12..p + 16].copy_from_slice(&sequence_number.to_be_bytes());
    p += MFHD_SIZE;

    // traf
    let traf_size = TRAF_HEADER_SIZE
        + TFHD_SIZE
        + TFDT_SIZE
        + TRUN_HEADER_SIZE
        + TRUN_PER_SAMPLE * samples.len();
    moof[p..p + 4].copy_from_slice(&(traf_size as u32).to_be_bytes());
    moof[p + 4..p + 8].copy_from_slice(b"traf");
    p += TRAF_HEADER_SIZE;

    // tfhd
    moof[p..p + 4].copy_from_slice(&(TFHD_SIZE as u32).to_be_bytes());
    moof[p + 4..p + 8].copy_from_slice(b"tfhd");
    // Flags: 0x020000 = default-base-is-moof
    moof[p + 8..p + 12].copy_from_slice(&0x0002_0000_u32.to_be_bytes());
    moof[p + 12..p + 16].copy_from_slice(&1u32.to_be_bytes()); // track ID
    p += TFHD_SIZE;

    // tfdt
    moof[p..p + 4].copy_from_slice(&(TFDT_SIZE as u32).to_be_bytes());
    moof[p + 4..p + 8].copy_from_slice(b"tfdt");
    // Version 1 for 64-bit decode time.
    moof[p + 8..p + 12].copy_from_slice(&0x0100_0000_u32.to_be_bytes());
    moof[p + 12..p + 20].copy_from_slice(&base_media_decode_time.to_be_bytes());
    p += TFDT_SIZE;

    // trun
    let trun_size = TRUN_HEADER_SIZE + TRUN_PER_SAMPLE * samples.len();
    moof[p..p + 4].copy_from_slice(&(trun_size as u32).to_be_bytes());
    moof[p + 4..p + 8].copy_from_slice(b"trun");
    // Flags:
    //  0x000001 data-offset-present
    //  0x000100 sample-duration-present
    //  0x000200 sample-size-present
    //  0x000400 sample-flags-present
    //  0x000800 sample-composition-time-offset-present
    let trun_flags: u32 = 0x000001 | 0x000100 | 0x000200 | 0x000400 | 0x000800;
    // Version 1 for signed composition time offsets.
    moof[p + 8..p + 12].copy_from_slice(&(0x0100_0000 | trun_flags).to_be_bytes());
    moof[p + 12..p + 16].copy_from_slice(&(samples.len() as u32).to_be_bytes());
    moof[p + 16..p + 20].copy_from_slice(&data_offset.to_be_bytes());
    p += TRUN_HEADER_SIZE;

    // Per-sample: duration, size (left zero, patched after conversion), flags, cts.
    for (i, s) in samples.iter().enumerate() {
        let duration = sample_duration(samples, i);
        moof[p..p + 4].copy_from_slice(&duration.to_be_bytes());
        // size (p+4..p+8) is patched in the mdat conversion loop.
        let flags: u32 = if s.is_sync { 0x0200_0000 } else { 0x0101_0000 };
        moof[p + 8..p + 12].copy_from_slice(&flags.to_be_bytes());
        let cts = (s.pts as i64 - s.dts as i64) as i32;
        moof[p + 12..p + 16].copy_from_slice(&cts.to_be_bytes());
        p += TRUN_PER_SAMPLE;
    }

    debug_assert_eq!(p, total);
}

// ============================================================================
// Init segment box building
// ============================================================================

fn write_box<F: FnOnce(&mut Vec<u8>)>(out: &mut Vec<u8>, typ: &[u8; 4], body: F) {
    let start = out.len();
    out.extend_from_slice(&[0, 0, 0, 0]); // size placeholder
    out.extend_from_slice(typ);
    body(out);
    let size = (out.len() - start) as u32;
    out[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

fn write_ftyp(out: &mut Vec<u8>) {
    write_box(out, b"ftyp", |o| {
        o.extend_from_slice(b"iso5"); // major brand
        o.extend_from_slice(&0u32.to_be_bytes()); // minor version
        o.extend_from_slice(b"iso5");
        o.extend_from_slice(b"iso6");
        o.extend_from_slice(b"mp41");
    });
}

fn write_moov(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"moov", |o| {
        write_mvhd(o, config.timescale);
        write_mvex(o);
        write_trak(o, config);
    });
}

fn write_mvhd(out: &mut Vec<u8>, timescale: u32) {
    write_box(out, b"mvhd", |o| {
        o.extend_from_slice(&0u32.to_be_bytes()); // version + flags
        o.extend_from_slice(&0u32.to_be_bytes()); // creation time
        o.extend_from_slice(&0u32.to_be_bytes()); // modification time
        o.extend_from_slice(&timescale.to_be_bytes());
        o.extend_from_slice(&0u32.to_be_bytes()); // duration (unknown for live)
        o.extend_from_slice(&0x0001_0000_u32.to_be_bytes()); // rate (1.0)
        o.extend_from_slice(&0x0100_u16.to_be_bytes()); // volume
        o.extend_from_slice(&[0u8; 10]); // reserved
                                         // Unity matrix.
        o.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
        o.extend_from_slice(&[0u8; 12]);
        o.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
        o.extend_from_slice(&[0u8; 12]);
        o.extend_from_slice(&0x4000_0000_u32.to_be_bytes());
        o.extend_from_slice(&[0u8; 24]); // pre-defined
        o.extend_from_slice(&2u32.to_be_bytes()); // next track ID
    });
}

fn write_mvex(out: &mut Vec<u8>) {
    write_box(out, b"mvex", |o| {
        write_box(o, b"trex", |t| {
            t.extend_from_slice(&0u32.to_be_bytes()); // version + flags
            t.extend_from_slice(&1u32.to_be_bytes()); // track ID
            t.extend_from_slice(&1u32.to_be_bytes()); // default sample description index
            t.extend_from_slice(&0u32.to_be_bytes()); // default sample duration
            t.extend_from_slice(&0u32.to_be_bytes()); // default sample size
            t.extend_from_slice(&0u32.to_be_bytes()); // default sample flags
        });
    });
}

fn write_trak(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"trak", |o| {
        write_tkhd(o, config);
        write_mdia(o, config);
    });
}

fn write_tkhd(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"tkhd", |o| {
        o.extend_from_slice(&0x0000_0003_u32.to_be_bytes()); // enabled + in_movie
        o.extend_from_slice(&0u32.to_be_bytes()); // creation time
        o.extend_from_slice(&0u32.to_be_bytes()); // modification time
        o.extend_from_slice(&1u32.to_be_bytes()); // track ID
        o.extend_from_slice(&0u32.to_be_bytes()); // reserved
        o.extend_from_slice(&0u32.to_be_bytes()); // duration
        o.extend_from_slice(&[0u8; 8]); // reserved
        o.extend_from_slice(&0u16.to_be_bytes()); // layer
        o.extend_from_slice(&0u16.to_be_bytes()); // alternate group
        o.extend_from_slice(&0u16.to_be_bytes()); // volume (0 for video)
        o.extend_from_slice(&0u16.to_be_bytes()); // reserved
                                                  // Unity matrix.
        o.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
        o.extend_from_slice(&[0u8; 12]);
        o.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
        o.extend_from_slice(&[0u8; 12]);
        o.extend_from_slice(&0x4000_0000_u32.to_be_bytes());
        // Width/height in fixed-point 16.16.
        o.extend_from_slice(&((config.width) << 16).to_be_bytes());
        o.extend_from_slice(&((config.height) << 16).to_be_bytes());
    });
}

fn write_mdia(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"mdia", |o| {
        write_mdhd(o, config.timescale);
        write_hdlr_video(o);
        write_minf(o, config);
    });
}

fn write_mdhd(out: &mut Vec<u8>, timescale: u32) {
    write_box(out, b"mdhd", |o| {
        o.extend_from_slice(&0u32.to_be_bytes()); // version + flags
        o.extend_from_slice(&0u32.to_be_bytes()); // creation time
        o.extend_from_slice(&0u32.to_be_bytes()); // modification time
        o.extend_from_slice(&timescale.to_be_bytes());
        o.extend_from_slice(&0u32.to_be_bytes()); // duration (unknown)
        o.extend_from_slice(&encode_language_code("und"));
        o.extend_from_slice(&0u16.to_be_bytes()); // quality
    });
}

fn encode_language_code(language: &str) -> [u8; 2] {
    let chars: Vec<char> = language.chars().take(3).collect();
    let c1 = chars.first().copied().unwrap_or('u') as u16;
    let c2 = chars.get(1).copied().unwrap_or('n') as u16;
    let c3 = chars.get(2).copied().unwrap_or('d') as u16;
    let packed = ((c1.saturating_sub(0x60) & 0x1F) << 10)
        | ((c2.saturating_sub(0x60) & 0x1F) << 5)
        | (c3.saturating_sub(0x60) & 0x1F);
    packed.to_be_bytes()
}

fn write_hdlr_video(out: &mut Vec<u8>) {
    write_box(out, b"hdlr", |o| {
        o.extend_from_slice(&0u32.to_be_bytes()); // version + flags
        o.extend_from_slice(&0u32.to_be_bytes()); // pre-defined
        o.extend_from_slice(b"vide");
        o.extend_from_slice(&[0u8; 12]); // reserved
        o.extend_from_slice(b"VideoHandler\0");
    });
}

fn write_minf(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"minf", |o| {
        write_vmhd(o);
        write_dinf(o);
        write_stbl(o, config);
    });
}

fn write_vmhd(out: &mut Vec<u8>) {
    write_box(out, b"vmhd", |o| {
        o.extend_from_slice(&0x0000_0001_u32.to_be_bytes());
        o.extend_from_slice(&[0u8; 8]); // graphics mode + op color
    });
}

fn write_dinf(out: &mut Vec<u8>) {
    write_box(out, b"dinf", |o| {
        write_box(o, b"dref", |d| {
            d.extend_from_slice(&0u32.to_be_bytes()); // version + flags
            d.extend_from_slice(&1u32.to_be_bytes()); // entry count
            write_box(d, b"url ", |u| {
                u.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // self-contained
            });
        });
    });
}

fn write_stbl(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"stbl", |o| {
        write_stsd(o, config);
        // Empty sample tables: actual data lives in moof/trun.
        write_box(o, b"stts", |s| {
            s.extend_from_slice(&0u32.to_be_bytes());
            s.extend_from_slice(&0u32.to_be_bytes());
        });
        write_box(o, b"stsc", |s| {
            s.extend_from_slice(&0u32.to_be_bytes());
            s.extend_from_slice(&0u32.to_be_bytes());
        });
        write_box(o, b"stsz", |s| {
            s.extend_from_slice(&0u32.to_be_bytes());
            s.extend_from_slice(&0u32.to_be_bytes());
            s.extend_from_slice(&0u32.to_be_bytes());
        });
        write_box(o, b"stco", |s| {
            s.extend_from_slice(&0u32.to_be_bytes());
            s.extend_from_slice(&0u32.to_be_bytes());
        });
    });
}

fn write_stsd(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"stsd", |o| {
        o.extend_from_slice(&0u32.to_be_bytes()); // version + flags
        o.extend_from_slice(&1u32.to_be_bytes()); // entry count
        match config.codec {
            VideoCodec::H264 => write_avc1(o, config),
            VideoCodec::H265 => write_hvc1(o, config),
            VideoCodec::Av1 => write_av01(o, config),
            VideoCodec::Vp9 => write_vp09(o, config),
        }
    });
}

fn write_visual_sample_entry_header(out: &mut Vec<u8>, width: u32, height: u32) {
    out.extend_from_slice(&[0u8; 6]); // reserved
    out.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    out.extend_from_slice(&0u16.to_be_bytes()); // pre-defined
    out.extend_from_slice(&0u16.to_be_bytes()); // reserved
    out.extend_from_slice(&[0u8; 12]); // pre-defined
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // 72 dpi
    out.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // reserved
    out.extend_from_slice(&1u16.to_be_bytes()); // frame count
    out.extend_from_slice(&[0u8; 32]); // compressor name
    out.extend_from_slice(&0x0018_u16.to_be_bytes()); // 24-bit depth
    out.extend_from_slice(&0xffff_u16.to_be_bytes()); // pre-defined (-1)
}

fn write_avc1(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"avc1", |o| {
        write_visual_sample_entry_header(o, config.width, config.height);
        write_avcc(o, config);
    });
}

fn write_avcc(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"avcC", |o| {
        o.push(1); // configuration version
        o.push(config.sps.get(1).copied().unwrap_or(0x42)); // profile
        o.push(config.sps.get(2).copied().unwrap_or(0x00)); // profile compatibility
        o.push(config.sps.get(3).copied().unwrap_or(0x1e)); // level
        o.push(0xff); // 6 reserved + lengthSizeMinusOne (4-byte length prefix)
        o.push(0xe1); // 3 reserved + numSPS
        o.extend_from_slice(&(config.sps.len() as u16).to_be_bytes());
        o.extend_from_slice(&config.sps);
        o.push(1); // numPPS
        o.extend_from_slice(&(config.pps.len() as u16).to_be_bytes());
        o.extend_from_slice(&config.pps);
    });
}

fn write_hvc1(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"hvc1", |o| {
        write_visual_sample_entry_header(o, config.width, config.height);
        write_hvcc(o, config);
    });
}

fn write_hvcc(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"hvcC", |o| {
        let num_arrays: u8 = if config.vps.is_some() { 3 } else { 2 };

        // Profile/tier/level extracted from SPS without owning the bytes.
        let byte1 = (sps_general_profile_space(&config.sps) << 6)
            | (if sps_general_tier_flag(&config.sps) {
                0x20
            } else {
                0
            })
            | (sps_general_profile_idc(&config.sps) & 0x1f);
        let general_level_idc = sps_general_level_idc(&config.sps);

        o.push(1); // configuration version
        o.push(byte1);
        o.extend_from_slice(&[0x60, 0x00, 0x00, 0x00]); // profile compatibility
        o.extend_from_slice(&[0x90, 0x00, 0x00, 0x00, 0x00, 0x00]); // constraint indicator
        o.push(general_level_idc);
        o.extend_from_slice(&[0xf0, 0x00]); // min_spatial_segmentation_idc
        o.push(0xfc); // parallelismType
        o.push(0xfd); // chromaFormat (4:2:0)
        o.push(0xf8); // bitDepthLumaMinus8 (8-bit)
        o.push(0xf8); // bitDepthChromaMinus8 (8-bit)
        o.extend_from_slice(&[0, 0]); // avgFrameRate
        o.push(0x03); // constantFrameRate=0, numTemporalLayers=0, lengthSizeMinusOne=3
        o.push(num_arrays);

        if let Some(vps) = &config.vps {
            o.push(0b1010_0000); // VPS array
            o.extend_from_slice(&1u16.to_be_bytes());
            o.extend_from_slice(&(vps.len() as u16).to_be_bytes());
            o.extend_from_slice(vps);
        }

        o.push(0b1010_0001); // SPS array
        o.extend_from_slice(&1u16.to_be_bytes());
        o.extend_from_slice(&(config.sps.len() as u16).to_be_bytes());
        o.extend_from_slice(&config.sps);

        o.push(0b1010_0010); // PPS array
        o.extend_from_slice(&1u16.to_be_bytes());
        o.extend_from_slice(&(config.pps.len() as u16).to_be_bytes());
        o.extend_from_slice(&config.pps);
    });
}

fn write_av01(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"av01", |o| {
        write_visual_sample_entry_header(o, config.width, config.height);
        write_av1c(o, config);
    });
}

fn write_av1c(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"av1C", |o| {
        let seq_header = config.av1_sequence_header.as_deref().unwrap_or(&[]);
        let av1_config = extract_av1_config(seq_header);
        let (seq_profile, seq_level_idx) = av1_config
            .as_ref()
            .map(|c| (c.seq_profile, c.seq_level_idx))
            .unwrap_or((0, 0));

        let byte1 = ((seq_profile & 0x07) << 5) | (seq_level_idx & 0x1f);
        let byte2 = av1_config
            .as_ref()
            .map(|c| {
                ((c.seq_tier & 0x01) << 7)
                    | (if c.high_bitdepth { 0x40 } else { 0 })
                    | (if c.twelve_bit { 0x20 } else { 0 })
                    | (if c.monochrome { 0x10 } else { 0 })
                    | (if c.chroma_subsampling_x { 0x08 } else { 0 })
                    | (if c.chroma_subsampling_y { 0x04 } else { 0 })
                    | (c.chroma_sample_position & 0x03)
            })
            .unwrap_or(0);
        let obu_bytes = av1_config
            .as_ref()
            .map(|c| c.sequence_header.as_slice())
            .unwrap_or(seq_header);

        o.push(0x81); // marker (1) + version (1)
        o.push(byte1);
        o.push(byte2);
        o.push(0x00); // no initial presentation delay
        o.extend_from_slice(obu_bytes);
    });
}

fn write_vp09(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"vp09", |o| {
        write_visual_sample_entry_header(o, config.width, config.height);
        write_vpcc(o, config);
    });
}

fn write_vpcc(out: &mut Vec<u8>, config: &FragmentConfig) {
    write_box(out, b"vpcC", |o| {
        if let Some(vp9_config) = &config.vp9_config {
            o.push(1);
            o.push(vp9_config.profile);
            o.push(vp9_config.level);
            o.push(vp9_config.bit_depth);
            o.push(vp9_config.color_space);
            o.push(vp9_config.transfer_function);
            o.push(vp9_config.matrix_coefficients);
            o.push(vp9_config.full_range_flag);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_box_offset(data: &[u8], typ: &[u8; 4]) -> Option<usize> {
        data.windows(4)
            .position(|w| w == typ)
            .and_then(|pos| pos.checked_sub(4))
    }

    fn read_u32_be(data: &[u8], offset: usize) -> u32 {
        u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap())
    }

    fn read_u64_be(data: &[u8], offset: usize) -> u64 {
        u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap())
    }

    fn h264_config() -> FragmentConfig {
        FragmentConfig::default()
    }

    #[test]
    fn init_segment_contains_ftyp_moov() {
        let muxer = FragmentedMuxer::new(h264_config());
        let mut init = Vec::new();
        muxer.write_init(&mut init);

        assert_eq!(&init[4..8], b"ftyp");
        let ftyp_size = u32::from_be_bytes(init[0..4].try_into().unwrap()) as usize;
        assert_eq!(&init[ftyp_size + 4..ftyp_size + 8], b"moov");
    }

    #[test]
    fn fragment_contains_moof_mdat() {
        let muxer = FragmentedMuxer::new(h264_config());
        let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xaa, 0xbb, 0xcc, 0xdd];
        let samples = [
            SampleSpec {
                frame: &frame,
                pts: 0,
                dts: 0,
                is_sync: true,
            },
            SampleSpec {
                frame: &frame,
                pts: 3000,
                dts: 3000,
                is_sync: false,
            },
        ];

        let mut out = Vec::new();
        muxer.write_fragment(&mut out, 1, 0, &samples).unwrap();

        assert_eq!(&out[4..8], b"moof");
        let moof_size = u32::from_be_bytes(out[0..4].try_into().unwrap()) as usize;
        assert_eq!(&out[moof_size + 4..moof_size + 8], b"mdat");
    }

    #[test]
    fn fragment_with_no_samples_is_noop() {
        let muxer = FragmentedMuxer::new(h264_config());
        let mut out = Vec::new();
        muxer.write_fragment(&mut out, 1, 0, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn non_monotonic_dts_is_rejected() {
        let muxer = FragmentedMuxer::new(h264_config());
        let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xaa];
        let samples = [
            SampleSpec {
                frame: &frame,
                pts: 0,
                dts: 100,
                is_sync: true,
            },
            SampleSpec {
                frame: &frame,
                pts: 3000,
                dts: 50,
                is_sync: false,
            },
        ];
        let mut out = Vec::new();
        let err = muxer.write_fragment(&mut out, 1, 0, &samples).unwrap_err();
        assert_eq!(
            err,
            FragmentedError::NonMonotonicDts {
                prev_dts: 100,
                curr_dts: 50
            }
        );
    }

    #[test]
    fn tfdt_carries_caller_supplied_decode_time() {
        let muxer = FragmentedMuxer::new(h264_config());
        let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xaa];
        let samples = [SampleSpec {
            frame: &frame,
            pts: 0,
            dts: 0,
            is_sync: true,
        }];

        let mut out = Vec::new();
        muxer.write_fragment(&mut out, 5, 12345, &samples).unwrap();

        let tfdt_off = find_box_offset(&out, b"tfdt").expect("tfdt box");
        // payload: version+flags (4), decode time (8) starting at tfdt_off + 8
        let base = read_u64_be(&out, tfdt_off + 8 + 4);
        assert_eq!(base, 12345);
    }

    #[test]
    fn mfhd_carries_caller_sequence_number() {
        let muxer = FragmentedMuxer::new(h264_config());
        let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xaa];
        let samples = [SampleSpec {
            frame: &frame,
            pts: 0,
            dts: 0,
            is_sync: true,
        }];

        let mut out = Vec::new();
        muxer.write_fragment(&mut out, 42, 0, &samples).unwrap();

        let mfhd_off = find_box_offset(&out, b"mfhd").expect("mfhd box");
        let seq = read_u32_be(&out, mfhd_off + 8 + 4);
        assert_eq!(seq, 42);
    }

    #[test]
    fn trun_single_sample_uses_default_duration_3000() {
        let muxer = FragmentedMuxer::new(h264_config());
        let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xaa];
        let samples = [SampleSpec {
            frame: &frame,
            pts: 0,
            dts: 0,
            is_sync: true,
        }];

        let mut out = Vec::new();
        muxer.write_fragment(&mut out, 1, 0, &samples).unwrap();

        let trun_off = find_box_offset(&out, b"trun").expect("trun box");
        // payload: version+flags(4), sample_count(4), data_offset(4), then sample_duration(4)
        let duration = read_u32_be(&out, trun_off + 8 + 12);
        assert_eq!(duration, 3000);
    }

    #[test]
    fn fragment_buffer_can_be_reused_across_calls() {
        let muxer = FragmentedMuxer::new(h264_config());
        let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xaa];
        let samples = [SampleSpec {
            frame: &frame,
            pts: 0,
            dts: 0,
            is_sync: true,
        }];

        let mut out = Vec::with_capacity(4096);
        muxer.write_fragment(&mut out, 1, 0, &samples).unwrap();
        let cap_after_first = out.capacity();
        out.clear();
        muxer.write_fragment(&mut out, 2, 3000, &samples).unwrap();
        assert_eq!(out.capacity(), cap_after_first);
    }

    #[test]
    fn moof_size_for_matches_actual_emit() {
        let muxer = FragmentedMuxer::new(h264_config());
        let frame = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0xaa];
        let samples: Vec<SampleSpec> = (0..30)
            .map(|i| SampleSpec {
                frame: &frame,
                pts: i * 3000,
                dts: i * 3000,
                is_sync: i == 0,
            })
            .collect();

        let mut out = Vec::new();
        muxer.write_fragment(&mut out, 1, 0, &samples).unwrap();

        let moof_size = u32::from_be_bytes(out[0..4].try_into().unwrap()) as usize;
        assert_eq!(moof_size, moof_size_for(samples.len()));
    }
}
