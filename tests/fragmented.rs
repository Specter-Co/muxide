//! Tests for fragmented MP4 muxing

use muxide::api::VideoCodec;
use muxide::codec::vp9::Vp9Config;
use muxide::fragmented::{ColorInfo, FragmentConfig, FragmentedError, FragmentedMuxer, SampleSpec};

fn h264_config() -> FragmentConfig {
    FragmentConfig {
        codec: VideoCodec::H264,
        sps: vec![0x00, 0x00, 0x00, 0x01, 0x67],
        pps: vec![0x00, 0x00, 0x00, 0x01, 0x68],
        ..FragmentConfig::default()
    }
}

fn h265_config() -> FragmentConfig {
    FragmentConfig {
        codec: VideoCodec::H265,
        sps: vec![
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00,
        ],
        pps: vec![0x44, 0x01, 0xc0, 0x73, 0xc0, 0x4c, 0x90],
        vps: Some(vec![0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00]),
        ..FragmentConfig::default()
    }
}

fn av1_config() -> FragmentConfig {
    let seq_header = vec![
        0x0A, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ];
    FragmentConfig {
        codec: VideoCodec::Av1,
        sps: vec![],
        pps: vec![],
        av1_sequence_header: Some(seq_header),
        ..FragmentConfig::default()
    }
}

fn vp9_config() -> FragmentConfig {
    FragmentConfig {
        codec: VideoCodec::Vp9,
        sps: vec![],
        pps: vec![],
        vp9_config: Some(Vp9Config {
            width: 1920,
            height: 1080,
            profile: 0,
            bit_depth: 8,
            color_space: 0,
            transfer_function: 0,
            matrix_coefficients: 0,
            level: 0,
            full_range_flag: 0,
        }),
        ..FragmentConfig::default()
    }
}

fn write_init_and_fragment(config: FragmentConfig, sample: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let muxer = FragmentedMuxer::new(config);

    let mut init = Vec::new();
    muxer.write_init(&mut init);

    let samples = [
        SampleSpec {
            frame: sample,
            pts: 0,
            dts: 0,
            is_sync: true,
        },
        SampleSpec {
            frame: sample,
            pts: 3000,
            dts: 3000,
            is_sync: false,
        },
    ];
    let mut fragment = Vec::new();
    muxer.write_fragment(&mut fragment, 1, 0, &samples).unwrap();

    (init, fragment)
}

#[test]
fn dts_must_be_monotonic() {
    let muxer = FragmentedMuxer::new(h264_config());
    let data = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x01, 0x02, 0x03];

    let samples = [
        SampleSpec {
            frame: &data,
            pts: 0,
            dts: 3000,
            is_sync: true,
        },
        SampleSpec {
            frame: &data,
            pts: 3000,
            dts: 1000,
            is_sync: false,
        },
    ];
    let mut out = Vec::new();
    let result = muxer.write_fragment(&mut out, 1, 0, &samples);
    assert!(matches!(
        result,
        Err(FragmentedError::NonMonotonicDts {
            prev_dts: 3000,
            curr_dts: 1000
        })
    ));
}

#[test]
fn h264_init_and_fragment() {
    let data = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x01, 0x02, 0x03];
    let (init, fragment) = write_init_and_fragment(h264_config(), &data);

    assert!(init.windows(4).any(|w| w == b"ftyp"));
    assert!(init.windows(4).any(|w| w == b"moov"));
    assert!(init.windows(4).any(|w| w == b"avc1"));

    assert!(fragment.windows(4).any(|w| w == b"moof"));
    assert!(fragment.windows(4).any(|w| w == b"mdat"));
}

#[test]
fn h265_init_and_fragment() {
    let data = vec![0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xaf, 0x06];
    let (init, fragment) = write_init_and_fragment(h265_config(), &data);

    assert!(init.windows(4).any(|w| w == b"hvc1"));
    assert!(fragment.windows(4).any(|w| w == b"moof"));
    assert!(fragment.windows(4).any(|w| w == b"mdat"));
}

#[test]
fn av1_init_and_fragment() {
    let data = vec![0x12, 0x00, 0x32, 0x02, 0x00, 0x00];
    let (init, fragment) = write_init_and_fragment(av1_config(), &data);

    assert!(init.windows(4).any(|w| w == b"av01"));
    assert!(fragment.windows(4).any(|w| w == b"moof"));
    assert!(fragment.windows(4).any(|w| w == b"mdat"));
}

#[test]
fn vp9_init_and_fragment() {
    let data = vec![0x49, 0x83, 0x42, 0x00, 0x00, 0x00];
    let (init, fragment) = write_init_and_fragment(vp9_config(), &data);

    assert!(init.windows(4).any(|w| w == b"vp09"));
    assert!(fragment.windows(4).any(|w| w == b"moof"));
    assert!(fragment.windows(4).any(|w| w == b"mdat"));
}

#[test]
fn init_omits_colr_when_color_unset() {
    let data = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x01, 0x02, 0x03];
    let (init, _) = write_init_and_fragment(h264_config(), &data);
    assert!(!init.windows(4).any(|w| w == b"colr"));
}

#[test]
fn init_writes_colr_nclx_for_each_codec() {
    let full_range_bt709 = ColorInfo {
        primaries: 1,
        transfer: 1,
        matrix: 1,
        full_range: true,
    };
    for mut config in [h264_config(), h265_config(), av1_config()] {
        let codec = config.codec;
        config.color = Some(full_range_bt709);
        let muxer = FragmentedMuxer::new(config);
        let mut init = Vec::new();
        muxer.write_init(&mut init);

        let pos = init
            .windows(4)
            .position(|w| w == b"colr")
            .unwrap_or_else(|| panic!("{codec:?}: no colr box in init segment"));
        // Box payload: "nclx", primaries u16, transfer u16, matrix u16,
        // full_range_flag in the top bit of the final byte.
        let payload = &init[pos + 4..pos + 15];
        assert_eq!(&payload[0..4], b"nclx", "{codec:?}");
        assert_eq!(&payload[4..10], &[0, 1, 0, 1, 0, 1], "{codec:?}");
        assert_eq!(payload[10], 0x80, "{codec:?}");
    }
}
