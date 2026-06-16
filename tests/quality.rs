//! Quality-bar suite for the dominant rates 48000 -> 16000 and 44100 -> 16000.
//!
//! Measures the assembled resampler through its public API against the design
//! spec: alias rejection, passband flatness, impulse latency, and a
//! cross-platform determinism golden-vector guard.
//!
//! The tone tests (alias, passband, impulse) generate their inputs with the
//! standard library's sin: those tests assert dB thresholds with wide margin,
//! where a ~1e-15 input perturbation is irrelevant. The golden-vector input is
//! generated with NO transcendentals (a deterministic integer LCG), because its
//! assertion is bit-exact across platforms and std sin/cos are documented as
//! platform-dependent.

use decibri_resampler::{PolyphaseResampler, Resampler};
use std::f64::consts::PI;

// ---- Shared helpers --------------------------------------------------------

/// A pure sine of unit amplitude at `freq` (Hz), `n` samples at `rate` (Hz).
fn sine(freq: f64, rate: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * PI * freq * i as f64 / rate).sin() as f32)
        .collect()
}

/// Output rate `out_rate`, run `input` through `process` only (no flush).
fn resample_process_only(in_rate: u32, out_rate: u32, input: &[f32]) -> Vec<f32> {
    let mut r = PolyphaseResampler::new(in_rate, out_rate).unwrap();
    let mut out = Vec::new();
    r.process(input, &mut out);
    out
}

/// Run `input` through `process` then `flush` (the complete finite output).
fn resample_full(in_rate: u32, out_rate: u32, input: &[f32]) -> Vec<f32> {
    let mut r = PolyphaseResampler::new(in_rate, out_rate).unwrap();
    let mut out = Vec::new();
    r.process(input, &mut out);
    r.flush(&mut out);
    out
}

/// Amplitude of the tone at `freq` in `samples` (sampled at `rate`), via a
/// Hann-windowed single-bin DFT. The window suppresses the negative-frequency
/// image so the estimate is accurate even for low frequencies over a modest
/// region.
fn tone_amplitude(samples: &[f32], freq: f64, rate: f64) -> f64 {
    let m = samples.len();
    let w = 2.0 * PI * freq / rate;
    let denom = (m as f64) - 1.0;
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    let mut wsum = 0.0_f64;
    for (n, &s) in samples.iter().enumerate() {
        let hann = 0.5 - 0.5 * (2.0 * PI * n as f64 / denom).cos();
        let phase = w * n as f64;
        wsum += hann;
        re += hann * s as f64 * phase.cos();
        im -= hann * s as f64 * phase.sin();
    }
    2.0 * (re * re + im * im).sqrt() / wsum
}

/// The frequency an input tone `f_in` folds to in the output band.
fn fold(f_in: f64, out_rate: f64) -> f64 {
    let k = (f_in / out_rate).round();
    (f_in - k * out_rate).abs()
}

/// Samples skipped at the start of a tone run to clear the group-delay
/// transient and filter settling before measuring steady state.
const LEAD_SKIP: usize = 400;

/// Input length for the tone runs (long enough for an accurate measurement).
const TONE_INPUT_LEN: usize = 6000;

/// Reference passband frequency for the unity-gain normalization.
const REF_FREQ: f64 = 1000.0;

// ---- 1. Alias rejection ----------------------------------------------------

/// Worst-case alias rejection (dB) and its input frequency over the sweep.
fn alias_rejection(in_rate: u32, out_rate: u32, tones: &[f64]) -> (f64, f64) {
    let r = f64::from(out_rate);
    let i = f64::from(in_rate);

    // Unity passband reference.
    let reference = {
        let out = resample_process_only(in_rate, out_rate, &sine(REF_FREQ, i, TONE_INPUT_LEN));
        tone_amplitude(&out[LEAD_SKIP..], REF_FREQ, r)
    };

    let mut worst_db = f64::NEG_INFINITY;
    let mut worst_freq = 0.0;
    for &f_in in tones {
        let out = resample_process_only(in_rate, out_rate, &sine(f_in, i, TONE_INPUT_LEN));
        let f_out = fold(f_in, r);
        let amp = tone_amplitude(&out[LEAD_SKIP..], f_out, r);
        let db = 20.0 * (amp / reference).log10();
        if db > worst_db {
            worst_db = db;
            worst_freq = f_in;
        }
    }
    (worst_db, worst_freq)
}

#[test]
fn alias_rejection_48k_to_16k() {
    // Tones in [8400, 16000) fold into the protected passband [0, 7600].
    let tones = [8400.0, 9000.0, 10500.0, 12000.0, 13500.0, 15000.0];
    let (worst_db, worst_freq) = alias_rejection(48000, 16000, &tones);
    println!(
        "alias 48k->16k: worst rejection {worst_db:.2} dB at {worst_freq:.0} Hz (folds to {:.0} Hz)",
        fold(worst_freq, 16000.0)
    );
    assert!(
        worst_db <= -60.0,
        "worst alias rejection {worst_db:.2} dB must clear -60 dB"
    );
}

#[test]
fn alias_rejection_44k_to_16k() {
    // Adds 20000 Hz from the next band (folds to 4000 Hz); 44.1k Nyquist is 22050.
    let tones = [8400.0, 9000.0, 10500.0, 12000.0, 13500.0, 15000.0, 20000.0];
    let (worst_db, worst_freq) = alias_rejection(44100, 16000, &tones);
    println!(
        "alias 44.1k->16k: worst rejection {worst_db:.2} dB at {worst_freq:.0} Hz (folds to {:.0} Hz)",
        fold(worst_freq, 16000.0)
    );
    assert!(
        worst_db <= -60.0,
        "worst alias rejection {worst_db:.2} dB must clear -60 dB"
    );
}

// ---- 2. Passband flatness --------------------------------------------------

/// Passband ripple (dB) over a tone sweep across [200, 7600] Hz.
fn passband_ripple(in_rate: u32, out_rate: u32) -> f64 {
    let r = f64::from(out_rate);
    let i = f64::from(in_rate);
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;
    for step in 1..=38 {
        let f = step as f64 * 200.0; // 200 .. 7600
        let out = resample_process_only(in_rate, out_rate, &sine(f, i, TONE_INPUT_LEN));
        let amp = tone_amplitude(&out[LEAD_SKIP..], f, r);
        if amp > max {
            max = amp;
        }
        if amp < min {
            min = amp;
        }
    }
    20.0 * (max / min).log10()
}

#[test]
fn passband_flatness_48k_to_16k() {
    let ripple = passband_ripple(48000, 16000);
    println!("passband 48k->16k: ripple {ripple:.6} dB over [200, 7600] Hz (bound < 0.001 dB)");
    assert!(
        ripple < 0.001,
        "passband ripple {ripple:.6} dB must be under 0.001 dB"
    );
}

#[test]
fn passband_flatness_44k_to_16k() {
    let ripple = passband_ripple(44100, 16000);
    println!("passband 44.1k->16k: ripple {ripple:.6} dB over [200, 7600] Hz (bound < 0.001 dB)");
    assert!(
        ripple < 0.001,
        "passband ripple {ripple:.6} dB must be under 0.001 dB"
    );
}

// ---- General-path quality (arbitrary rate) ---------------------------------
//
// 44101 -> 16000 has a reduced phase count far past the exact threshold, so the
// constructor routes it to the arbitrary-ratio general engine (the dispatch is
// asserted in the engine's own unit tests). These confirm the general engine's
// realized quality meets the same bar as the exact engine.

#[test]
fn alias_rejection_general_44101_to_16000() {
    let tones = [8400.0, 9000.0, 10500.0, 12000.0, 13500.0, 15000.0, 20000.0];
    let (worst_db, worst_freq) = alias_rejection(44101, 16000, &tones);
    println!(
        "alias general 44101->16000: worst rejection {worst_db:.2} dB at {worst_freq:.0} Hz (folds to {:.0} Hz)",
        fold(worst_freq, 16000.0)
    );
    assert!(
        worst_db <= -60.0,
        "general-path worst alias rejection {worst_db:.2} dB must clear -60 dB"
    );
}

#[test]
fn passband_flatness_general_44101_to_16000() {
    let ripple = passband_ripple(44101, 16000);
    println!("passband general 44101->16000: ripple {ripple:.5} dB over [200, 7600] Hz");
    assert!(
        ripple < 0.1,
        "general-path passband ripple {ripple:.5} dB must be under 0.1 dB"
    );
}

// ---- 3. Impulse / latency --------------------------------------------------

/// Output index of the peak magnitude of the impulse response.
fn impulse_peak_index(in_rate: u32, out_rate: u32) -> usize {
    let out = resample_full(in_rate, out_rate, &[1.0]);
    let mut peak_idx = 0usize;
    let mut peak_val = f32::MIN;
    for (idx, &v) in out.iter().enumerate() {
        let mag = v.abs();
        if mag > peak_val {
            peak_val = mag;
            peak_idx = idx;
        }
    }
    peak_idx
}

#[test]
fn impulse_latency_48k_to_16k() {
    let reported = PolyphaseResampler::new(48000, 16000)
        .unwrap()
        .latency_samples();
    let peak = impulse_peak_index(48000, 16000);
    println!("impulse 48k->16k: peak at output index {peak}, reported latency {reported}");
    assert_eq!(peak, reported);
    assert_eq!(reported, 106);
}

#[test]
fn impulse_latency_44k_to_16k() {
    let reported = PolyphaseResampler::new(44100, 16000)
        .unwrap()
        .latency_samples();
    let peak = impulse_peak_index(44100, 16000);
    println!("impulse 44.1k->16k: peak at output index {peak}, reported latency {reported}");
    assert_eq!(peak, reported);
    assert_eq!(reported, 106);
}

// ---- 5. Upsampling: image rejection, flatness, latency, unity gain ---------
//
// Upsampling has no input content above the input Nyquist, so the anti-imaging
// property to measure is image rejection, not alias rejection. Inserting L-1
// zeros between input samples places spectral images of a baseband tone f0 at
// k*in +/- f0; the interpolation lowpass (cutoff 0.4875*in, stopband edge
// 0.5*in) must suppress the images that land below the output Nyquist out/2.

/// Worst-case image rejection (dB) for a baseband tone `f0`, measured at the
/// listed image frequencies relative to the passed baseband amplitude.
fn upsample_image_rejection(in_rate: u32, out_rate: u32, f0: f64, images: &[f64]) -> f64 {
    let out = resample_process_only(
        in_rate,
        out_rate,
        &sine(f0, f64::from(in_rate), TONE_INPUT_LEN),
    );
    let region = &out[LEAD_SKIP..];
    let r = f64::from(out_rate);
    let base = tone_amplitude(region, f0, r);
    let mut worst = f64::NEG_INFINITY;
    for &img in images {
        let amp = tone_amplitude(region, img, r);
        let db = 20.0 * (amp / base).log10();
        if db > worst {
            worst = db;
        }
    }
    worst
}

/// Passband ripple (dB) over a tone sweep across [200, `max_freq`] Hz.
fn upsample_passband_ripple(in_rate: u32, out_rate: u32, max_freq: f64) -> f64 {
    let r = f64::from(out_rate);
    let i = f64::from(in_rate);
    let steps = (max_freq / 200.0).round() as usize;
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;
    for step in 1..=steps {
        let f = step as f64 * 200.0;
        let out = resample_process_only(in_rate, out_rate, &sine(f, i, TONE_INPUT_LEN));
        let amp = tone_amplitude(&out[LEAD_SKIP..], f, r);
        if amp > max {
            max = amp;
        }
        if amp < min {
            min = amp;
        }
    }
    20.0 * (max / min).log10()
}

/// Baseband amplitude of a passband tone after upsampling (unity-gain check).
fn upsample_baseband_gain(in_rate: u32, out_rate: u32, f0: f64) -> f64 {
    let out = resample_process_only(
        in_rate,
        out_rate,
        &sine(f0, f64::from(in_rate), TONE_INPUT_LEN),
    );
    tone_amplitude(&out[LEAD_SKIP..], f0, f64::from(out_rate))
}

#[test]
fn image_rejection_16k_to_48k() {
    // out/in = 48000/16000 = 3 -> L=3, M=1. Output Nyquist 24000.
    // f0=1000 -> images at 16000 +/- 1000 = 15000, 17000.
    // f0=2500 -> images at 16000 +/- 2500 = 13500, 18500.
    let a = upsample_image_rejection(16000, 48000, 1000.0, &[15000.0, 17000.0]);
    let b = upsample_image_rejection(16000, 48000, 2500.0, &[13500.0, 18500.0]);
    let worst = a.max(b);
    println!(
        "image 16k->48k: worst rejection {worst:.2} dB (f0=1000: {a:.2} dB, f0=2500: {b:.2} dB)"
    );
    assert!(
        worst <= -60.0,
        "worst image rejection {worst:.2} dB must clear -60 dB"
    );
}

#[test]
fn image_rejection_8k_to_16k() {
    // out/in = 16000/8000 = 2 -> L=2, M=1. Output Nyquist 8000.
    // f0=1000 -> image at 8000 - 1000 = 7000 (below Nyquist).
    // f0=2500 -> image at 8000 - 2500 = 5500 (below Nyquist).
    let a = upsample_image_rejection(8000, 16000, 1000.0, &[7000.0]);
    let b = upsample_image_rejection(8000, 16000, 2500.0, &[5500.0]);
    let worst = a.max(b);
    println!(
        "image 8k->16k: worst rejection {worst:.2} dB (f0=1000: {a:.2} dB, f0=2500: {b:.2} dB)"
    );
    assert!(
        worst <= -60.0,
        "worst image rejection {worst:.2} dB must clear -60 dB"
    );
}

#[test]
fn passband_flatness_16k_to_48k() {
    // Sweep [200, 0.475*16000] = [200, 7600] Hz.
    let ripple = upsample_passband_ripple(16000, 48000, 7600.0);
    println!("passband 16k->48k: ripple {ripple:.6} dB over [200, 7600] Hz");
    assert!(
        ripple < 0.1,
        "passband ripple {ripple:.6} dB must be under 0.1 dB"
    );
}

#[test]
fn passband_flatness_8k_to_16k() {
    // Sweep [200, 0.475*8000] = [200, 3800] Hz.
    let ripple = upsample_passband_ripple(8000, 16000, 3800.0);
    println!("passband 8k->16k: ripple {ripple:.6} dB over [200, 3800] Hz");
    assert!(
        ripple < 0.1,
        "passband ripple {ripple:.6} dB must be under 0.1 dB"
    );
}

#[test]
fn unity_gain_upsampling() {
    // A unit-amplitude passband tone passes at unity, confirming the L phase
    // scaling is correct for the upsampling ratios too.
    let g_16_48 = upsample_baseband_gain(16000, 48000, 1000.0);
    let g_8_16 = upsample_baseband_gain(8000, 16000, 1000.0);
    println!("unity gain upsampling: 16k->48k {g_16_48:.6}, 8k->16k {g_8_16:.6} (bound 0.001)");
    assert!(
        (g_16_48 - 1.0).abs() < 0.001,
        "16k->48k baseband gain {g_16_48:.6} must be within 0.001 of unity"
    );
    assert!(
        (g_8_16 - 1.0).abs() < 0.001,
        "8k->16k baseband gain {g_8_16:.6} must be within 0.001 of unity"
    );
}

#[test]
fn impulse_latency_16k_to_48k() {
    // L=3, M=1: group delay (N-1)/(2*M) at the controlling rate = 318.
    let reported = PolyphaseResampler::new(16000, 48000)
        .unwrap()
        .latency_samples();
    let peak = impulse_peak_index(16000, 48000);
    println!("impulse 16k->48k: peak at output index {peak}, reported latency {reported}");
    assert_eq!(peak, reported);
    assert_eq!(reported, 318);
}

#[test]
fn impulse_latency_8k_to_16k() {
    // L=2, M=1: group delay (N-1)/(2*M) at the controlling rate = 212.
    let reported = PolyphaseResampler::new(8000, 16000)
        .unwrap()
        .latency_samples();
    let peak = impulse_peak_index(8000, 16000);
    println!("impulse 8k->16k: peak at output index {peak}, reported latency {reported}");
    assert_eq!(peak, reported);
    assert_eq!(reported, 212);
}

// ---- 4. Golden-vector determinism guard ------------------------------------

/// Deterministic broadband input in [-1, 1), from a linear congruential
/// generator mapped to f32 with integer-only state. No transcendentals, so the
/// sequence is bit-identical on every platform.
fn lcg_input(n: usize) -> Vec<f32> {
    let mut state = 0x1234_5678_u32;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn assert_bit_exact(got: &[f32], expected: &[f32], context: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "golden length changed for {context}: regenerate (see DECIBRI_REGEN_RESAMPLER_GOLDEN)"
    );
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "golden mismatch at {i} for {context}: got {g:?}, expected {e:?}. \
             A bit-exact mismatch is a determinism leak (FMA, denormals, reorder); \
             investigate as a bug before regenerating."
        );
    }
}

fn print_golden(name: &str, data: &[f32]) {
    let mut body = String::new();
    for (i, p) in data.iter().enumerate() {
        if i % 14 == 0 {
            body.push_str("\n    ");
        }
        body.push_str(&format!("{p:?}, "));
    }
    println!("const {name}: &[f32] = &[{body}\n];");
}

/// Resampler output for a fixed 256-sample LCG input. Regenerate via
/// `DECIBRI_REGEN_RESAMPLER_GOLDEN=1 cargo test --test quality golden -- --nocapture`,
/// paste the printed consts here, then rerun without the variable to confirm.
const EXPECTED_48K_TO_16K: &[f32] = &[
    1.3236783e-7,
    1.6383442e-6,
    -3.6475622e-6,
    6.6716275e-6,
    -9.371014e-6,
    1.0925195e-5,
    -1.4504766e-5,
    1.7143571e-5,
    -2.1521635e-5,
    2.5326615e-5,
    -3.205225e-5,
    3.9633378e-5,
    -4.7344503e-5,
    5.4967062e-5,
    -6.551661e-5,
    7.491509e-5,
    -8.4369174e-5,
    9.2571114e-5,
    -9.816043e-5,
    0.00010620572,
    -0.000109657114,
    0.00011246637,
    -0.00010987144,
    0.00010676007,
    -9.922229e-5,
    8.903174e-5,
    -7.279174e-5,
    5.1555588e-5,
    -2.4672538e-5,
    -8.383159e-6,
    5.1995587e-5,
    -0.000102872255,
    0.00016256081,
    -0.00023418556,
    0.0003159159,
    -0.00040792653,
    0.0005120216,
    -0.00062749995,
    0.00075453543,
    -0.00089292135,
    0.0010413965,
    -0.0012005584,
    0.0013722018,
    -0.0015512854,
    0.0017383562,
    -0.0019313039,
    0.002130838,
    -0.0023309558,
    0.002532657,
    -0.0027314338,
    0.0029238726,
    -0.0031082958,
    0.0032777244,
    -0.0034307945,
    0.0035606544,
    -0.0036659548,
    0.0037396029,
    -0.0037775848,
    0.0037775687,
    -0.0037316799,
    0.003634674,
    -0.0034834591,
    0.003272831,
    -0.002995741,
    0.0026506067,
    -0.0022310233,
    0.0017339666,
    -0.0011562658,
    0.00049300806,
    0.00025578748,
    -0.001091027,
    0.0020153266,
    -0.0030295579,
    0.0041322876,
    -0.0053230263,
    0.0066008368,
    -0.007959695,
    0.009396411,
    -0.010906267,
    0.012482625,
    -0.01411806,
    0.015805287,
    -0.017532313,
    0.019291777,
    -0.021073379,
    0.022860628,
    -0.024643935,
    0.026408283,
    -0.028138854,
    0.029820023,
    -0.031435378,
    0.032967713,
    -0.034399018,
    0.035710387,
    -0.036881905,
    0.03789241,
    -0.038719095,
    0.039336905,
    -0.03971749,
    0.039827503,
    -0.039625514,
    0.03905604,
    -0.03803579,
    0.03641286,
    -0.03378631,
    0.02802398,
    0.011439405,
    -0.1879234,
    0.7974884,
    -0.009476942,
    -0.51906496,
    0.10450412,
    0.041326188,
    -0.016167069,
    0.5333625,
    -0.22493911,
    0.48526636,
    -0.16609415,
    0.2522245,
    -0.08293016,
    0.13705845,
    -0.15311679,
    -0.74941,
    -0.22116971,
    -0.29367596,
    0.5445122,
    -0.09232074,
    -0.47249666,
    -0.26545596,
    0.009041864,
    0.07494823,
    0.12569803,
    0.33342427,
    -0.62721837,
    -0.21859714,
    0.13955918,
    -0.14340375,
    0.515131,
    -0.16171609,
    0.37792474,
    -0.4771257,
    -0.4640836,
    0.12710708,
    0.3438135,
    -0.14302205,
    -0.05349884,
    -0.16477232,
    0.69547147,
    -0.15849741,
    -0.081630185,
    0.20608781,
    -0.2613104,
    -0.16678011,
    0.4304715,
    -0.23896798,
    -0.069809034,
    0.8652805,
    0.14443626,
    0.46607327,
    -0.15356632,
    0.7513489,
    -0.19654153,
    -0.113413975,
    -0.22136211,
    0.051113945,
    -0.0917417,
    -0.31023198,
    -0.07400962,
    -0.11659687,
    -0.2729347,
    0.31062222,
    0.25789568,
    -0.23301426,
    -0.3796696,
    -0.5967793,
    0.074406534,
    0.14987528,
    0.38898388,
    0.4564841,
    0.44232577,
    0.32505855,
    -0.5678287,
    -0.79671514,
    -0.26616842,
    0.5064691,
    0.36958957,
    -0.4165966,
    -0.22856428,
    -0.17595431,
    0.38025165,
    0.022212466,
    0.027814396,
    0.015860168,
    -0.010472969,
    0.008931578,
    -0.008551434,
    0.008606527,
    -0.008835323,
    0.009124463,
    -0.00941806,
    0.009685948,
    -0.009910905,
    0.0100829145,
    -0.010196331,
    0.010248394,
    -0.010238365,
    0.010166998,
    -0.010036203,
    0.009848792,
    -0.009608292,
    0.009318801,
    -0.008984854,
    0.008611319,
    -0.008204154,
    0.007768058,
    -0.0073065874,
    0.006825535,
    -0.0063305437,
    0.005829987,
    -0.0053247646,
    0.0048215725,
    -0.004322601,
    0.0038329868,
    -0.0033564842,
    0.0028962838,
    -0.0024548878,
    0.0020354798,
    -0.0016407946,
    0.0012704225,
    -0.00092871155,
    0.00061320193,
    -0.00032501933,
    6.9885085e-5,
    0.0001581897,
    -0.0003600908,
    0.0005358444,
    -0.00068323995,
    0.00080529856,
    -0.00090174197,
    0.0009770761,
    -0.0010337952,
    0.0010703349,
    -0.0010902057,
    0.0010945721,
    -0.0010822025,
    0.0010596367,
    -0.0010255411,
    0.0009825837,
    -0.000931523,
    0.00087898277,
    -0.0008204774,
    0.00075863954,
    -0.0006952815,
    0.0006306593,
    -0.00056641863,
    0.000502965,
    -0.0004430554,
    0.0003838054,
    -0.0003297548,
    0.0002797371,
    -0.00023184353,
    0.00018952183,
    -0.00014835197,
    0.000113982875,
    -8.280368e-5,
    5.6074517e-5,
    -3.2046442e-5,
    1.3182415e-5,
    2.4636943e-6,
    -1.6467906e-5,
    2.7857164e-5,
    -3.5955978e-5,
    4.090687e-5,
    -4.4594828e-5,
    4.7387202e-5,
    -4.9159433e-5,
    4.919649e-5,
    -4.764586e-5,
    4.6665402e-5,
    -4.3306885e-5,
    4.0660343e-5,
    -3.8644775e-5,
    3.6964128e-5,
    -3.1904197e-5,
    2.9629042e-5,
    -2.5826981e-5,
    2.247817e-5,
    -2.0002853e-5,
    1.6421409e-5,
    -1.2295315e-5,
    1.21696485e-5,
    -9.072759e-6,
    5.081709e-6,
    -4.1914964e-6,
    2.8498785e-6,
    -4.737662e-7,
    -1.3497262e-7,
    -3.1756824e-7,
];
const EXPECTED_44K_TO_16K: &[f32] = &[
    1.440736e-7,
    1.2482598e-6,
    -3.5046874e-6,
    7.032662e-6,
    -8.717297e-6,
    1.002825e-5,
    -1.0762542e-5,
    1.4814694e-5,
    -1.7275492e-5,
    2.0877642e-5,
    -2.3121585e-5,
    2.2569406e-5,
    -2.042638e-5,
    1.8820841e-5,
    -1.5064813e-5,
    1.0048108e-5,
    -2.5238007e-6,
    -6.1671108e-6,
    1.4008284e-5,
    -2.8191123e-5,
    3.9014532e-5,
    -5.2121366e-5,
    7.177231e-5,
    -9.233529e-5,
    0.00011656244,
    -0.0001408988,
    0.00016681809,
    -0.00019362426,
    0.00021980448,
    -0.00024818353,
    0.00027603036,
    -0.0003081025,
    0.00033595067,
    -0.00036616923,
    0.00039542123,
    -0.0004207034,
    0.0004415328,
    -0.00045970944,
    0.00047138263,
    -0.0004818868,
    0.00048196543,
    -0.00047591593,
    0.00046291962,
    -0.00044070277,
    0.00040804557,
    -0.0003683643,
    0.00031852498,
    -0.00025831725,
    0.00018646577,
    -0.00010268407,
    8.143289e-6,
    9.8126715e-5,
    -0.00021649586,
    0.000344711,
    -0.00048671736,
    0.0006424859,
    -0.00080224604,
    0.0009753484,
    -0.0011554502,
    0.0013418057,
    -0.0015325749,
    0.0017260084,
    -0.0019230454,
    0.002119208,
    -0.002314789,
    0.0025082862,
    -0.0026958992,
    0.0028749602,
    -0.0030455906,
    0.0032012856,
    -0.0033458055,
    0.0034750598,
    -0.0035843514,
    0.0036725432,
    -0.0037397023,
    0.0037807168,
    -0.0037963467,
    0.003780845,
    -0.0037361148,
    0.0036618866,
    -0.0035505644,
    0.0034067326,
    -0.0032249931,
    0.0030040448,
    -0.0027451536,
    0.0024443294,
    -0.0020990644,
    0.0017077058,
    -0.001267812,
    0.0007746112,
    -0.0002264323,
    -0.00038547418,
    0.0010719728,
    -0.0018383467,
    0.0027007167,
    -0.0036774052,
    0.0047938745,
    -0.006085501,
    0.007602523,
    -0.009418381,
    0.011644629,
    -0.014459785,
    0.018171722,
    -0.023376515,
    0.031486064,
    -0.0476109,
    0.113325424,
    -0.32863325,
    0.77704483,
    0.27259484,
    -0.54212946,
    -0.13403028,
    0.16117133,
    -0.0645265,
    0.17051274,
    0.49295968,
    -0.351271,
    0.6829663,
    -0.35490024,
    0.36867458,
    -0.09968044,
    0.050840758,
    0.13159415,
    -0.7767517,
    -0.32829332,
    -0.4041938,
    0.115793124,
    0.42083767,
    -0.15970139,
    -0.57670516,
    -0.12398472,
    -0.1300115,
    0.21363464,
    -0.049961686,
    0.4830484,
    -0.34522936,
    -0.59448385,
    0.16768521,
    -0.10673118,
    0.11047319,
    0.4207691,
    -0.17457412,
    0.40826243,
    -0.49114424,
    -0.5154298,
    0.0520353,
    0.35213685,
    0.026205558,
    -0.14833121,
    -0.109860994,
    0.11009997,
    0.71025145,
    -0.46950334,
    0.16825537,
    0.07318495,
    -0.19704106,
    -0.2439983,
    0.40871426,
    0.0022893061,
    -0.42969987,
    0.7937393,
    0.396246,
    0.31790686,
    0.23862886,
    -0.021216504,
    0.8050926,
    -0.39067075,
    -0.004203722,
    -0.291717,
    0.05840607,
    -0.029327959,
    -0.31379086,
    -0.14483906,
    -0.07210077,
    -0.19823927,
    -0.17472981,
    0.52206796,
    0.0135446265,
    -0.15168315,
    -0.5048212,
    -0.50297123,
    -0.07607352,
    0.25919148,
    0.20049672,
    0.58368945,
    0.31976813,
    0.55859303,
    -0.078564346,
    -0.7212827,
    -0.7491651,
    -0.09206372,
    0.5151929,
    0.40208685,
    -0.44272575,
    -0.21429466,
    -0.2641745,
    0.26677746,
    0.23025267,
    -0.07769834,
    0.103266686,
    -0.06146697,
    0.051085424,
    -0.04557408,
    0.041775316,
    -0.038777146,
    0.03620998,
    -0.033894908,
    0.031736203,
    -0.02967982,
    0.02769474,
    -0.025763487,
    0.023876993,
    -0.022031508,
    0.020225767,
    -0.018462166,
    0.016747708,
    -0.015083434,
    0.013470774,
    -0.011919602,
    0.010429424,
    -0.0090085,
    0.0076599806,
    -0.006384638,
    0.005189213,
    -0.004071724,
    0.003033815,
    -0.0020779292,
    0.0012039506,
    -0.0004129094,
    -0.0002979951,
    0.0009282332,
    -0.0014800536,
    0.0019597104,
    -0.0023681293,
    0.0027061058,
    -0.0029822567,
    0.0031929975,
    -0.0033504954,
    0.0034566687,
    -0.0035149448,
    0.003532796,
    -0.0035089122,
    0.0034502326,
    -0.0033609548,
    0.0032499263,
    -0.0031153443,
    0.0029636328,
    -0.0027971624,
    0.0026226507,
    -0.0024383566,
    0.0022495314,
    -0.0020582206,
    0.0018722721,
    -0.0016867494,
    0.0015054587,
    -0.0013320774,
    0.0011668203,
    -0.0010064183,
    0.0008581262,
    -0.0007215764,
    0.0005957246,
    -0.0004810649,
    0.00037518775,
    -0.0002803701,
    0.00019885284,
    -0.00012645098,
    6.4296386e-5,
    -1.076759e-5,
    -3.645602e-5,
    7.4126765e-5,
    -0.00010357631,
    0.00012882001,
    -0.00014458504,
    0.00015535005,
    -0.0001613434,
    0.00016391768,
    -0.00016196517,
    0.00015569036,
    -0.00014989164,
    0.00014192887,
    -0.00013212171,
    0.0001225412,
    -0.000110634304,
    9.9871235e-5,
    -8.9277026e-5,
    7.85239e-5,
    -6.984338e-5,
    5.9372196e-5,
    -4.8552414e-5,
    4.2496642e-5,
    -3.4724177e-5,
    2.8047658e-5,
    -2.2500339e-5,
    1.828502e-5,
    -1.6376134e-5,
    1.0993073e-5,
    -8.874531e-6,
    7.1880627e-6,
    -3.0820986e-6,
    1.0248119e-6,
    -1.1587134e-6,
    3.9494682e-7,
    1.4976306e-6,
    -1.7672623e-6,
    9.0096854e-8,
];

#[test]
fn golden_vectors_are_bit_exact() {
    let input = lcg_input(256);
    let got_48 = resample_full(48000, 16000, &input);
    let got_44 = resample_full(44100, 16000, &input);

    if std::env::var("DECIBRI_REGEN_RESAMPLER_GOLDEN").is_ok() {
        print_golden("EXPECTED_48K_TO_16K", &got_48);
        print_golden("EXPECTED_44K_TO_16K", &got_44);
        panic!(
            "DECIBRI_REGEN_RESAMPLER_GOLDEN is set: copy the printed consts into \
             tests/quality.rs, then rerun without the variable to confirm green."
        );
    }

    assert_bit_exact(&got_48, EXPECTED_48K_TO_16K, "48k->16k");
    assert_bit_exact(&got_44, EXPECTED_44K_TO_16K, "44.1k->16k");
}
