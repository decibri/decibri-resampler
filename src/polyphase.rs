//! Polyphase resampler with constructor dispatch.
//!
//! [`PolyphaseResampler`] converts between any pair of integer-Hz sample rates.
//! The constructor reduces `out/in` to lowest terms `L/M` and selects one of
//! three internal engines:
//!
//! - identity passthrough when `in == out`;
//! - the exact rational engine when the phase count `L` is small enough
//!   ([`MAX_EXACT_PHASES`]): `L` polyphase phases, exact rational stepping, no
//!   interpolation;
//! - the arbitrary-ratio general engine otherwise: a fixed [`GENERAL_PHASES`]
//!   oversampled prototype with an exact-rational fixed-point phase accumulator
//!   and linear interpolation between adjacent phases.
//!
//! All three share the unit-2 push-and-drain streaming discipline (a fixed-size
//! input-history ring, outputs emitted as their newest tap arrives), so chunk
//! boundaries are invisible in the output for every engine.

use crate::kernel;
use crate::resampler::Resampler;
use crate::ResamplerError;
use tracing::{debug, warn};

/// Phases the arbitrary-ratio general engine oversamples its prototype to.
const GENERAL_PHASES: usize = 512;

/// Largest reduced phase count `L` the exact engine handles; above this the
/// general engine is used.
///
/// The exact table is `L * tpp` f32 values and grows linearly with `L`; the
/// general table is fixed at `GENERAL_PHASES * tpp`. Every common audio rate
/// reduces to `L <= 640` when converting to 16 kHz or 48 kHz (the largest is
/// 11025, with `L = 640`), so 1024 routes all of them to the exact engine with
/// margin while keeping the exact table near or below the general table's size.
/// Only genuinely large-`L` rates (oddball or pathological integer rates, whose
/// exact table would be many megabytes) fall through to the bounded general
/// engine.
const MAX_EXACT_PHASES: usize = 1024;

/// Maximum supported windowed-sinc prototype length, in taps. Construction
/// rejects a rate pair whose prototype would exceed this length.
const MAX_FILTER_TAPS: usize = 1 << 22;

/// Greatest common divisor (Euclid).
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ===== Exact rational engine ================================================

/// Exact rational `L/M` polyphase engine: `L` phases, `M` decimation, exact
/// integer phase stepping, no interpolation.
struct ExactEngine {
    l: usize,
    m: usize,
    tpp: usize,
    table: Vec<f32>,
    ring: Vec<f32>,
    head: usize,
    phase: usize,
    input_base: u64,
    next_in: u64,
    latency: usize,
}

impl ExactEngine {
    fn new(in_rate: u32, out_rate: u32, l: usize, m: usize) -> Self {
        let proto =
            kernel::build_prototype(in_rate, out_rate, m as u32, kernel::DESIGN_STOPBAND_DB);
        let n = proto.num_taps;
        let tpp = n.div_ceil(l);

        // Decompose into L phases: phase p, tap j is h[p + L*j], scaled by L for
        // unity passband gain. Indices at or past N are the zero padding.
        let l_f = l as f64;
        let mut table = vec![0.0_f32; l * tpp];
        for (flat, slot) in table.iter_mut().enumerate() {
            let p = flat / tpp;
            let j = flat % tpp;
            let idx = p + l * j;
            if idx < n {
                *slot = (proto.coeffs[idx] * l_f) as f32;
            }
        }

        Self {
            l,
            m,
            tpp,
            table,
            ring: vec![0.0_f32; tpp],
            head: tpp - 1,
            phase: 0,
            input_base: 0,
            next_in: 0,
            latency: proto.group_delay_out,
        }
    }

    fn compute_output(&self) -> f32 {
        let base = self.phase * self.tpp;
        let taps = &self.table[base..base + self.tpp];
        let head = self.head;
        let tpp = self.tpp;
        let mut acc = 0.0_f64;
        let mut j = 0usize;
        while j <= head {
            acc += taps[j] as f64 * self.ring[head - j] as f64;
            j += 1;
        }
        while j < tpp {
            acc += taps[j] as f64 * self.ring[head + tpp - j] as f64;
            j += 1;
        }
        acc as f32
    }

    fn push_and_drain(&mut self, sample: f32, output: &mut Vec<f32>) {
        self.head = (self.head + 1) % self.tpp;
        self.ring[self.head] = sample;
        let idx = self.next_in;
        self.next_in += 1;

        while self.input_base == idx {
            let y = self.compute_output();
            output.push(y);
            self.phase += self.m;
            self.input_base += (self.phase / self.l) as u64;
            self.phase %= self.l;
        }
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        for &sample in input {
            self.push_and_drain(sample, output);
        }
    }

    fn flush(&mut self, output: &mut Vec<f32>) {
        for _ in 0..(self.tpp - 1) {
            self.push_and_drain(0.0, output);
        }
    }

    fn reset(&mut self) {
        self.ring.fill(0.0);
        self.head = self.tpp - 1;
        self.phase = 0;
        self.input_base = 0;
        self.next_in = 0;
    }
}

// ===== Arbitrary-ratio general engine =======================================

/// Arbitrary-ratio engine: a `GENERAL_PHASES`-phase oversampled prototype with
/// an exact-rational fixed-point phase accumulator and linear interpolation
/// between adjacent phases.
struct GeneralEngine {
    in_rate: u64,
    out_rate: u64,
    phases: usize,
    tpp: usize,
    table: Vec<f32>,
    ring: Vec<f32>,
    head: usize,
    /// Absolute input index of the newest tap the next output needs.
    input_base: u64,
    /// Sub-sample position numerator over `out_rate`, in `0..out_rate`.
    rem: u64,
    next_in: u64,
    /// Phase and interpolation fraction for the current output (set per output).
    phase: usize,
    eta: f64,
    latency: usize,
}

impl GeneralEngine {
    fn new(in_rate: u32, out_rate: u32) -> Self {
        let phases = GENERAL_PHASES;
        let (coeffs, n) = kernel::build_oversampled_prototype(
            in_rate,
            out_rate,
            phases,
            kernel::DESIGN_STOPBAND_DB,
        );
        let tpp = n.div_ceil(phases);

        // Phase p, tap j is h[p + phases*j], scaled by `phases` for unity gain.
        let scale = phases as f64;
        let mut table = vec![0.0_f32; phases * tpp];
        for (flat, slot) in table.iter_mut().enumerate() {
            let p = flat / tpp;
            let j = flat % tpp;
            let idx = p + phases * j;
            if idx < n {
                *slot = (coeffs[idx] * scale) as f32;
            }
        }

        // Impulse response peak (group delay) at output index
        // round((N-1) * out / (2 * phases * in)), since y[n] = h(n*in/out) and h
        // peaks at its center (N-1)/(2*phases) input samples.
        let num = (n as u64 - 1) * u64::from(out_rate);
        let den = 2 * phases as u64 * u64::from(in_rate);
        let latency = ((num + den / 2) / den) as usize;

        Self {
            in_rate: u64::from(in_rate),
            out_rate: u64::from(out_rate),
            phases,
            tpp,
            table,
            ring: vec![0.0_f32; tpp],
            head: tpp - 1,
            input_base: 0,
            rem: 0,
            next_in: 0,
            phase: 0,
            eta: 0.0,
            latency,
        }
    }

    fn compute_output(&self) -> f32 {
        let tpp = self.tpp;
        let head = self.head;
        let eta = self.eta;
        let base_p = self.phase * tpp;
        // The next phase for interpolation; the top phase wraps to phase 0 of the
        // next input sample (tap j+1).
        let wrap = self.phase + 1 == self.phases;
        let base_next = if wrap { 0 } else { (self.phase + 1) * tpp };

        let coef = |j: usize| -> f64 {
            let a = self.table[base_p + j] as f64;
            let b = if wrap {
                if j + 1 < tpp {
                    self.table[base_next + j + 1] as f64
                } else {
                    0.0
                }
            } else {
                self.table[base_next + j] as f64
            };
            a + eta * (b - a)
        };

        let mut acc = 0.0_f64;
        let mut j = 0usize;
        while j <= head {
            acc += coef(j) * self.ring[head - j] as f64;
            j += 1;
        }
        while j < tpp {
            acc += coef(j) * self.ring[head + tpp - j] as f64;
            j += 1;
        }
        acc as f32
    }

    fn push_and_drain(&mut self, sample: f32, output: &mut Vec<f32>) {
        self.head = (self.head + 1) % self.tpp;
        self.ring[self.head] = sample;
        let idx = self.next_in;
        self.next_in += 1;

        while self.input_base == idx {
            // Sub-sample position from the current remainder, by exact integer
            // math: phase = floor(rem*P / out), eta = (rem*P mod out) / out.
            let rp = self.rem * self.phases as u64;
            self.phase = (rp / self.out_rate) as usize;
            self.eta = (rp % self.out_rate) as f64 / self.out_rate as f64;

            let y = self.compute_output();
            output.push(y);

            // Advance the input position by exactly in/out (Bresenham).
            self.rem += self.in_rate;
            self.input_base += self.rem / self.out_rate;
            self.rem %= self.out_rate;
        }
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        for &sample in input {
            self.push_and_drain(sample, output);
        }
    }

    fn flush(&mut self, output: &mut Vec<f32>) {
        for _ in 0..(self.tpp - 1) {
            self.push_and_drain(0.0, output);
        }
    }

    fn reset(&mut self) {
        self.ring.fill(0.0);
        self.head = self.tpp - 1;
        self.input_base = 0;
        self.rem = 0;
        self.next_in = 0;
        self.phase = 0;
        self.eta = 0.0;
    }
}

// ===== Public resampler with dispatch =======================================

enum Engine {
    Identity,
    Exact(ExactEngine),
    General(GeneralEngine),
}

/// A streaming sample-rate resampler.
///
/// Construct it with [`PolyphaseResampler::new`] for an input and output rate,
/// then drive it through the [`Resampler`] trait.
///
/// ```
/// use decibri_resampler::{PolyphaseResampler, Resampler};
///
/// let mut resampler = PolyphaseResampler::new(48_000, 16_000).unwrap();
/// let mut output = Vec::new();
/// resampler.process(&[0.0_f32; 480], &mut output);
/// resampler.flush(&mut output);
/// assert!(!output.is_empty());
/// ```
pub struct PolyphaseResampler {
    engine: Engine,
    /// Set by [`flush`](Resampler::flush), cleared by
    /// [`reset`](Resampler::reset) and at construction. Backs the debug-only
    /// process-after-flush-without-reset contract check.
    flushed: bool,
}

impl PolyphaseResampler {
    /// Creates a resampler converting from `in_rate` to `out_rate` (both in Hz).
    ///
    /// Returns [`ResamplerError::ZeroSampleRate`] if either rate is zero. The
    /// rate pair is fixed for the lifetime of the instance.
    pub fn new(in_rate: u32, out_rate: u32) -> Result<Self, ResamplerError> {
        if in_rate == 0 || out_rate == 0 {
            warn!(
                in_rate = in_rate,
                out_rate = out_rate,
                error = "zero_sample_rate",
                "rejected resampler construction"
            );
            return Err(ResamplerError::ZeroSampleRate);
        }

        if in_rate == out_rate {
            debug!(
                in_rate = in_rate,
                out_rate = out_rate,
                path = "identity",
                latency_samples = 0,
                "constructed resampler"
            );
            return Ok(Self {
                engine: Engine::Identity,
                flushed: false,
            });
        }

        let g = gcd(out_rate, in_rate);
        let l = (out_rate / g) as usize;
        let m = (in_rate / g) as usize;

        // Reject a rate pair whose prototype would exceed the maximum supported
        // length, computed for the engine dispatch selects, before building or
        // allocating anything.
        let use_exact = l <= MAX_EXACT_PHASES;
        let taps = if use_exact {
            kernel::exact_prototype_len(in_rate, out_rate, m as u32, kernel::DESIGN_STOPBAND_DB)
        } else {
            kernel::general_prototype_len(
                in_rate,
                out_rate,
                GENERAL_PHASES,
                kernel::DESIGN_STOPBAND_DB,
            )
        };
        if taps > MAX_FILTER_TAPS {
            warn!(
                in_rate = in_rate,
                out_rate = out_rate,
                taps = taps,
                max = MAX_FILTER_TAPS,
                error = "rate_pair_unsupported",
                "rejected resampler construction"
            );
            return Err(ResamplerError::RatePairUnsupported { in_rate, out_rate });
        }

        let engine = if use_exact {
            Engine::Exact(ExactEngine::new(in_rate, out_rate, l, m))
        } else {
            Engine::General(GeneralEngine::new(in_rate, out_rate))
        };
        let resampler = Self {
            engine,
            flushed: false,
        };
        debug!(
            in_rate = in_rate,
            out_rate = out_rate,
            path = if use_exact { "exact" } else { "general" },
            l = l,
            m = m,
            latency_samples = resampler.latency_samples(),
            "constructed resampler"
        );
        Ok(resampler)
    }

    /// Engine path label for diagnostics: `"identity"`, `"exact"`, or
    /// `"general"`. Metadata only; never carries sample data.
    fn engine_path(&self) -> &'static str {
        match &self.engine {
            Engine::Identity => "identity",
            Engine::Exact(_) => "exact",
            Engine::General(_) => "general",
        }
    }

    #[cfg(test)]
    fn new_forced_general(in_rate: u32, out_rate: u32) -> Self {
        Self {
            engine: Engine::General(GeneralEngine::new(in_rate, out_rate)),
            flushed: false,
        }
    }

    #[cfg(test)]
    fn engine_kind(&self) -> EngineKind {
        match self.engine {
            Engine::Identity => EngineKind::Identity,
            Engine::Exact(_) => EngineKind::Exact,
            Engine::General(_) => EngineKind::General,
        }
    }

    #[cfg(test)]
    fn as_exact(&self) -> &ExactEngine {
        match &self.engine {
            Engine::Exact(e) => e,
            _ => panic!("expected the exact engine"),
        }
    }

    #[cfg(test)]
    fn as_general(&self) -> &GeneralEngine {
        match &self.engine {
            Engine::General(e) => e,
            _ => panic!("expected the general engine"),
        }
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
enum EngineKind {
    Identity,
    Exact,
    General,
}

impl Resampler for PolyphaseResampler {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        debug_assert!(!self.flushed, "process called after flush without reset");
        match &mut self.engine {
            Engine::Identity => output.extend_from_slice(input),
            Engine::Exact(e) => e.process(input, output),
            Engine::General(e) => e.process(input, output),
        }
    }

    fn flush(&mut self, output: &mut Vec<f32>) {
        let before = output.len();
        match &mut self.engine {
            Engine::Identity => {}
            Engine::Exact(e) => e.flush(output),
            Engine::General(e) => e.flush(output),
        }
        self.flushed = true;
        debug!(tail = output.len() - before, "flushed resampler tail");
    }

    fn latency_samples(&self) -> usize {
        match &self.engine {
            Engine::Identity => 0,
            Engine::Exact(e) => e.latency,
            Engine::General(e) => e.latency,
        }
    }

    fn is_identity(&self) -> bool {
        matches!(self.engine, Engine::Identity)
    }

    fn reset(&mut self) {
        match &mut self.engine {
            Engine::Identity => {}
            Engine::Exact(e) => e.reset(),
            Engine::General(e) => e.reset(),
        }
        self.flushed = false;
        debug!(path = self.engine_path(), "reset resampler state");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Deterministic broadband test signal in [-1, 1) from an LCG.
    fn signal(n: usize) -> Vec<f32> {
        let mut state = 0x1234_5678_u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn sine(freq: f64, rate: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / rate).sin() as f32)
            .collect()
    }

    // Hann-windowed single-bin DFT amplitude of the tone at `freq`. Magnitude
    // only, so it is invariant to the sub-sample phase difference between two
    // filter designs.
    fn tone_amplitude(samples: &[f32], freq: f64, rate: f64) -> f64 {
        let m = samples.len();
        let w = 2.0 * PI * freq / rate;
        let denom = m as f64 - 1.0;
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let mut wsum = 0.0_f64;
        for (n, &s) in samples.iter().enumerate() {
            let hann = 0.5 - 0.5 * (2.0 * PI * n as f64 / denom).cos();
            let ph = w * n as f64;
            wsum += hann;
            re += hann * s as f64 * ph.cos();
            im -= hann * s as f64 * ph.sin();
        }
        2.0 * (re * re + im * im).sqrt() / wsum
    }

    fn collect_single(in_rate: u32, out_rate: u32, input: &[f32]) -> Vec<f32> {
        let mut r = PolyphaseResampler::new(in_rate, out_rate).unwrap();
        let mut out = Vec::new();
        r.process(input, &mut out);
        r.flush(&mut out);
        out
    }

    fn collect_chunked(in_rate: u32, out_rate: u32, input: &[f32], sizes: &[usize]) -> Vec<f32> {
        let mut r = PolyphaseResampler::new(in_rate, out_rate).unwrap();
        let mut out = Vec::new();
        let mut i = 0usize;
        let mut s = 0usize;
        while i < input.len() {
            let size = sizes[s % sizes.len()];
            s += 1;
            if size == 0 {
                r.process(&[], &mut out);
                continue;
            }
            let end = (i + size).min(input.len());
            r.process(&input[i..end], &mut out);
            i = end;
        }
        r.flush(&mut out);
        out
    }

    fn assert_bit_identical(a: &[f32], b: &[f32], context: &str) {
        assert_eq!(a.len(), b.len(), "length mismatch: {context}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "sample {i} differs: {context}");
        }
    }

    fn argmax_abs(samples: &[f32]) -> usize {
        let mut peak_idx = 0usize;
        let mut peak_val = f32::MIN;
        for (idx, &v) in samples.iter().enumerate() {
            if v.abs() > peak_val {
                peak_val = v.abs();
                peak_idx = idx;
            }
        }
        peak_idx
    }

    #[test]
    fn zero_rate_is_rejected() {
        assert!(matches!(
            PolyphaseResampler::new(0, 16000),
            Err(ResamplerError::ZeroSampleRate)
        ));
        assert!(matches!(
            PolyphaseResampler::new(48000, 0),
            Err(ResamplerError::ZeroSampleRate)
        ));
    }

    #[test]
    fn extreme_rate_pair_is_rejected() {
        // 4_000_000_001 -> 16_000 is coprime, routing to the general engine whose
        // prototype would far exceed MAX_FILTER_TAPS. The guard returns the error
        // before any prototype is built or table allocated.
        assert!(matches!(
            PolyphaseResampler::new(4_000_000_001, 16_000),
            Err(ResamplerError::RatePairUnsupported {
                in_rate: 4_000_000_001,
                out_rate: 16_000,
            })
        ));
    }

    #[test]
    fn cap_boundary_is_enforced() {
        // 316_800_000 -> 16_000 reduces to L=1, M=19_800 with a 4_197_601-tap
        // prototype, just over MAX_FILTER_TAPS: rejected before allocation.
        assert!(matches!(
            PolyphaseResampler::new(316_800_000, 16_000),
            Err(ResamplerError::RatePairUnsupported { .. })
        ));
        // 288_000_000 -> 16_000 reduces to L=1, M=18_000 with a 3_816_001-tap
        // prototype, just under the cap: constructs.
        assert!(PolyphaseResampler::new(288_000_000, 16_000).is_ok());
    }

    #[test]
    fn real_rates_construct() {
        // Every real rate is far under the cap and constructs; the guard rejects
        // only pathological pairs.
        for &(i, o) in &[
            (48_000u32, 16_000u32),
            (44_100, 16_000),
            (8_000, 16_000),
            (16_000, 48_000),
            (96_000, 16_000),
            (768_000, 16_000),
            (11_025, 16_000),
            (44_101, 16_000), // general path
        ] {
            assert!(
                PolyphaseResampler::new(i, o).is_ok(),
                "{i}->{o} must construct"
            );
        }
    }

    #[test]
    fn process_after_flush_then_reset_is_ok() {
        // The documented flow process -> flush -> reset -> process must not trip
        // the debug-only process-after-flush contract check.
        let input = signal(500);
        let mut r = PolyphaseResampler::new(48000, 16000).unwrap();
        let mut a = Vec::new();
        r.process(&input, &mut a);
        r.flush(&mut a);
        r.reset();
        let mut b = Vec::new();
        r.process(&input, &mut b);
        r.flush(&mut b);
        assert!(!b.is_empty());
    }

    #[test]
    fn chunk_seam_bit_invariance() {
        let strategies: &[&[usize]] = &[
            &[1],
            &[2],
            &[3],
            &[7],
            &[13],
            &[64],
            &[100],
            &[0, 1, 5, 0, 13, 64, 2, 0, 128, 3, 7],
        ];
        // Includes 44101 -> 16000, which routes to the general engine.
        for &(in_rate, out_rate) in &[
            (48000u32, 16000u32),
            (44100u32, 16000u32),
            (44101u32, 16000u32),
        ] {
            let input = signal(6000);
            let single = collect_single(in_rate, out_rate, &input);
            for sizes in strategies {
                let chunked = collect_chunked(in_rate, out_rate, &input, sizes);
                assert_bit_identical(
                    &single,
                    &chunked,
                    &format!("{in_rate}->{out_rate} chunks {sizes:?}"),
                );
            }
            println!(
                "chunk-seam {in_rate}->{out_rate}: {} outputs, bit-identical across {} chunkings",
                single.len(),
                strategies.len()
            );
        }
    }

    #[test]
    fn identity_is_byte_identical_passthrough() {
        let mut r = PolyphaseResampler::new(16000, 16000).unwrap();
        assert!(r.is_identity());
        assert_eq!(r.latency_samples(), 0);
        let input = signal(500);
        let mut out = Vec::new();
        r.process(&input, &mut out);
        let after_process = out.len();
        r.flush(&mut out);
        assert_eq!(
            out.len(),
            after_process,
            "flush must append nothing for identity"
        );
        assert_bit_identical(&out, &input, "identity passthrough");
    }

    #[test]
    fn l1_matches_direct_reference() {
        // 48k->16k is L=1, M=3: input_base = 3n, phase always 0.
        let input = signal(2000);
        let mut r = PolyphaseResampler::new(48000, 16000).unwrap();
        let mut out = Vec::new();
        r.process(&input, &mut out);
        r.flush(&mut out);

        // Independent reference using the engine's own f32 phase-0 taps.
        let ex = r.as_exact();
        let tpp = ex.tpp;
        let m = ex.m as i64;
        let taps = &ex.table[0..tpp];
        let k = input.len() as i64;
        let max_base = k - 1 + (tpp as i64 - 1);
        let mut reference = Vec::new();
        let mut n = 0i64;
        loop {
            let base = n * m;
            if base > max_base {
                break;
            }
            let mut acc = 0.0_f64;
            for (j, &tap) in taps.iter().enumerate() {
                let xi = base - j as i64;
                let xv = if xi >= 0 && (xi as usize) < input.len() {
                    input[xi as usize]
                } else {
                    0.0
                };
                acc += tap as f64 * xv as f64;
            }
            reference.push(acc as f32);
            n += 1;
        }

        assert_bit_identical(&out, &reference, "L=1 vs direct reference");
        println!(
            "L=1 48k->16k: {} outputs match an independent direct dot-product reference bit-for-bit",
            out.len()
        );
    }

    #[test]
    fn l_gt_1_matches_closed_form_reference() {
        // 44100 -> 16000 reduces to L=160, M=441. The streaming engine derives
        // (input_base, phase) incrementally (phase += M; input_base += phase/L;
        // phase %= L). This reference derives them in closed form from n*M:
        // input_base = (n*M)/L, phase = (n*M)%L, which is the rational-resampling
        // definition. Matching the two proves the incremental phase stepping and
        // the ring cursor against the definition, which a golden vector cannot.
        let input = signal(2000);
        let mut r = PolyphaseResampler::new(44100, 16000).unwrap();
        let mut out = Vec::new();
        r.process(&input, &mut out);
        r.flush(&mut out);

        // Convolve the engine's own phase taps in its newest-tap-first order so
        // the comparison isolates the indexing and stepping, not the coefficients.
        let ex = r.as_exact();
        let l = ex.l as u64;
        let m = ex.m as u64;
        let tpp = ex.tpp;
        let table = &ex.table;

        let k = input.len() as i64;
        let max_base = k - 1 + (tpp as i64 - 1);
        let mut reference = Vec::new();
        let mut n: u64 = 0;
        loop {
            let acc_idx = n * m;
            let input_base = (acc_idx / l) as i64;
            if input_base > max_base {
                break;
            }
            let phase = (acc_idx % l) as usize;
            let taps = &table[phase * tpp..phase * tpp + tpp];
            let mut acc = 0.0_f64;
            for (j, &tap) in taps.iter().enumerate() {
                let xi = input_base - j as i64;
                let xv = if xi >= 0 && (xi as usize) < input.len() {
                    input[xi as usize]
                } else {
                    0.0
                };
                acc += tap as f64 * xv as f64;
            }
            reference.push(acc as f32);
            n += 1;
        }

        assert_bit_identical(&out, &reference, "L>1 vs closed-form polyphase reference");
        println!(
            "L=160 44100->16000: {} outputs match an independent closed-form polyphase reference bit-for-bit",
            out.len()
        );
    }

    #[test]
    fn latency_matches_group_delay() {
        assert_eq!(
            PolyphaseResampler::new(48000, 16000)
                .unwrap()
                .latency_samples(),
            106
        );
        assert_eq!(
            PolyphaseResampler::new(44100, 16000)
                .unwrap()
                .latency_samples(),
            106
        );
    }

    #[test]
    fn flush_recovers_the_tail() {
        let input = signal(3000);
        let mut r = PolyphaseResampler::new(48000, 16000).unwrap();
        let tail = r.as_exact().tpp - 1;
        let mut before = Vec::new();
        r.process(&input, &mut before);
        let after_process = before.len();
        r.flush(&mut before);
        let total = before.len();
        assert!(
            total > after_process,
            "flush must emit the held tail (process {after_process}, total {total})"
        );
        let bare = input.len() * 16000 / 48000;
        assert!(
            total >= bare,
            "total {total} below bare resampled count {bare}"
        );
        println!(
            "flush 48k->16k: {} from process, {} after flush (+{} tail), tap-span tail = {}",
            after_process,
            total,
            total - after_process,
            tail
        );
    }

    #[test]
    fn reset_reproduces_fresh_output() {
        let input = signal(1500);
        let mut r = PolyphaseResampler::new(48000, 16000).unwrap();
        let mut a = Vec::new();
        r.process(&input, &mut a);
        r.flush(&mut a);
        r.reset();
        let mut b = Vec::new();
        r.process(&input, &mut b);
        r.flush(&mut b);
        assert_bit_identical(&a, &b, "reset vs fresh");

        let fresh = collect_single(48000, 16000, &input);
        assert_bit_identical(&a, &fresh, "reset vs new instance");
    }

    #[test]
    fn no_steady_state_reallocation() {
        let mut r = PolyphaseResampler::new(48000, 16000).unwrap();
        let ring_cap = r.as_exact().ring.capacity();
        let table_cap = r.as_exact().table.capacity();
        let input = signal(64);
        let mut out = Vec::with_capacity(1 << 20);
        for _ in 0..1000 {
            r.process(&input, &mut out);
        }
        r.flush(&mut out);
        assert_eq!(
            r.as_exact().ring.capacity(),
            ring_cap,
            "ring must not reallocate"
        );
        assert_eq!(
            r.as_exact().table.capacity(),
            table_cap,
            "table must not reallocate"
        );
    }

    #[test]
    fn empty_input_is_a_no_op() {
        let mut r = PolyphaseResampler::new(48000, 16000).unwrap();
        let mut out = Vec::new();
        r.process(&[], &mut out);
        assert!(out.is_empty());

        let mut id = PolyphaseResampler::new(16000, 16000).unwrap();
        let mut out2 = Vec::new();
        id.process(&[], &mut out2);
        assert!(out2.is_empty());
    }

    // ---- Unit 3: dispatch and the general engine ---------------------------

    #[test]
    fn dispatch_selects_the_right_engine() {
        assert_eq!(
            PolyphaseResampler::new(16000, 16000).unwrap().engine_kind(),
            EngineKind::Identity
        );
        // An oddball rate with L beyond the threshold routes to the general path.
        assert_eq!(
            PolyphaseResampler::new(44101, 16000).unwrap().engine_kind(),
            EngineKind::General
        );

        // Every common rate, to 16000 and to 48000, routes to the exact engine.
        let rates = [
            8000u32, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000,
        ];
        for &target in &[16000u32, 48000u32] {
            for &src in &rates {
                let kind = PolyphaseResampler::new(src, target).unwrap().engine_kind();
                let expected = if src == target {
                    EngineKind::Identity
                } else {
                    EngineKind::Exact
                };
                assert_eq!(kind, expected, "{src}->{target} routed to {kind:?}");
            }
        }
    }

    #[test]
    fn general_chunk_seam_bit_invariance() {
        // 44101 -> 16000 routes to the general engine.
        let r = PolyphaseResampler::new(44101, 16000).unwrap();
        assert_eq!(r.engine_kind(), EngineKind::General);

        let strategies: &[&[usize]] =
            &[&[1], &[2], &[3], &[7], &[13], &[64], &[0, 1, 5, 13, 0, 97]];
        let input = signal(6000);
        let single = collect_single(44101, 16000, &input);
        for sizes in strategies {
            let chunked = collect_chunked(44101, 16000, &input, sizes);
            assert_bit_identical(
                &single,
                &chunked,
                &format!("general 44101->16000 {sizes:?}"),
            );
        }
        println!(
            "general chunk-seam 44101->16000: {} outputs, bit-identical across {} chunkings",
            single.len(),
            strategies.len()
        );
    }

    #[test]
    fn general_latency_matches_impulse_peak() {
        let r = PolyphaseResampler::new(44101, 16000).unwrap();
        let reported = r.latency_samples();
        let out = collect_single(44101, 16000, &[1.0]);
        let peak = argmax_abs(&out);
        println!("general 44101->16000: impulse peak {peak}, reported latency {reported}");
        assert_eq!(peak, reported);
    }

    #[test]
    fn exact_vs_general_consistency() {
        // 44100 -> 16000 is exact (L=160) by default; force it through the
        // general engine and compare. The two use different prototype lengths,
        // so their true group delays differ by a small sub-sample fraction
        // (both round to latency 106). A sample-by-sample comparison would be
        // dominated by that timing offset, not by quality, so consistency is
        // measured in amplitude (phase-invariant): both engines must pass each
        // passband tone at the same gain, to a tight floor.
        let in_rate = 44100u32;
        let out_rate = 16000u32;
        let r_out = f64::from(out_rate);
        let mut worst_db = f64::NEG_INFINITY;
        for &f in &[500.0, 1500.0, 3000.0, 5000.0, 7000.0] {
            let input = sine(f, f64::from(in_rate), 8000);

            let mut exact = PolyphaseResampler::new(in_rate, out_rate).unwrap();
            assert_eq!(exact.engine_kind(), EngineKind::Exact);
            let mut a = Vec::new();
            exact.process(&input, &mut a);
            exact.flush(&mut a);

            let mut general = PolyphaseResampler::new_forced_general(in_rate, out_rate);
            let mut b = Vec::new();
            general.process(&input, &mut b);
            general.flush(&mut b);

            let amp_e = tone_amplitude(&a[400..a.len() - 400], f, r_out);
            let amp_g = tone_amplitude(&b[400..b.len() - 400], f, r_out);
            let diff_db = 20.0 * (amp_e - amp_g).abs().log10();
            if diff_db > worst_db {
                worst_db = diff_db;
            }
        }
        println!(
            "exact-vs-general passband amplitude consistency 44100->16000: worst {worst_db:.2} dB"
        );
        assert!(
            worst_db <= -60.0,
            "exact and general passband amplitudes differ by {worst_db:.2} dB, above the -60 dB floor"
        );
    }

    #[test]
    fn general_no_steady_state_reallocation() {
        let mut r = PolyphaseResampler::new(44101, 16000).unwrap();
        let ring_cap = r.as_general().ring.capacity();
        let table_cap = r.as_general().table.capacity();
        let input = signal(64);
        let mut out = Vec::with_capacity(1 << 20);
        for _ in 0..1000 {
            r.process(&input, &mut out);
        }
        r.flush(&mut out);
        assert_eq!(
            r.as_general().ring.capacity(),
            ring_cap,
            "ring must not reallocate"
        );
        assert_eq!(
            r.as_general().table.capacity(),
            table_cap,
            "table must not reallocate"
        );
    }
}
