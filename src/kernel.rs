//! Windowed-sinc prototype filter construction.
//!
//! Builds the Kaiser-windowed sinc lowpass prototype that an L/M polyphase
//! resampler decomposes into phases. The prototype is designed at the common
//! rate `fs_op = M * fout` for the reduced ratio `out/in = L/M`, band-limited
//! to the controlling Nyquist `min(in, out) / 2`. All coefficient arithmetic
//! uses the deterministic special functions in [`crate::special`] (no
//! standard-library transcendentals, no fused multiply-add, fixed evaluation
//! order), so coefficients are reproducible bit-for-bit across platforms.

use crate::special;
use std::f64::consts::PI;

/// Attenuation the prototype is designed to, in dB.
///
/// For a Kaiser window the stopband floor is set by `beta` (the window shape),
/// while the length sets the transition width; raising the design attenuation
/// raises `beta` and so the realized floor. The Kaiser length formula is also
/// slightly optimistic at the band edge. Designing to 84 dB leaves the measured
/// stopband comfortably past the 80 dB target rather than sitting on it.
pub(crate) const DESIGN_STOPBAND_DB: f64 = 84.0;

/// A windowed-sinc lowpass prototype.
pub(crate) struct Prototype {
    /// Prototype coefficients, length [`Prototype::num_taps`], symmetric, in
    /// `f64` with unity DC gain.
    pub coeffs: Vec<f64>,
    /// Filter length N. Odd, with `N - 1` a multiple of `2 * M`.
    pub num_taps: usize,
    /// Group delay in output samples, `(N - 1) / (2 * M)`, an exact integer.
    pub group_delay_out: usize,
}

/// The unrounded Kaiser windowed-sinc length for operating rate `fs_op` and
/// controlling Nyquist base `control`, designed to `design_db` attenuation:
/// `numtaps = (design_db - 7.95) / (2.285 * pi * width) + 1`, with
/// `width = 2 * transition / fs_op` and `transition = 0.025 * control`.
fn kaiser_len(fs_op: f64, control: f64, design_db: f64) -> usize {
    let transition = 0.025 * control;
    let width = 2.0 * transition / fs_op;
    let n0 = (design_db - 7.95) / (2.285 * PI * width) + 1.0;
    n0.ceil() as usize
}

/// The prototype length the exact engine uses for this rate pair and decimation
/// factor `m`: the Kaiser length at operating rate `m * out_rate`, rounded up so
/// `N - 1` is a multiple of `2 * m`. Shared with [`build_prototype`] as the one
/// source of truth for the exact-engine length.
pub(crate) fn exact_prototype_len(in_rate: u32, out_rate: u32, m: u32, design_db: f64) -> usize {
    let fs_op = f64::from(m) * f64::from(out_rate);
    let control = f64::from(in_rate.min(out_rate));
    let mut n = kaiser_len(fs_op, control, design_db);
    let two_m = 2 * m as usize;
    let rem = (n - 1) % two_m;
    if rem != 0 {
        n += two_m - rem;
    }
    n
}

/// The prototype length the general engine uses for this rate pair and phase
/// count: the Kaiser length at operating rate `phases * in_rate`, rounded up to
/// an odd length. Shared with [`build_oversampled_prototype`] as the one source
/// of truth for the general-engine length.
pub(crate) fn general_prototype_len(
    in_rate: u32,
    out_rate: u32,
    phases: usize,
    design_db: f64,
) -> usize {
    let fs_op = phases as f64 * f64::from(in_rate);
    let control = f64::from(in_rate.min(out_rate));
    let mut n = kaiser_len(fs_op, control, design_db);
    if n.is_multiple_of(2) {
        n += 1;
    }
    n
}

/// Builds the windowed-sinc prototype for the rate pair `in_rate` to `out_rate`
/// and decimation factor `m` (the denominator of the reduced ratio
/// `out/in = L/M`), designed to `design_db` stopband attenuation.
///
/// Geometry is referenced to the controlling rate `min(in_rate, out_rate)`:
/// passband edge `0.475`, stopband edge `0.5`, transition `0.025`, and cutoff
/// at the transition midpoint `0.4875`, all times the controlling rate. The
/// prototype is designed at the common rate `fs_op = m * out_rate`. The length
/// comes from the Kaiser formula, then is rounded up so that `N - 1` is a
/// multiple of `2 * m`, making the group delay an exact integer number of
/// output samples.
pub(crate) fn build_prototype(in_rate: u32, out_rate: u32, m: u32, design_db: f64) -> Prototype {
    let fs_op = f64::from(m) * f64::from(out_rate); // common rate M*fout = L*fin
    let control = f64::from(in_rate.min(out_rate)); // base of the controlling Nyquist
    let cutoff = 0.4875 * control; // transition midpoint

    // Kaiser parameters (A > 50 regime for the beta formula).
    let beta = 0.1102 * (design_db - 8.7);

    // Length from the Kaiser formula, rounded so (N - 1) is a multiple of 2 * M
    // (the group delay (N - 1) / (2 * M) is then an exact integer count of output
    // samples). exact_prototype_len is the one source of truth for this length.
    let n = exact_prototype_len(in_rate, out_rate, m, design_db);
    let two_m = 2 * m as usize;

    let center = (n - 1) / 2; // integer (N is odd here)
    let wc = 2.0 * PI * cutoff / fs_op; // cutoff in radians/sample at fs_op
    let i0_beta = special::i0(beta);

    let mut coeffs = vec![0.0_f64; n];
    for (i, coeff) in coeffs.iter_mut().enumerate() {
        let offset = i as i64 - center as i64;
        // Ideal lowpass impulse response (sinc).
        let ideal = if offset == 0 {
            wc / PI
        } else {
            let off = offset as f64;
            special::sin(wc * off) / (PI * off)
        };
        // Kaiser window: I0(beta * sqrt(1 - t^2)) / I0(beta), t in [-1, 1].
        let t = (i as f64 - center as f64) / center as f64;
        let arg = beta * (1.0 - t * t).max(0.0).sqrt();
        let window = special::i0(arg) / i0_beta;
        *coeff = ideal * window;
    }

    // Normalize to unity DC gain, summed in a fixed order.
    let mut sum = 0.0_f64;
    for &c in &coeffs {
        sum += c;
    }
    let inv = 1.0 / sum;
    for c in &mut coeffs {
        *c *= inv;
    }

    Prototype {
        coeffs,
        num_taps: n,
        group_delay_out: (n - 1) / two_m,
    }
}

/// Builds an oversampled windowed-sinc prototype for the arbitrary-ratio path.
///
/// The same Kaiser-windowed lowpass as [`build_prototype`] (cutoff
/// `0.4875 * min(in, out)`, transition `0.025 * min(in, out)`, `design_db`
/// attenuation), but designed at the operating rate `phases * in_rate` so it can
/// be decomposed into `phases` sub-sample phases for fractional interpolation.
/// Returns the coefficients (unity DC gain, `f64`) and the length N. The general
/// engine interpolates between adjacent phases and reports its own group delay,
/// so no `(N - 1) / (2 * M)` alignment is applied here.
pub(crate) fn build_oversampled_prototype(
    in_rate: u32,
    out_rate: u32,
    phases: usize,
    design_db: f64,
) -> (Vec<f64>, usize) {
    let fs_op = phases as f64 * f64::from(in_rate);
    let control = f64::from(in_rate.min(out_rate));
    let cutoff = 0.4875 * control;

    let beta = 0.1102 * (design_db - 8.7);
    // Length from the Kaiser formula, rounded up to an odd length so the center
    // is a sample. general_prototype_len is the one source of truth.
    let n = general_prototype_len(in_rate, out_rate, phases, design_db);

    let center = (n - 1) / 2;
    let wc = 2.0 * PI * cutoff / fs_op;
    let i0_beta = special::i0(beta);

    let mut coeffs = vec![0.0_f64; n];
    for (i, coeff) in coeffs.iter_mut().enumerate() {
        let offset = i as i64 - center as i64;
        let ideal = if offset == 0 {
            wc / PI
        } else {
            let off = offset as f64;
            special::sin(wc * off) / (PI * off)
        };
        let t = (i as f64 - center as f64) / center as f64;
        let arg = beta * (1.0 - t * t).max(0.0).sqrt();
        let window = special::i0(arg) / i0_beta;
        *coeff = ideal * window;
    }

    let mut sum = 0.0_f64;
    for &c in &coeffs {
        sum += c;
    }
    let inv = 1.0 / sum;
    for c in &mut coeffs {
        *c *= inv;
    }

    (coeffs, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stopband attenuation the realized filter must clear, in dB.
    const TARGET_STOPBAND_DB: f64 = 80.0;

    // ---- Measurement tools (test-only) -------------------------------------
    //
    // These measure the realized frequency response of a built prototype. They
    // use the standard library's sin/cos (FFT twiddles and the direct response
    // sum): this is measurement code, not the coefficient path, so its
    // platform-dependent transcendental precision does not affect the shipped
    // filter.

    // In-place iterative radix-2 Cooley-Tukey FFT.
    fn fft(re: &mut [f64], im: &mut [f64]) {
        let n = re.len();
        assert!(n.is_power_of_two());

        // Bit-reversal permutation.
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        // Butterflies.
        let mut len = 2usize;
        while len <= n {
            let ang = -2.0 * PI / len as f64;
            let (wlen_re, wlen_im) = (ang.cos(), ang.sin());
            let mut base = 0usize;
            while base < n {
                let mut w_re = 1.0_f64;
                let mut w_im = 0.0_f64;
                let half = len / 2;
                for k in 0..half {
                    let a = base + k;
                    let b = base + k + half;
                    let v_re = re[b] * w_re - im[b] * w_im;
                    let v_im = re[b] * w_im + im[b] * w_re;
                    let u_re = re[a];
                    let u_im = im[a];
                    re[a] = u_re + v_re;
                    im[a] = u_im + v_im;
                    re[b] = u_re - v_re;
                    im[b] = u_im - v_im;
                    let nw_re = w_re * wlen_re - w_im * wlen_im;
                    let nw_im = w_re * wlen_im + w_im * wlen_re;
                    w_re = nw_re;
                    w_im = nw_im;
                }
                base += len;
            }
            len <<= 1;
        }
    }

    // Exact zero-phase magnitude of the symmetric FIR at a given frequency.
    fn response_mag(coeffs: &[f64], fs_op: f64, freq: f64) -> f64 {
        let center = (coeffs.len() - 1) / 2;
        let w = 2.0 * PI * freq / fs_op;
        let mut acc = coeffs[center];
        for k in 1..=center {
            acc += 2.0 * coeffs[center + k] * (w * k as f64).cos();
        }
        acc.abs()
    }

    struct Measurement {
        passband_ripple_db: f64,
        stopband_db: f64,
        stopband_freq: f64,
    }

    #[allow(clippy::needless_range_loop)]
    fn measure(coeffs: &[f64], fs_op: f64, passband_edge: f64, stopband_edge: f64) -> Measurement {
        let n = coeffs.len();

        // FFT size: power of two, at least 4*N, with resolution <= ~12 Hz.
        let mut nf = (4 * n).next_power_of_two();
        while fs_op / nf as f64 > 12.0 {
            nf <<= 1;
        }

        let mut re = vec![0.0_f64; nf];
        let mut im = vec![0.0_f64; nf];
        re[..n].copy_from_slice(coeffs);
        fft(&mut re, &mut im);

        let bin_hz = fs_op / nf as f64;
        let mag = |k: usize| (re[k] * re[k] + im[k] * im[k]).sqrt();
        let reference = mag(0); // unity DC gain after normalization

        // Passband ripple over [0, fp].
        let kp = (passband_edge / bin_hz).floor() as usize;
        let mut pmax = f64::MIN;
        let mut pmin = f64::MAX;
        for k in 0..=kp {
            let g = mag(k);
            if g > pmax {
                pmax = g;
            }
            if g < pmin {
                pmin = g;
            }
        }
        let passband_ripple_db = 20.0 * (pmax / pmin).log10();

        // Worst-case stopband over [fs, fs_op/2], scanned on FFT bins.
        let ks = (stopband_edge / bin_hz).ceil() as usize;
        let mut worst = 0.0_f64;
        let mut worst_k = ks;
        for k in ks..=(nf / 2) {
            let g = mag(k);
            if g > worst {
                worst = g;
                worst_k = k;
            }
        }

        // Refine the peak with the exact response around the worst bin, to
        // recover the true sidelobe peak that can fall between FFT bins.
        let mut peak = worst;
        let mut peak_freq = worst_k as f64 * bin_hz;
        let lo = (worst_k as f64 - 3.0) * bin_hz;
        let hi = (worst_k as f64 + 3.0) * bin_hz;
        let steps = 240usize;
        for s in 0..=steps {
            let f = lo + (hi - lo) * (s as f64) / (steps as f64);
            if f < stopband_edge || f > fs_op / 2.0 {
                continue;
            }
            let g = response_mag(coeffs, fs_op, f);
            if g > peak {
                peak = g;
                peak_freq = f;
            }
        }

        Measurement {
            passband_ripple_db,
            stopband_db: 20.0 * (peak / reference).log10(),
            stopband_freq: peak_freq,
        }
    }

    // Geometry the measurement references, mirroring build_prototype.
    fn geometry(in_rate: u32, out_rate: u32, m: u32) -> (f64, f64, f64) {
        let fs_op = f64::from(m) * f64::from(out_rate);
        let control = f64::from(in_rate.min(out_rate));
        (fs_op, 0.475 * control, 0.5 * control)
    }

    // ---- The measurement gate ----------------------------------------------

    #[test]
    fn fft_of_unit_impulse_is_flat() {
        let n = 64usize;
        let mut re = vec![0.0_f64; n];
        let mut im = vec![0.0_f64; n];
        re[0] = 1.0;
        fft(&mut re, &mut im);
        for (k, (&rk, &ik)) in re.iter().zip(&im).enumerate() {
            let m = (rk * rk + ik * ik).sqrt();
            assert!((m - 1.0).abs() < 1e-12, "bin {k} magnitude {m}");
        }
    }

    #[test]
    fn kernel_meets_spec_48k_to_16k() {
        // out/in = 16000/48000 = 1/3 -> L = 1, M = 3.
        let (in_rate, out_rate, m) = (48000u32, 16000u32, 3u32);
        let proto = build_prototype(in_rate, out_rate, m, DESIGN_STOPBAND_DB);
        let (fs_op, fp, fs) = geometry(in_rate, out_rate, m);

        assert_eq!((proto.num_taps - 1) % (2 * m as usize), 0);
        assert_eq!(
            proto.group_delay_out,
            (proto.num_taps - 1) / (2 * m as usize)
        );

        let r = measure(&proto.coeffs, fs_op, fp, fs);
        let beta = 0.1102 * (DESIGN_STOPBAND_DB - 8.7);
        println!(
            "48k->16k: N={}, group_delay_out={}, beta={:.6}, stopband={:.3} dB @ {:.1} Hz, passband_ripple={:.6} dB",
            proto.num_taps, proto.group_delay_out, beta, r.stopband_db, r.stopband_freq, r.passband_ripple_db
        );

        assert!(
            r.stopband_db <= -TARGET_STOPBAND_DB,
            "stopband {:.3} dB must clear -{} dB",
            r.stopband_db,
            TARGET_STOPBAND_DB
        );
        assert!(
            r.passband_ripple_db < 0.1,
            "passband ripple {:.6} dB exceeds 0.1 dB",
            r.passband_ripple_db
        );
    }

    #[test]
    fn kernel_meets_spec_44k_to_16k() {
        // out/in = 16000/44100 = 160/441 -> L = 160, M = 441.
        let (in_rate, out_rate, m) = (44100u32, 16000u32, 441u32);
        let proto = build_prototype(in_rate, out_rate, m, DESIGN_STOPBAND_DB);
        let (fs_op, fp, fs) = geometry(in_rate, out_rate, m);

        assert_eq!((proto.num_taps - 1) % (2 * m as usize), 0);
        assert_eq!(
            proto.group_delay_out,
            (proto.num_taps - 1) / (2 * m as usize)
        );

        let r = measure(&proto.coeffs, fs_op, fp, fs);
        let beta = 0.1102 * (DESIGN_STOPBAND_DB - 8.7);
        println!(
            "44.1k->16k: N={}, group_delay_out={}, beta={:.6}, stopband={:.3} dB @ {:.1} Hz, passband_ripple={:.6} dB",
            proto.num_taps, proto.group_delay_out, beta, r.stopband_db, r.stopband_freq, r.passband_ripple_db
        );

        assert!(
            r.stopband_db <= -TARGET_STOPBAND_DB,
            "stopband {:.3} dB must clear -{} dB",
            r.stopband_db,
            TARGET_STOPBAND_DB
        );
        assert!(
            r.passband_ripple_db < 0.1,
            "passband ripple {:.6} dB exceeds 0.1 dB",
            r.passband_ripple_db
        );
    }

    #[test]
    fn prototype_is_reproducible() {
        let a = build_prototype(48000, 16000, 3, DESIGN_STOPBAND_DB);
        let b = build_prototype(48000, 16000, 3, DESIGN_STOPBAND_DB);
        assert_eq!(a.coeffs.len(), b.coeffs.len());
        for (x, y) in a.coeffs.iter().zip(&b.coeffs) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}
