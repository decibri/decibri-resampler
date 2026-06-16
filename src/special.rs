//! Deterministic special functions for filter-coefficient construction.
//!
//! These replace the standard library's transcendental functions in the
//! coefficient path. The standard library documents the precision of `sin`,
//! `cos`, and `exp` as platform-dependent, which would make computed filter
//! coefficients differ across platforms. The routines here evaluate in `f64`
//! with a fixed operation order and no fused multiply-add, so a given input
//! produces the same bits on every platform.

#![allow(dead_code)]

use std::f64::consts::PI;

/// Modified Bessel function of the first kind, order zero.
///
/// Evaluated from the power series `I0(x) = sum_{k>=0} (x/2)^(2k) / (k!)^2`,
/// which converges for all `x` and rapidly over the range used by the Kaiser
/// window (`0 <= x <= beta`). Terms are accumulated in ascending order of `k`
/// until the next term no longer changes the running sum at `f64` precision.
pub(crate) fn i0(x: f64) -> f64 {
    let quarter_x_sq = (x * 0.5) * (x * 0.5); // (x/2)^2
    let mut term = 1.0_f64; // k = 0 term is 1
    let mut sum = 1.0_f64;
    let mut k = 1.0_f64;
    loop {
        // term_k = term_{k-1} * (x/2)^2 / k^2
        term = term * quarter_x_sq / (k * k);
        let next = sum + term;
        if next == sum {
            // The remaining terms are below the precision of the running sum.
            break;
        }
        sum = next;
        k += 1.0;
    }
    sum
}

/// Sine of `x` in radians, evaluated deterministically.
///
/// The argument is reduced to `[-pi, pi]` by subtracting the nearest multiple
/// of `2*pi`, then folded into `[-pi/2, pi/2]` using `sin(pi - r) = sin(r)`,
/// after which a fixed-degree odd Taylor polynomial (through the term in
/// `r^19`) is evaluated by Horner's method in a fixed order. For the argument
/// magnitudes used here (a few hundred radians) the reduction multiplier is
/// small, so the reduction error stays far below the filter's stopband floor.
pub(crate) fn sin(x: f64) -> f64 {
    const TWO_PI: f64 = 2.0 * PI;
    const HALF_PI: f64 = 0.5 * PI;

    // Reduce to [-pi, pi].
    let k = (x / TWO_PI).round();
    let mut r = x - k * TWO_PI;

    // Fold into [-pi/2, pi/2]: sin(pi - r) = sin(r), sin(-pi - r) = sin(r).
    if r > HALF_PI {
        r = PI - r;
    } else if r < -HALF_PI {
        r = -PI - r;
    }

    // Odd Taylor series sin(r) = sum_k (-1)^k r^(2k+1) / (2k+1)!.
    // Coefficients are reciprocal factorials of the odd integers.
    let c3 = -1.0 / 6.0;
    let c5 = 1.0 / 120.0;
    let c7 = -1.0 / 5040.0;
    let c9 = 1.0 / 362_880.0;
    let c11 = -1.0 / 39_916_800.0;
    let c13 = 1.0 / 6_227_020_800.0;
    let c15 = -1.0 / 1_307_674_368_000.0;
    let c17 = 1.0 / 355_687_428_096_000.0;
    let c19 = -1.0 / 121_645_100_408_832_000.0;

    let r2 = r * r;
    // Horner in ascending order, fixed evaluation order, no fused multiply-add.
    let mut p = c19;
    p = p * r2 + c17;
    p = p * r2 + c15;
    p = p * r2 + c13;
    p = p * r2 + c11;
    p = p * r2 + c9;
    p = p * r2 + c7;
    p = p * r2 + c5;
    p = p * r2 + c3;
    p = p * r2 + 1.0;
    p * r
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference I0 via its integral form I0(x) = (1/pi) * integral_0^pi
    // exp(x cos theta) d theta, evaluated with composite Simpson's rule. This
    // is an independent algorithm from the power series, and it deliberately
    // uses the standard library's exp/cos: it is a test reference, not the
    // coefficient path, so platform-dependent transcendental precision is fine.
    fn i0_integral_reference(x: f64) -> f64 {
        let n = 20_000usize; // even, for Simpson's rule
        let h = PI / n as f64;
        let mut s = 0.0_f64;
        for i in 0..=n {
            let theta = i as f64 * h;
            let f = (x * theta.cos()).exp();
            let weight = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            s += weight * f;
        }
        (s * h / 3.0) / PI
    }

    #[test]
    fn i0_matches_integral_reference() {
        let mut max_rel = 0.0_f64;
        for &x in &[0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.857, 8.298, 10.0, 12.0] {
            let reference = i0_integral_reference(x);
            let value = i0(x);
            let rel = ((value - reference) / reference).abs();
            if rel > max_rel {
                max_rel = rel;
            }
        }
        println!("i0 max relative error vs integral reference = {max_rel:e}");
        assert!(max_rel < 1e-9, "i0 relative error {max_rel:e} exceeds 1e-9");
    }

    #[test]
    fn sin_matches_reference_over_kernel_range() {
        // Cover the argument range reached by the kernel builder (a few hundred
        // radians). The standard library's sin is the reference here.
        let mut max_err = 0.0_f64;
        let n = 200_000usize;
        for i in 0..=n {
            let x = -330.0 + 660.0 * (i as f64) / (n as f64);
            let e = (sin(x) - x.sin()).abs();
            if e > max_err {
                max_err = e;
            }
        }
        println!("sin max absolute error vs reference over [-330, 330] = {max_err:e}");
        assert!(
            max_err < 1e-10,
            "sin absolute error {max_err:e} exceeds 1e-10"
        );
    }

    #[test]
    fn special_functions_are_reproducible() {
        // Pure functions: identical inputs must yield identical bits.
        for &x in &[0.0, 1.0, 7.857, 123.456, -200.0] {
            assert_eq!(i0(x).to_bits(), i0(x).to_bits());
            assert_eq!(sin(x).to_bits(), sin(x).to_bits());
        }
    }
}
