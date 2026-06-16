//! The streaming [`Resampler`] trait, this crate's public seam.

/// A streaming, stateful sample-rate resampler for mono `f32` audio.
///
/// An implementation converts a continuous stream of mono `f32` samples from a
/// fixed input rate to a fixed output rate. It is a long-lived object fed one
/// chunk after another for the life of a stream, not a one-shot whole-buffer
/// function: it carries internal state (anti-alias filter history and the
/// fractional sample position) between calls.
///
/// An implementation is constructed for a single `(input_rate, output_rate)`
/// pair plus any quality parameter. The rate pair is fixed for the lifetime of
/// the instance; a different rate pair requires a new instance. Construction
/// validates the rates and is the only fallible operation (it returns
/// [`ResamplerError`](crate::ResamplerError) when a rate is invalid); once an
/// instance exists, every method on this trait is infallible.
///
/// # The streaming carry
///
/// Successive [`process`](Resampler::process) calls carry filter history and the
/// fractional sample position across the chunk boundary, so the boundary is
/// invisible in the output. Feeding a stream as many small chunks and then
/// calling [`flush`](Resampler::flush) once produces the same output samples, in
/// the same order, as feeding the whole stream in a single call followed by a
/// single `flush`.
///
/// # Output length
///
/// Because resampling changes the sample count, the number of samples produced
/// per call varies. [`process`](Resampler::process) appends its output to the
/// caller-owned buffer and may append nothing when the input is too small to
/// produce a complete output sample, in which case the input is retained
/// internally and contributes to a later call. The number of samples a call
/// produces is the growth in the output buffer's length. Implementations append
/// to the buffer and do not clear or reallocate it, so a caller that reuses one
/// buffer across calls performs no per-call allocation in steady state.
///
/// # End of stream: flush once, then reset to reuse
///
/// At end of stream, call [`flush`](Resampler::flush) once to drain the samples
/// still held in the filter's group-delay tail and any partial-frame carry, so
/// the final audio is not dropped; those samples are appended to the caller's
/// buffer. After flushing, the instance must be [`reset`](Resampler::reset)
/// before it is fed more input. Calling [`process`](Resampler::process) after
/// `flush` without an intervening `reset` continues from the drained state and
/// is not the intended use.
///
/// # Identity bypass
///
/// When the input and output rates are equal,
/// [`is_identity`](Resampler::is_identity) returns `true` and the implementation
/// is a passthrough: [`process`](Resampler::process) appends the input unchanged
/// with no filtering, [`latency_samples`](Resampler::latency_samples) is `0`, and
/// [`flush`](Resampler::flush) appends nothing. The output is byte-identical to
/// the input.
///
/// # Input assumptions
///
/// The stream is a single mono channel of `f32`, nominally in `[-1.0, 1.0]`.
/// Multi-channel downmix and sample-format conversion are the caller's
/// responsibility and happen before [`process`](Resampler::process); this trait
/// does not deinterleave, downmix, or convert sample formats.
///
/// Implementations are [`Send`], so an instance can be handed between threads.
pub trait Resampler: Send {
    /// Resamples `input` and appends the produced samples to `output`.
    ///
    /// `input` is mono `f32` of any length, including empty. The number of
    /// samples produced varies and may be zero for a small input; it is the
    /// growth in `output.len()` across the call. State carries across calls so
    /// the chunk boundary is invisible (see the [trait](Resampler) contract).
    /// `output` is appended to, never cleared, and the caller owns it.
    ///
    /// Non-finite input samples (`NaN`, infinities) propagate through the filter
    /// to the output and are not detected or altered, so sanitizing input is the
    /// caller's responsibility.
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>);

    /// Drains the end-of-stream tail into `output`.
    ///
    /// Appends the samples still held in the filter's group-delay tail and any
    /// partial-frame carry, so the final audio is not dropped. Call once at end
    /// of stream; after it, call [`reset`](Resampler::reset) before feeding more
    /// input. For an identity bypass this appends nothing.
    fn flush(&mut self, output: &mut Vec<f32>);

    /// Returns the constant algorithmic latency, in output samples.
    ///
    /// This is the anti-alias filter's group delay expressed at the output rate.
    /// It is constant for the instance and is `0` for an identity bypass.
    /// Callers use it to account for the delay the resampler introduces.
    fn latency_samples(&self) -> usize;

    /// Returns `true` when the input and output rates are equal.
    ///
    /// In that case the resampler is a passthrough: the output equals the input
    /// and [`latency_samples`](Resampler::latency_samples) is `0`.
    fn is_identity(&self) -> bool;

    /// Clears all streaming state, returning the instance to its
    /// just-constructed condition.
    ///
    /// Filter history, the fractional sample position, and any retained input are
    /// cleared; the configured rates and filter geometry are kept and no
    /// reallocation occurs. Use it to restart a stream on the same instance.
    fn reset(&mut self);
}
