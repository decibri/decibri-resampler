use thiserror::Error;

/// Errors produced by the resampler.
///
/// This enum is `#[non_exhaustive]`: consumers pattern-matching on it must
/// include a `_ =>` catch-all arm so future variant additions are not
/// source-breaking.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ResamplerError {
    /// A construction sample rate was zero. Input and output rates must both
    /// be greater than zero.
    #[error("sample rate must be greater than 0")]
    ZeroSampleRate,

    /// Construction rejected a rate pair whose anti-aliasing filter would
    /// exceed the crate's maximum supported prototype length.
    ///
    /// The payload carries the offending rates; they are not formatted into the
    /// Display string, so the message text stays stable.
    #[error("the requested rate pair requires a filter larger than the supported maximum")]
    RatePairUnsupported { in_rate: u32, out_rate: u32 },
}
