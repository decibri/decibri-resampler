<!-- markdownlint-disable MD033 MD041 MD059 -->
# decibri-resampler

A streaming, anti-aliased sample-rate resampler for mono `f32` audio.

<a href="https://crates.io/crates/decibri-resampler"><img src="https://img.shields.io/crates/v/decibri-resampler" alt="crates.io version"></a>&nbsp;
<a href="https://github.com/decibri/decibri-resampler/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>&nbsp;

## Properties

- Streaming and stateful: processes successive chunks, carrying filter state across calls so chunk boundaries are invisible in the output.
- Anti-aliased polyphase windowed-sinc conversion: over 80 dB of stopband attenuation and under 0.001 dB of passband ripple, measured on the 48000 to 16000 and 44100 to 16000 conversions.
- Deterministic by construction: a fixed accumulation order, no fused multiply-add, and no platform-dependent transcendentals in the coefficient path.
- Reports its algorithmic latency in output samples (106 for 48000 to 16000 and 44100 to 16000).
- Identity bypass at equal input and output rates: byte-identical passthrough with zero added latency.
- No per-call heap allocation in steady state.
- Exact handling of common integer-Hz conversions, with an arbitrary-ratio path for other rates.
- Emits `tracing` diagnostics at construction, flush, and reset; attach a `tracing` subscriber to capture them.

## Usage

```rust
use decibri_resampler::{PolyphaseResampler, Resampler};

// Construct for an input and output rate in Hz. The rate pair is fixed for the
// lifetime of the instance.
let mut resampler = PolyphaseResampler::new(48_000, 16_000).unwrap();

// Feed successive chunks of any size; resampled output is appended to the buffer.
let mut output = Vec::new();
resampler.process(&[0.0_f32; 480], &mut output);
resampler.process(&[0.0_f32; 320], &mut output);

// Drain the filter tail at end of stream.
resampler.flush(&mut output);

assert!(!output.is_empty());
```

## API

The crate's public surface is the `Resampler` trait, implemented by `PolyphaseResampler`. Input and output are a single mono `f32` channel; channel handling and sample-format conversion are out of scope.

- `PolyphaseResampler::new(in_rate, out_rate)` constructs a resampler for a fixed input and output rate pair, returning `ResamplerError` if either rate is zero.
- `process(&[f32], &mut Vec<f32>)` consumes an input chunk of any length and appends the resampled output, which is variable in length and may be empty for a small input.
- `flush(&mut Vec<f32>)` drains the group-delay tail into the output buffer at end of stream.
- `latency_samples() -> usize` reports the constant algorithmic latency in output samples.
- `is_identity() -> bool` is true when the input and output rates are equal.
- `reset()` clears all streaming state, keeping the configured rates.

## License

Apache-2.0 © 2026 [Decibri](https://decibri.com).
