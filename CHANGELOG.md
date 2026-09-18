# Changelog

## 2.0.0

The first release aimed at production use. The breaking changes are removals of
API that had no implementation behind it.

### Breaking

- **Removed `tensor::Tensor`.** The trait had one implementor, `Matrix`, and
  was never used as a bound anywhere in the crate or its examples. Call the
  inherent methods on `Matrix` instead; every signature is unchanged.
- **Removed the `layers` module.** It re-exported `Dense` and `DenseLayer` from
  `network` and nothing else. Import them from `rusting_brain` directly.
- **Removed the `rusting_brain` binary.** `src/main.rs` duplicated
  `examples/xor.rs`. Run `cargo run --example xor` instead.
- **`gpu_matrix` is now private.** `GpuMatrix` and `gpu_dot` existed only for
  the matmul benchmark in `gpu_test`; the module documentation claimed the
  dense CUDA path used them, which was never true. The unused `to_cpu` and the
  unused `context` field are gone, and the remaining calls return `Result`
  instead of unwrapping a driver failure, which the rest of the device code
  has never done.

### Added

- Crate-level documentation with worked examples for training, KV-cached
  generation, and moving a model to a device.
- Module-level documentation on every source file.
- Tutorial chapters 11–16: CUDA training, importing ONNX models, a
  troubleshooting reference, transformer language models, mixture of experts,
  and an end-to-end training run. Chapters 14–16 are new material; the course
  previously stopped at dense networks.
- `tutorials/README.md`, an index over all sixteen chapters.
- `examples/language_model.rs`: trains a character-level transformer language
  model and generates from the checkpoint in a separate process. `--moe`
  switches the same model to sparse layers. This was the one model type the
  examples did not cover.
- An MSRV job in CI pinned to 1.85, so the `rust-version` in `Cargo.toml` is
  checked rather than asserted.
- `gemm_probe` and `flash_probe` declare `required-features = ["cuda"]`, so
  `cargo check --all-targets` no longer fails without the feature.

### Fixed

- Pinned `kstring` to 2.0.2. 2.0.4 requires rustc 1.96.0, which broke
  `--features onnx` against the declared MSRV of 1.85.
- Clippy is clean under `--all-targets` with no features, with `cuda`, and with
  `onnx`.

### Documentation

- `README.md` rewritten around measured throughput against PyTorch and
  TensorFlow on the same model and hardware, rather than unqualified claims.
- Planning documents moved under `docs/`.
- `Network::forward` documents that it panics on a wrong-length input, next to
  the `predict` that returns an error instead.
- `.cargo/config.toml` says what `target-cpu=native` does to a binary that is
  copied to another machine.
