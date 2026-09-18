# 11. Training on the GPU with CUDA

**You need:** chapters 1–10, an NVIDIA GPU, and the CUDA toolkit.

**Time:** 1 hour, plus however long the driver install takes.

**Full code:** [`code/11_gpu_cuda.rs`](code/11_gpu_cuda.rs)

Everything so far ran on your CPU, and for the models in chapters 3–10 that was
the right choice. A 200-parameter XOR network on a GPU is slower than on a CPU:
the work takes microseconds and the round trip to the card takes longer than
the work.

This chapter is about the point where that flips.

---

## 11.1 What a GPU actually is

A CPU core is a very clever unit that runs one instruction stream very fast. A
Ryzen 5 7600X has six of them.

A GPU is the opposite trade. An RTX 3060 has 3584 tiny cores that are only
useful when they all do *the same thing* to *different data* at the same time.
Ask one to run an `if` chain and it crawls. Ask all 3584 to each multiply one
pair of numbers out of a matrix and it finishes in one shot.

Neural networks are almost entirely the second thing. `weights · inputs` for a
512×1408 layer is 720,896 independent multiply-accumulates with no branches and
no dependencies between them. That is the shape a GPU was built for.

So the rule is:

> The GPU wins when each step has enough arithmetic to fill it. It loses when
> the step is small, because you still pay to ship the data across PCIe.

Concretely, on an RTX 3060:

| Model | CPU | GPU | Worth it? |
|---|---|---|---|
| XOR, 17 parameters | instant | instant + overhead | No |
| Chapter 10 churn model, ~2k parameters | instant | instant + overhead | No |
| 2M-parameter dense net, 4096 rows | seconds | seconds | Marginal |
| 55M-parameter transformer | ~90 min/epoch | ~2 min/epoch | Yes |

There is no threshold you can look up. Measure it. Section 11.7 shows how.

---

## 11.2 Installing

The `cuda` feature is off by default, which is why everything up to here built
in thirty seconds with no GPU in the machine.

```bash
cargo add rusting_brain --features cuda
```

You also need:

- An NVIDIA driver recent enough for your card.
- The CUDA toolkit, for `libcublas` and `libnvrtc`.

[INSTALL_CUDA.md](../INSTALL_CUDA.md) in the repository root has the per-platform
detail. Check it worked:

```bash
nvidia-smi
```

If that prints a table with your GPU and a driver version, the driver is fine.

RustingBrain compiles its kernels at startup with NVRTC rather than shipping
precompiled binaries, so you do not need `nvcc` on the machine that *runs* the
program — only the runtime libraries.

---

## 11.3 Asking the device what it can do

Before you commit an overnight run to a GPU, ask it whether it is actually
going to work:

```rust
use rusting_brain::{accelerator_doctor, TrainingBackend};

let backend = TrainingBackend::Cuda {
    device: 0,
    memory_budget_mib: 8192,
};

match accelerator_doctor(backend)? {
    Some(report) => {
        println!("{} ({}), {} MiB total, {:?} MiB free",
            report.name, report.revision,
            report.total_memory_mib, report.free_memory_mib);
        println!("cuBLAS:  {}", report.gemm_available);
        println!("kernels: {}", report.kernel_available);
        println!("alloc:   {}", report.allocation_test);
    }
    None => println!("CPU backend; nothing to probe"),
}
```

Typical output:

```
NVIDIA GeForce RTX 3060 (8.6), 12288 MiB total, Some(11030) MiB free
cuBLAS:  true
kernels: true
alloc:   true
```

`allocation_test` is the one that catches the interesting failures. It really
allocates the budget you asked for and frees it again. A machine with a browser
holding 6 GiB of VRAM will report `total_memory_mib: 12288` and then fail the
allocation test for a 8192 MiB budget, which is exactly the answer you want
*before* the run rather than forty minutes in.

---

## 11.4 The memory budget

`memory_budget_mib` is a promise you make to the library about how much VRAM it
may use, and it is checked before anything is allocated.

It is not the card's total. Your desktop, your compositor, your browser, and
the driver itself all live in the same 12 GiB. On a 12 GiB card, `8192` to
`9000` is a sensible application budget.

You can find out what a given network and batch size will cost before you go
near a device:

```rust
use rusting_brain::estimate_tensor_memory_mib;

let needed = estimate_tensor_memory_mib(&model, batch_size)?;
println!("{needed} MiB of tensors for batch {batch_size}");
```

The estimate counts, for each layer, the weights, their gradients, and both
Adam moments (four copies of every parameter), plus activations and deltas for
the batch. It is deliberately an upper bound: if the estimate fits your budget,
the run fits.

If it does not fit, the lever is `batch_size`. Parameters cost the same
whatever you do; activations scale with the batch.

---

## 11.5 Training a dense network on the GPU

The smallest change: swap `fit` for `fit_with_backend`.

```rust
use rusting_brain::{Activation, Dataset, Network, Optimizer, TrainConfig, TrainingBackend};

let mut model = Network::builder()
    .input_size(16)
    .dense(256, Activation::Relu)
    .dense(256, Activation::Relu)
    .dense(1, Activation::Linear)
    .optimizer(Optimizer::adam(0.001))
    .seed(42)
    .build();

let config = TrainConfig { epochs: 50, batch_size: 256, shuffle: true, seed: Some(7) };

model.fit_with_backend(
    &data,
    config,
    TrainingBackend::Cuda { device: 0, memory_budget_mib: 8192 },
)?;
```

Everything else in your program is unchanged. The trained weights live in the
same `Network`, `predict` works the same way, and `save_json` writes the same
file a CPU run would. A model trained on a GPU can be loaded and served on a
machine that has never had a driver installed.

### It will not silently fall back

If the driver is missing, cuBLAS fails, a kernel does not compile, the
allocation does not fit the budget, or a numerical check trips, this returns
`Err` — it does not quietly finish on the CPU.

That is a deliberate design choice, and it is worth understanding why. A silent
fallback turns "my GPU run took nine hours" into a mystery you only solve by
noticing the fan never spun up. Failing loudly costs you one error message;
falling back costs you a day.

```
Error: CUDA backend is unavailable: this crate was built without the `cuda` feature
```

means you forgot `--features cuda`. It is the most common one.

---

## 11.6 Keeping the device warm between epochs

`fit_with_backend` sets up a device context, compiles the kernels, uploads the
dataset, trains, and tears it all down. Calling it once per epoch — which is
what you do if you want to checkpoint and early-stop, as in chapter 7 — pays
that setup fifty times.

`TrainingSession` keeps it:

```rust
use rusting_brain::{TrainingSession, TrainingBackend};

let one_epoch = TrainConfig { epochs: 1, batch_size: 256, shuffle: true, seed: Some(7) };

let mut session = TrainingSession::new(
    &model, &data, one_epoch,
    TrainingBackend::Cuda { device: 0, memory_budget_mib: 8192 },
)?
.expect("Cpu backend has no session; use Network::fit");

let mut best = f32::INFINITY;
for epoch in 0..50 {
    let loss = session.train_epoch()?;
    println!("epoch {epoch}: loss {loss:.6}");

    if loss < best {
        best = loss;
        let checkpoint = session.checkpoint()?;   // pulls weights back to the host
        model.restore_cuda_checkpoint(checkpoint)?;
        model.save_json("best.json")?;
    }
}

println!("{:?}", session.stats());
```

Two things to notice.

**`train_epoch()` does not copy anything back.** The weights stay on the
device. That is the whole point — the loss comes back as one float and nothing
else crosses PCIe.

**`checkpoint()` is the copy.** Call it when you actually want a host-side
snapshot, not every epoch out of habit. On the chapter-7 early-stopping pattern
that means calling it only when the validation loss improved.

`stats()` reports what the session did:

```rust
AcceleratorStats {
    peak_allocated_bytes: 41_943_040,
    setup_time: 412ms,
    training_time: 3.1s,
    checkpoint_time: 8ms,
    epochs: 50,
    batches: 800,
    host_to_device_bytes: 4_194_304,
    device_to_host_bytes: 131_072,
    dataset_resident: true,
}
```

`dataset_resident: true` means the whole training set fitted in the budget and
lives on the card. If it is `false`, the session is staging batches across PCIe
every step and your throughput is bounded by the bus rather than the GPU —
raise the budget or shrink the dataset.

---

## 11.7 Measuring instead of assuming

Never take "GPUs are faster" on faith for *your* model. Time both:

```rust
use std::time::Instant;

let cpu_start = Instant::now();
let mut cpu_model = model.clone();
cpu_model.fit(&data, config)?;
let cpu_time = cpu_start.elapsed();

let gpu_start = Instant::now();
let mut gpu_model = model.clone();
gpu_model.fit_with_backend(&data, config, backend)?;
let gpu_time = gpu_start.elapsed();

println!("cpu {cpu_time:?}, gpu {gpu_time:?}, speedup {:.2}x",
    cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
```

Run it on your real dataset with your real architecture. A speedup below about
1.5× is not worth the extra dependency and the extra failure mode.

The repository ships a ready-made version:

```bash
cargo run --release --example cuda_benchmark --features cuda
```

Two things will skew this if you let them:

- **Release mode.** A debug build makes the CPU side ten to fifty times slower
  and the GPU side barely slower, so it flatters the GPU enormously. Always
  compare `--release` against `--release`.
- **The first run.** Kernel compilation and context creation cost a few hundred
  milliseconds once. On a 3-second benchmark that is noise you are measuring
  instead of the thing you wanted.

---

## 11.8 On a Mac

Apple Silicon has no CUDA. RustingBrain has a Metal backend for dense networks
that follows the same contract:

```bash
cargo add rusting_brain --features metal
```

```rust
TrainingBackend::Metal { device: 0, memory_budget_mib: 8192 }
```

`accelerator_doctor`, `TrainingSession`, `train_epoch`, `checkpoint`, and
`stats` all work unchanged — that is what `TrainingSession` being backend-
agnostic buys you. The `metal` feature is inert on Linux and Windows: it stays
a valid thing to ask for, and the backend reports `MetalFeatureDisabled`
instead of failing to build.

The transformer path in chapters 14–16 is CUDA-only.

---

## 11.9 If something went wrong

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA backend is unavailable: built without the cuda feature` | Missing feature flag | `cargo run --features cuda` |
| `CUDA backend error: driver` at startup | No driver, or too old | Check `nvidia-smi` runs |
| `cannot find -lcublas` at link time | Toolkit not installed or not on the library path | See [INSTALL_CUDA.md](../INSTALL_CUDA.md) |
| Allocation fails well under the card's total | Something else is holding VRAM | `nvidia-smi` to see who; close it or lower the budget |
| GPU is *slower* than CPU | Model too small, or batch too small | Raise `batch_size`; if it is still slower, stay on CPU |
| `dataset_resident: false` in stats | Dataset does not fit the budget | Raise the budget, or accept the PCIe cost |
| GPU run gives different numbers than CPU | Float addition is not associative and the two sum in different orders | Expected. Differences in the 6th decimal are fine; differences in the 2nd are a bug |

---

## Exercises

1. Run `accelerator_doctor` while a game or a browser with hardware
   acceleration is open. Watch `free_memory_mib` drop and the allocation test
   fail.
2. Find the crossover point for your machine: train the same architecture at
   `input_size` 16, 64, 256, and 1024 on both backends and plot the speedup.
3. Time `fit_with_backend` called 50 times against a `TrainingSession` running
   50 epochs. Compare `setup_time` in the stats to the difference.
4. Take the chapter 10 churn model and train it on the GPU. It will be slower.
   Work out from `stats()` where the time actually goes.

---

## Recap

- A GPU is thousands of slow cores that are only useful in lockstep. Matrix
  multiplication is exactly that shape; small models are not.
- `--features cuda` is opt-in; the crate builds and tests without a driver.
- `accelerator_doctor` probes a device *before* you commit a run to it, and
  `allocation_test` is the check that catches a busy card.
- `memory_budget_mib` is a promise, not the card's total. Leave room for the
  desktop.
- `fit_with_backend` is the one-line switch; `TrainingSession` keeps the device
  warm when you checkpoint per epoch.
- `train_epoch()` keeps everything on the device. `checkpoint()` is the copy —
  call it when you need it, not every epoch.
- Nothing falls back to the CPU silently. An error costs you a message; a
  silent fallback costs you a day.
- Measure in `--release`, on your real model, before believing any of this.

---

**Next:** [12. Importing Models From TensorFlow and PyTorch](12_import_models.md)
