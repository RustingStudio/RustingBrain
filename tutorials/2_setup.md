# 2. Setup — From Nothing to a Running Model

**You need:** a computer. Windows, macOS, or Linux. Nothing else.

**Time:** 10 minutes, most of it waiting for a download.

There is no CUDA to install, no Python version to fight, no `pip` dependency
conflicts, no 500 MB of wheels. Two commands and you're training.

---

## 2.1 Install Rust

Rust comes with `rustup`, which installs the compiler and `cargo` (the build
tool and package manager) together.

### Linux / macOS

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Press `1` for the default install. Then either restart your terminal or run:

```bash
source "$HOME/.cargo/env"
```

Fish shell users: `source "$HOME/.cargo/env.fish"`.

### Windows

Download and run the installer from <https://rustup.rs>. If it asks for Visual
Studio C++ build tools, say yes — Rust needs a linker and that's where Windows
keeps it. Then open a **new** terminal so your `PATH` updates.

### Check it worked

```bash
cargo --version
```

You want something like `cargo 1.85.0` or newer. RustingBrain needs **1.85+**
because it uses the 2024 edition.

If you get "command not found", your terminal hasn't picked up the new `PATH`.
Close it, open a fresh one, and try again.

> Already have Rust but an old version? `rustup update` fixes it.

---

## 2.2 Make a project

```bash
cargo new ml-tutorial
cd ml-tutorial
```

This creates:

```
ml-tutorial/
├── Cargo.toml     ← project settings and dependency list
└── src/
    └── main.rs    ← your code
```

Add RustingBrain as a dependency:

```bash
cargo add rusting_brain
```

That edits `Cargo.toml` for you. Open it and you'll see:

```toml
[dependencies]
rusting_brain = "0.1"
```

That's the install. That was the whole install.

> **Working from a clone instead?** If you cloned the RustingBrain repository to
> read this file, you can run the built-in examples directly from inside it with
> `cargo run --example xor` — no new project needed. But making your own project
> is better for learning, because you'll be writing the code yourself.

---

## 2.3 Prove it works

Replace everything in `src/main.rs` with:

```rust
use rusting_brain::{Activation, Network, Optimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Network::builder()
        .input_size(2)
        .dense(4, Activation::Relu)
        .dense(1, Activation::Sigmoid)
        .optimizer(Optimizer::adam(0.01))
        .seed(42)
        .build();

    println!("Model built successfully.");
    println!("  inputs:  {}", model.input_size());
    println!("  outputs: {}", model.output_size());
    println!("  layers:  {}", model.layers().len());

    let prediction = model.predict(&[0.5, 0.5])?;
    println!("  untrained prediction: {:.4}", prediction[0]);

    Ok(())
}
```

Run it:

```bash
cargo run
```

The first run compiles the library too, so give it 20–60 seconds. Later runs are
instant. You should see:

```
Model built successfully.
  inputs:  2
  outputs: 1
  layers:  2
  untrained prediction: 0.5091
```

**If you see that, you are done. You have a working machine learning setup.**

The prediction is meaningless — the model is untrained, its parameters are still
random. That's the point: you just built the pile of numbers from chapter 1.
Chapter 3 makes them good.

> Your exact prediction number should match `0.5091` since we set `.seed(42)`,
> which fixes the random initialisation. If yours differs slightly, that's a
> floating-point difference across platforms and is harmless.

---

## 2.4 What that code actually said

You'll write this shape constantly, so let's name the parts:

```rust
Network::builder()                        // start describing a network
    .input_size(2)                        // it takes 2 numbers in
    .dense(4, Activation::Relu)           // hidden layer: 4 neurons, ReLU
    .dense(1, Activation::Sigmoid)        // output layer: 1 neuron, sigmoid
    .optimizer(Optimizer::adam(0.01))     // update rule + learning rate
    .seed(42)                             // reproducible random start
    .build();                             // make it
```

Each `.dense(n, act)` adds one layer of `n` neurons. **The last `.dense(...)`
you write is your output layer** — there's no separate method for it. So this
network has 2 layers of parameters: 2→4 and 4→1.

Parameter count, using the rule from chapter 1:

```
layer 1:  2 × 4 + 4  = 12
layer 2:  4 × 1 + 1  =  5
                       ──
total                  17 parameters
```

Seventeen numbers, currently random. Training will find better values for all
seventeen.

### About `?` and `Result`

`model.predict(...)` returns a `Result` because it can fail — most commonly when
you pass the wrong number of inputs. The `?` after it means "if this failed,
stop and return the error". That's why `main` is declared as returning
`Result<(), Box<dyn std::error::Error>>`.

You'll see this pattern in every chapter. It's Rust refusing to let you ignore
errors, and it's the reason a shape mistake gives you a clear message instead of
a mysterious crash six lines later.

---

## 2.5 Optional extras (skip these for now)

You do **not** need these to follow the course. Come back when a later chapter
tells you to.

**GPU training** (chapter 11) needs an NVIDIA GPU and the CUDA toolkit:

```bash
cargo add rusting_brain --features cuda
```

**Loading models from TensorFlow or PyTorch** (chapter 12) needs the ONNX
feature:

```bash
cargo add rusting_brain --features onnx
```

Both are off by default, which is why the basic install is so small.

---

## 2.6 If something went wrong

| Symptom | Fix |
|---------|-----|
| `cargo: command not found` | Open a new terminal. If it persists, re-run the rustup installer. |
| `error: package requires rustc 1.85` | `rustup update` |
| `linker 'cc' not found` (Linux) | `sudo apt install build-essential` (Debian/Ubuntu) or `sudo pacman -S base-devel` (Arch) |
| `link.exe not found` (Windows) | Install "Desktop development with C++" from the Visual Studio Installer. |
| Compile is very slow the first time | Normal. It's compiling dependencies once. Subsequent builds are cached. |
| `cargo add` doesn't exist | Old cargo. `rustup update`, or add `rusting_brain = "0.1"` to `Cargo.toml` by hand. |

For anything else, chapter 13 is a full troubleshooting reference.

---

## Recap

- Rust installs with one command and brings its own package manager.
- `cargo new` makes a project, `cargo add rusting_brain` adds the library.
- `cargo run` builds and runs.
- `Network::builder()` describes an architecture; the last `.dense()` is the
  output layer.
- You built a 17-parameter network. It predicts nonsense, because nobody has
  trained it yet.

---

**Next:** [3. The XOR Problem — your first real trained network](3_xor_problem.md)
