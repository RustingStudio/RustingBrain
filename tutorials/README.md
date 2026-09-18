# The RustingBrain Course

Sixteen chapters, from what a neural network is to training a mixture-of-experts
language model on a GPU. No prior machine-learning knowledge assumed. Rust
assumed at the level of "I can read a `for` loop".

Every chapter has runnable code in [`code/`](code/) and ends with exercises.
Chapters 1–10 need nothing but Rust. Chapter 11 and 14–16 are much faster with
an NVIDIA GPU, and all of them still run without one.

---

## Part I — Foundations

| # | Chapter | You need | Time |
|---|---|---|---|
| 1 | [Introduction — How a Neural Network Actually Works](1_introduction.md) | nothing | 30 min |
| 2 | [Setup — From Nothing to a Running Model](2_setup.md) | a computer | 10 min |
| 3 | [The XOR Problem — Your First Trained Network](3_xor_problem.md) | 1–2 | 30 min |
| 4 | [Working With Real Data](4_data.md) | 1–3 | 45 min |

Read 1 even if you are in a hurry. Everything after it assumes it.

## Part II — The Dense Toolkit

| # | Chapter | You need | Time |
|---|---|---|---|
| 5 | [Regression — Predicting Numbers](5_regression.md) | 1–4 | 45 min |
| 6 | [Classification — Choosing Between Categories](6_classification.md) | 1–5 | 45 min |
| 7 | [The Training Loop — Taking Control](7_training_loop.md) | 1–6 | 1 hr |
| 8 | [Evaluation — Measuring Honestly](8_evaluation.md) | 1–7 | 45 min |
| 9 | [Saving, Loading, and Using a Model](9_save_load.md) | 1–8 | 40 min |
| 10 | [A Complete Project](10_full_project.md) | 1–9 | 90 min |

Chapter 8 is the one people skip and regret. A model that looks good and is not
is worse than one that looks bad.

## Part III — Going Further

| # | Chapter | You need | Time |
|---|---|---|---|
| 11 | [Training on the GPU with CUDA](11_gpu_cuda.md) | 1–10, an NVIDIA GPU | 1 hr |
| 12 | [Importing Models From TensorFlow and PyTorch](12_import_models.md) | 1–10 | 45 min |
| 13 | [When Things Go Wrong](13_troubleshooting.md) | 1–10 | read once, return often |

## Part IV — Language Models

| # | Chapter | You need | Time |
|---|---|---|---|
| 14 | [Tokens and a Transformer Language Model](14_language_model.md) | 1–10 | 2 hr |
| 15 | [Mixture of Experts](15_mixture_of_experts.md) | 14 | 90 min |
| 16 | [Training a Language Model End to End](16_training_an_llm.md) | 14–15 | an afternoon |

---

## If you are here for one thing

| You want to | Start at |
|---|---|
| Understand what any of this means | [1](1_introduction.md) |
| Predict a number from a spreadsheet | [5](5_regression.md) |
| Sort things into categories | [6](6_classification.md) |
| Know whether your model is actually any good | [8](8_evaluation.md) |
| Ship a trained model | [9](9_save_load.md) |
| Make it run on your GPU | [11](11_gpu_cuda.md) |
| Run a model someone else trained | [12](12_import_models.md) |
| Fix a loss curve that looks wrong | [13](13_troubleshooting.md) |
| Train a language model | [14](14_language_model.md) |
| Fit a bigger model in the same memory | [15](15_mixture_of_experts.md) |
| Plan a run that takes days | [16](16_training_an_llm.md) |

## Running the code

Each chapter links its full source in [`code/`](code/). Set up a project once,
as chapter 2 describes, and paste a chapter's file into `src/main.rs`:

```bash
cargo run --release
```

`--release` is not optional for anything past chapter 10. A debug build is ten
to fifty times slower, and chapter 13.8 is a list of people who forgot.

## What is not here

Convolutions, encoder-decoder models, reinforcement learning, multi-GPU and
distributed training. None of them are implemented in the library, so none of
them are taught here.

The reference documentation is the source. Every public type carries the
reasoning behind it, which is the part that does not fit in a tutorial.
