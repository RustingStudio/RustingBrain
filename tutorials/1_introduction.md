# 1. Introduction - How a Neural Network Actually Works

**You need:** nothing. No Rust, no install, no maths degree. Just read.

**Time:** about 30 minutes.

This chapter has no code. If you have never understood what a neural network
_is_, this is the chapter that fixes that. Everything after this one is
practical, and everything after this one assumes you read this one.

---

## 1.1 What problem are we even solving?

Normal programming looks like this:

```
you write the rules  ->  computer applies them to data  ->  answers
```

You want to know if a number is even? You write `n % 2 == 0`. Done. You knew
the rule, so you typed the rule.

Now: write the rule that decides whether a photo contains a cat.

You can't. Not because you're a bad programmer — nobody can. There _is_ no
short rule. Whiskers are sometimes hidden, cats are sometimes black shapes in a
dark room, and a fox looks a lot like a cat from behind.

Machine learning flips the arrows:

```
you provide data + correct answers  ->  computer finds the rules  ->  rules
```

You show the computer 10,000 photos labelled "cat" and "not cat", and it works
out the rule by itself. The rule it finds is not readable Rust code. It is a
big pile of numbers. Those numbers are called **parameters**, and this whole
chapter is about what they are and how the computer finds good values for them.

---

## 1.2 The neuron: a weighted sum with an opinion

The smallest piece is a **neuron**. A neuron takes several numbers in and
produces one number out. That's all.

Say you want to predict whether you'll enjoy a movie. You have three inputs:

| Input | Meaning                   | Value for this movie |
| ----- | ------------------------- | -------------------- |
| `x1`  | IMDb rating / 10          | 0.8                  |
| `x2`  | Is it a comedy? (1 = yes) | 1.0                  |
| `x3`  | Length in hours / 3       | 0.5                  |

You don't care about all three equally. You _love_ comedies, you mildly care
about ratings, and you actively dislike long films. So you assign each input an
importance number. Those are the **weights**:

| Weight        | Value | Reading                |
| ------------- | ----- | ---------------------- |
| `w1` (rating) | 0.6   | mildly positive        |
| `w2` (comedy) | 1.5   | strongly positive      |
| `w3` (length) | -0.9  | negative — long is bad |

The neuron multiplies each input by its weight and adds them up:

```
0.8 × 0.6  +  1.0 × 1.5  +  0.5 × (-0.9)
=  0.48    +     1.5     +   (-0.45)
=  1.53
```

Then it adds one more number, the **bias**:

```
1.53 + b        where, say, b = -0.5
= 1.03
```

The bias is the neuron's baseline mood. A big positive bias means "I tend to
like things unless the inputs talk me out of it". A big negative bias means the
opposite. Without a bias, a neuron with all-zero inputs must output zero, which
is an arbitrary and often wrong restriction.

The full formula for one neuron:

```
z = (x1·w1 + x2·w2 + ... + xn·wn) + b
```

**Weights and biases together are the parameters.** They are the only things
that get changed by training. The inputs come from your data, the formula is
fixed forever — the weights and biases are the knobs.

> **Term: parameter.** A single number inside the model that training is allowed
> to change. If someone says "a 7-billion-parameter model", they mean it has
> 7,000,000,000 of these knobs. RustingBrain models in this course will have
> between 9 and a few millions.

---

## 1.3 Layers: neurons side by side, then stacked

One neuron gives you one output number. Usually you want more, so you put
several neurons **side by side**, all looking at the same inputs but each with
its own private set of weights and its own bias. That row of neurons is a
**layer**.

Here is a layer of 3 neurons reading 2 inputs:

```
            ┌──────────┐
  x1 ──┬───>│ neuron 1 │──▶ a1     w = [w11, w12], b = b1
       │    └──────────┘
       │    ┌──────────┐
       ├───>│ neuron 2 │──▶ a2     w = [w21, w22], b = b2
       │    └──────────┘
       │    ┌──────────┐
  x2 ──┴───>│ neuron 3 │──▶ a3     w = [w31, w32], b = b3
            └──────────┘
```

Every input connects to every neuron. That is what **dense** (or
"fully connected") means, and it is the only layer type a `Network` has — the
whole of chapters 1–13. Transformer language models are built from other layers
(attention, RMSNorm, SwiGLU, a mixture of experts); they are a separate model
type, `TransformerLm`, and they start in chapter 14. There are no convolutions
anywhere in the crate.

Counting parameters is easy: each of the 3 neurons has 2 weights and 1 bias, so
`3 × 2 = 6` weights plus `3` biases = **9 parameters**.

> **The rule:** a dense layer with `n` inputs and `m` neurons has
> `n × m` weights and `m` biases, so `n × m + m` parameters.

Then you **stack** layers. The outputs of layer 1 become the inputs of layer 2:

```
 inputs        layer 1          layer 2        output
   (2)      (3 neurons)      (1 neuron)         (1)

   x1  ─┬──>  ●  ─┐
        │──>  ●  ─┼────────▶     ●      ────▶  prediction
   x2  ─┴──>  ●  ─┘
```

- The first thing on the left is the **input layer**. It isn't really a layer —
  it's just your data. It has no weights.
- Layers in the middle are **hidden layers**. "Hidden" only means you never look
  at their outputs directly.
- The last layer is the **output layer**. Its size is decided by your problem:
  1 neuron to predict one number, 3 neurons to choose between 3 categories.

That whole picture is the **architecture**. When you design a model, choosing
the architecture means choosing how many hidden layers and how many neurons in
each.

---

## 1.4 Activations: why a stack of layers isn't automatically useless

Here is a problem that ruins everything if you don't fix it.

A neuron computes a weighted sum. A weighted sum of weighted sums is... still
just a weighted sum. Stack a hundred layers of pure weighted sums and you can
prove, algebraically, that the whole hundred-layer network is exactly equal to
one single layer. All that depth buys you literally nothing.

The fix is to bend the output of each neuron through a non-straight function
before passing it on. That function is the **activation function**:

```
z = w·x + b           (the weighted sum, called the "pre-activation")
a = activation(z)     (the neuron's actual output)
```

The bend is what makes stacking worthwhile. With bends between them, layers can
compose into genuinely complicated shapes; without them, they collapse. This is
not a minor detail — it is the entire reason deep networks work.

RustingBrain gives you five:

### `Relu` — `max(0, z)`

```
      a │      /
        │     /
        │    /
    ────┼──────────  z
        │
   flat │    slope 1
```

Negative in, zero out. Positive in, unchanged out. It is stupidly simple, very
fast, and it is the default choice for hidden layers. Use this unless you have a
reason not to.

_Watch out:_ a ReLU neuron that lands deep in the negative zone outputs 0 and
has a derivative of 0, so it stops learning forever. That's a "dead neuron". If
a lot of them die, your model quietly gets dumber. Lowering the learning rate is
the usual cure.

### `Sigmoid` — squashes everything into `(0, 1)`

```
      1 ┤        ╭──────
        │      ╭─╯
    0.5 ┤ ────╭╯
        │   ╭─╯
      0 ┤───╯
        └──────────────  z
```

Any input, no matter how large, comes out between 0 and 1. That makes it
perfect for the **output** neuron when you want a probability: "70% chance this
email is spam". Use it for yes/no questions.

_Watch out:_ far from zero the curve is nearly flat, so its derivative is nearly
zero, so learning nearly stops. Don't use it in hidden layers of deep networks.

### `Tanh` — squashes into `(-1, 1)`

Same S-shape as sigmoid but centred on zero. Because its outputs average out to
around 0 rather than 0.5, it usually trains a bit better than sigmoid in hidden
layers. It's a solid choice for small networks — we'll use it in chapter 3.

### `Softmax` — turns a row of numbers into probabilities that add to 1

The only activation that looks at all the neurons in the layer at once. Given
raw scores `[2.0, 1.0, 0.1]` it produces something like `[0.66, 0.24, 0.10]` —
all positive, summing to exactly 1.

Use it for the output layer when you're picking **one** answer out of several
("this is a cat, a dog, or a bird"). Never use it in a hidden layer.

### `Linear` — `a = z`, no bend at all

Does nothing. That is the point: use it on the **output** layer when you're
predicting a real number that could be anything — a house price, a temperature,
a stock value. Sigmoid would trap your prediction between 0 and 1; linear lets
it be 4.7 or -230.

### The cheat sheet you'll actually use

| Where                      | Use       | Why                                    |
| -------------------------- | --------- | -------------------------------------- |
| Hidden layers              | `Relu`    | fast, works, default                   |
| Hidden layers (small nets) | `Tanh`    | smoother, often better on tiny models  |
| Output — predict a number  | `Linear`  | output must be unbounded               |
| Output — yes/no            | `Sigmoid` | output is a probability                |
| Output — pick 1 of N       | `Softmax` | outputs are probabilities summing to 1 |

---

## 1.5 The forward pass: data flowing to a prediction

Pushing an input through every layer to get a prediction is the **forward
pass**. Let's do one entirely by hand, with real numbers, on the smallest
network that still has all the parts.

**The network:** one input, one hidden neuron (ReLU), one output neuron
(Linear). Four parameters total.

```
  x ──[w1]──▶ (hidden, ReLU) ──[w2]──▶ (output, Linear) ──▶ prediction
                 +b1                        +b2
```

**Current parameter values** (they started as random numbers):

```
w1 = 0.5      b1 = 0.1
w2 = -0.3     b2 = 0.2
```

**The input:** `x = 2.0`. **The correct answer:** `t = 1.0`.

### Step 1 — hidden layer

```
z1 = w1·x + b1  =  0.5 × 2.0 + 0.1  =  1.1
a1 = Relu(1.1)  =  1.1                      (positive, so unchanged)
```

### Step 2 — output layer

```
z2 = w2·a1 + b2  =  -0.3 × 1.1 + 0.2  =  -0.33 + 0.2  =  -0.13
a2 = Linear(-0.13) = -0.13
```

**The network predicts -0.13. The right answer was 1.0.** That is terrible, and
that's expected — the parameters are random. Now we make them less random.

---

## 1.6 The loss function: putting a number on "how wrong"

You can't improve what you can't measure. A **loss function** takes the
prediction and the correct answer and returns one number: how badly you did.
Low is good. Zero is perfect.

Continuing our example with **mean squared error**, `loss = (prediction − target)²`:

```
loss = (-0.13 - 1.0)²  =  (-1.13)²  =  1.2769
```

Remember `1.2769`. We're going to make it smaller.

RustingBrain has three losses, and picking the right one is mostly mechanical —
it's decided by the same thing that decides your output activation:

| Loss                 | Use for             | Pair with output activation |
| -------------------- | ------------------- | --------------------------- |
| `Mse`                | predicting a number | `Linear`                    |
| `BinaryCrossEntropy` | yes/no              | `Sigmoid`                   |
| `CrossEntropy`       | pick 1 of N         | `Softmax`                   |

Why not just use `Mse` for everything? Because for probabilities it barely
punishes confident wrong answers. If the truth is "yes" and you say 0.01, MSE
charges you `0.98`; binary cross entropy charges you `4.6`. The cross entropy
losses scream when the model is confidently wrong, which is exactly the feedback
you want.

> Squaring in MSE serves two purposes: being wrong by -3 is as bad as being wrong
> by +3 (the sign shouldn't matter), and being wrong by 10 is _much_ worse than
> being wrong by 1, not just ten times worse.

---

## 1.7 Gradient descent: which way is downhill?

We have a number that says how wrong we are. We want it smaller. We can change
the parameters. So: **for each parameter, should I increase it or decrease it?**

Imagine standing on a hill in thick fog, trying to reach the valley. You can't
see the valley. But you can feel the ground under your feet and tell which
direction slopes downward. So you take a small step that way, and repeat. You
will reach a low point.

That's the whole algorithm. The "slope under your feet" has a name: the
**gradient**. For one parameter `w`, the gradient is written `∂loss/∂w` and it
answers exactly one question:

> _If I nudge `w` up by a tiny amount, how much does the loss change?_

- Gradient is **positive** → increasing `w` increases the loss → **decrease `w`**.
- Gradient is **negative** → increasing `w` decreases the loss → **increase `w`**.

Either way you move _against_ the gradient. That's the update rule:

```
new_w = old_w − learning_rate × gradient
```

The **learning rate** is your step size, and it's the single most important
setting you'll choose:

```
too small (0.00001)          just right (0.01)          too big (5.0)
        ╲                          ╲                        ╲    ╱╲    ╱
         ╲___                       ╲__                      ╲__╱  ╲__╱
   crawls forever              slides to the bottom      bounces, never lands
                                                          (or explodes to NaN)
```

Typical values are `0.001` to `0.1`. If your loss barely moves, raise it. If
your loss jumps around or becomes `NaN`, lower it.

---

## 1.8 Backpropagation: getting the gradient for every parameter

Gradient descent needs a gradient for **every single parameter**. A real network
has thousands. Computing each one separately would be hopelessly slow.

**Backpropagation** ("backprop", the "backstep") computes all of them in one
sweep, by walking _backwards_ through the network. The trick is that the network
is a chain of operations, and calculus has a rule for chains: to find how the
loss responds to an early parameter, multiply together the sensitivities of
every step between it and the loss. That's the **chain rule**.

The practical version:

> Start at the output with "how wrong were we". Hand that blame backwards
> through the network. At each layer, use the blame arriving from the right to
> compute that layer's gradients, then pass a share of the blame further left —
> each neuron receiving blame in proportion to how much it contributed.

Forward pass = data flows left to right. Backward pass = blame flows right to
left. Let's do it on our example.

### The state we're in

```
x  = 2.0      t   = 1.0
w1 = 0.5      b1  = 0.1      z1 = 1.1    a1 = 1.1
w2 = -0.3     b2  = 0.2      z2 = -0.13  a2 = -0.13
loss = 1.2769
```

### Step 1 — blame at the output

How does the loss change if the prediction `a2` changes? The loss is
`(a2 − t)²`, whose derivative is `2(a2 − t)`:

```
∂loss/∂a2 = 2 × (-0.13 − 1.0) = 2 × (-1.13) = -2.26
```

Negative, which reads as: _"increase the prediction and the loss will drop."_
Correct — we predicted -0.13 and wanted 1.0.

Now push that through the output activation. `Linear`'s derivative is 1, so
nothing changes:

```
δ2 = ∂loss/∂z2 = -2.26 × 1 = -2.26
```

`δ2` ("delta 2") is the blame assigned to the output neuron.

### Step 2 — gradients for the output layer's parameters

`z2 = w2·a1 + b2`. Nudge `w2` by a tiny amount and `z2` moves by `a1` times
that amount. So:

```
∂loss/∂w2 = δ2 × a1 = -2.26 × 1.1 = -2.486
∂loss/∂b2 = δ2 × 1  = -2.26
```

Note what this means: **a weight's gradient is its blame times its input.** A
weight fed by a large input gets a large gradient — it had more influence, so it
takes more of the blame. A weight fed by a zero input gets zero gradient — it
did nothing, so it isn't blamed at all.

### Step 3 — send the blame backwards

How much of this is the hidden neuron's fault? It contributed through `w2`:

```
∂loss/∂a1 = δ2 × w2 = -2.26 × (-0.3) = 0.678
```

Then through the hidden neuron's own activation. `z1 = 1.1` is positive, so
ReLU's derivative there is 1:

```
δ1 = 0.678 × 1 = 0.678
```

This is the step people mean by "backprop": blame moving from layer 2 into
layer 1, scaled by the connection weight and by the activation's slope.

> This is also where **vanishing gradients** come from. Every layer multiplies
> the blame by an activation derivative. Sigmoid's derivative maxes out at 0.25,
> so ten sigmoid layers can shrink the blame by 0.25¹⁰ ≈ 0.000001 — the early
> layers receive essentially nothing and never learn. ReLU's derivative is
> exactly 1 for positive inputs, which is why it took over.

### Step 4 — gradients for the hidden layer's parameters

Same rule as step 2, using the hidden neuron's input `x = 2.0`:

```
∂loss/∂w1 = δ1 × x = 0.678 × 2.0 = 1.356
∂loss/∂b1 = δ1 × 1 = 0.678
```

### Step 5 — update everything

All four gradients, in one backward sweep. Now apply `w ← w − lr × gradient`
with a learning rate of `0.1`:

```
w1: 0.5  − 0.1 × ( 1.356) = 0.5  − 0.1356 =  0.3644
b1: 0.1  − 0.1 × ( 0.678) = 0.1  − 0.0678 =  0.0322
w2: -0.3 − 0.1 × (-2.486) = -0.3 + 0.2486 = -0.0514
b2: 0.2  − 0.1 × (-2.26)  = 0.2  + 0.226  =  0.4260
```

### Did it work?

Run the forward pass again with the new parameters:

```
z1 = 0.3644 × 2.0 + 0.0322 = 0.7610       a1 = 0.7610
z2 = -0.0514 × 0.7610 + 0.4260 = 0.3869   a2 = 0.3869

loss = (0.3869 − 1.0)² = 0.3759
```

**The loss fell from 1.2769 to 0.3759 in a single step.** The prediction moved
from -0.13 toward the target of 1.0.

That is the entire mechanism. Everything else in machine learning is this loop,
run millions of times over thousands of examples:

```
   ┌──────────────────────────────────────────────┐
   │                                              │
   ▼                                              │
forward pass ──▶ loss ──▶ backward pass ──▶ update parameters
(predict)      (measure)   (blame)          (improve)
```

> **A note on RustingBrain's internals, so the source doesn't confuse you.**
> The library tracks `target − prediction` instead of `prediction − target`,
> and then _adds_ `learning_rate × gradient` rather than subtracting. Two sign
> flips cancel, so it lands in the same place — it's the same descent written
> with the signs pushed around. It also folds MSE's constant factor of 2 into
> the learning rate, which is standard practice and changes nothing except the
> numeric meaning of "0.1".

---

## 1.9 Epochs, batches, and how training is organised

You don't have one example, you have thousands. Three words describe how they're
fed in.

**Batch.** Rather than updating after every single example (noisy, slow), you
run a small group forward, average their gradients, and update once. That group
is a batch. Batch size 32 is a safe default; small datasets can use a batch as
large as the whole dataset.

**Epoch.** One complete pass through your entire dataset. If you have 1,000
examples and a batch size of 100, one epoch is 10 updates. Training for 50
epochs means the model sees every example 50 times.

**Shuffling.** Reorder the data between epochs. If your file happens to be
sorted by category, unshuffled batches would each contain only one category and
the model would lurch back and forth. Always shuffle.

Putting them together, this is what "training" means:

```
for each epoch:
    shuffle the dataset
    for each batch in the dataset:
        forward pass on the batch
        compute the average loss
        backward pass to get gradients
        update every parameter
    record this epoch's average loss
```

That loop is exactly what `model.fit(...)` does in chapter 3.

---

## 1.10 Two problems you will definitely hit

### Overfitting: memorising instead of learning

Give a big model a small dataset and it will simply memorise the answers. Loss
on the training data goes to almost zero — and the model is useless on anything
it hasn't seen, because it learned the _examples_ instead of the _pattern_.

It's the difference between a student who understands algebra and one who
memorised the answers to last year's exam. Both score 100% on last year's exam.
Only one passes this year's.

This is why you **never** judge a model on the data it trained on. You hold some
data back:

```
   all your data
   ├── training set    (~70%)  the model learns from this
   ├── validation set  (~15%)  you check on this while tuning
   └── test set        (~15%)  touched once, at the very end
```

The tell-tale sign, which you'll see for real in chapter 7:

```
loss │
     │╲                          validation loss turns back up
     │ ╲___                     ↗
     │     ╲___________╱────────      ← overfitting starts here
     │  ╲___
     │      ╲______________________   ← training loss keeps falling
     └──────────────────────────────  epochs
```

Training loss falling while validation loss rises = you are overfitting. Stop
training, or use a smaller model, or get more data.

### Underfitting: too simple to learn the pattern

The opposite. The model is too small, or trained for too few epochs, or the
learning rate is too low. Both training _and_ validation loss stay high. Fix it
with a bigger network, more epochs, or a higher learning rate.

Good training lives between the two.

---

## 1.11 Glossary

| Term                    | Meaning                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| **Parameter**           | A number inside the model that training changes. Weights and biases. |
| **Weight**              | How strongly one input affects one neuron.                           |
| **Bias**                | A neuron's baseline offset, added after the weighted sum.            |
| **Neuron**              | Computes `activation(weighted sum of inputs + bias)`.                |
| **Layer**               | A row of neurons all reading the same inputs.                        |
| **Dense layer**         | A layer where every input connects to every neuron.                  |
| **Architecture**        | How many layers, how many neurons each, which activations.           |
| **Activation function** | The bend applied to a neuron's output. Makes depth meaningful.       |
| **Forward pass**        | Running data through the network to get a prediction.                |
| **Loss function**       | Measures how wrong a prediction is. Lower is better.                 |
| **Gradient**            | How much the loss changes when a parameter changes slightly.         |
| **Backpropagation**     | Computing every gradient in one backward sweep.                      |
| **Gradient descent**    | Repeatedly nudging parameters against their gradients.               |
| **Learning rate**       | Step size for those nudges.                                          |
| **Optimizer**           | The rule that turns gradients into parameter updates (SGD, Adam).    |
| **Batch**               | A group of examples processed before one update.                     |
| **Epoch**               | One full pass over the training data.                                |
| **Overfitting**         | Memorising training data; fails on new data.                         |
| **Underfitting**        | Too simple to capture the pattern; fails everywhere.                 |
| **Inference**           | Using a trained model to make predictions. No learning happens.      |

---

## What you should now be able to say out loud

- A neural network is a stack of layers of neurons; a neuron is a weighted sum
  plus a bias, bent by an activation function.
- The weights and biases are the parameters — the only things training changes.
- The forward pass turns an input into a prediction.
- The loss function scores how wrong that prediction was.
- Backpropagation walks backwards and computes, for every parameter, which
  direction would reduce the loss.
- The optimizer nudges every parameter in that direction, a little.
- Repeat over batches and epochs, and the pile of numbers becomes a model.

If any of those still feels shaky, reread the section for it now. The rest of
this course is about _doing_ this, and doing goes much better when you know
what's happening underneath.

---

**Next:** [2. Setup — Getting Rust and RustingBrain running](2_setup.md)
