"""Draws every figure in the tutorials, in a light and a dark variant.

    python3 tutorials/figures/make.py

Needs matplotlib and numpy. Output goes next to this file as SVG.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent
THEMES = {
    "light": dict(ink="#0b0b0b", soft="#52514e", grid="#d9d8d3", fill="#eeedea",
                  blue="#2a78d6", orange="#eb6834"),
    "dark": dict(ink="#f0efec", soft="#c3c2b7", grid="#45443f", fill="#2a2a28",
                 blue="#3987e5", orange="#d95926"),
}
T = THEMES["light"]


def canvas(width, height, rows=1, cols=1, **kwargs):
    figure, axes = plt.subplots(rows, cols, figsize=(width, height), **kwargs)
    return figure, axes


def plot_axes(ax, xlabel="", ylabel=""):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["grid"])
    ax.tick_params(colors=T["soft"], length=0, labelsize=10)
    ax.set_xlabel(xlabel, color=T["soft"], fontsize=11)
    ax.set_ylabel(ylabel, color=T["soft"], fontsize=11)


def diagram_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def arrow(ax, start, end, color=None, bend=0.0, width=1.4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=width,
        color=color or T["soft"], connectionstyle=f"arc3,rad={bend}",
        shrinkA=0, shrinkB=0))


def box(ax, x, y, text, width=2.2, height=0.7, color=None, text_color=None, size=11):
    """A rounded box centred on (x, y)."""
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0,rounding_size=0.12", linewidth=1.4,
        edgecolor=color or T["blue"], facecolor=T["fill"]))
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=text_color or T["ink"])


def node(ax, x, y, radius=0.28, filled=True, color=None):
    color = color or T["blue"]
    ax.add_patch(Circle((x, y), radius, linewidth=1.6, edgecolor=color,
                        facecolor=color if filled else "none"))


def label(ax, x, y, text, size=11, color=None, **kwargs):
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    ax.text(x, y, text, fontsize=size, color=color or T["ink"], **kwargs)


# --- chapter 1 -------------------------------------------------------------

def layer():
    figure, ax = canvas(7.2, 3.2)
    diagram_axes(ax, (-0.6, 9.6), (-0.6, 3.6))
    inputs = [(0, 2.25, "x1"), (0, 0.75, "x2")]
    for index, y in enumerate((3, 1.5, 0), start=1):
        for x, input_y, _ in inputs:
            arrow(ax, (x + 0.35, input_y), (2.45, y))
        box(ax, 3.4, y, f"neuron {index}", width=1.9)
        arrow(ax, (4.35, y), (5.3, y))
        label(ax, 5.6, y, f"a{index}")
        label(ax, 6.3, y, f"w = [w{index}1, w{index}2],  b = b{index}",
              color=T["soft"], size=10, ha="left")
    for x, y, name in inputs:
        label(ax, x, y, name)
    return figure


def network():
    figure, ax = canvas(7.2, 3.0)
    diagram_axes(ax, (-0.8, 9.4), (-0.7, 3.6))
    hidden = [(3, 2.4), (3, 1.2), (3, 0)]
    inputs = [(0, 1.8), (0, 0.6)]
    out = (6, 1.2)
    for start in inputs:
        for end in hidden:
            arrow(ax, (start[0] + 0.3, start[1]), (end[0] - 0.3, end[1]))
    for start in hidden:
        arrow(ax, (start[0] + 0.3, start[1]), (out[0] - 0.3, out[1]))
    arrow(ax, (out[0] + 0.3, out[1]), (7.3, out[1]))
    for (x, y), name in zip(inputs, ("x1", "x2")):
        node(ax, x, y, filled=False, color=T["soft"])
        label(ax, x, y, name, size=10)
    for x, y in hidden + [out]:
        node(ax, x, y)
    label(ax, 7.45, 1.2, "prediction", ha="left")
    for x, title, note in ((0, "inputs", "2"), (3, "layer 1", "3 neurons"),
                           (6, "layer 2", "1 neuron")):
        label(ax, x, 3.35, title)
        label(ax, x, 2.98, note, size=9.5, color=T["soft"])
    return figure


def activation(function, ticks):
    def draw():
        figure, ax = canvas(4.6, 2.9)
        z = np.linspace(-5, 5, 400)
        ax.axhline(0, color=T["grid"], linewidth=1)
        ax.axvline(0, color=T["grid"], linewidth=1)
        ax.plot(z, function(z), color=T["blue"], linewidth=2)
        plot_axes(ax, "z", "a")
        ax.set_yticks(ticks)
        ax.set_xticks([-4, -2, 0, 2, 4])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        return figure
    return draw


def learning_rate():
    figure, axes = canvas(8.4, 2.9, 1, 3, sharey=True)
    cases = (("too small", 0.02, -1.5, 9, "crawls forever"),
             ("just right", 0.3, -1.5, 9, "slides to the bottom"),
             ("too big", 1.06, -0.75, 7, "bounces, never lands"))
    curve = np.linspace(-1.8, 1.8, 200)
    for ax, (title, rate, position, steps, note) in zip(axes, cases):
        ax.plot(curve, curve ** 2, color=T["soft"], linewidth=1.5)
        path = [position]
        for _ in range(steps):
            path.append(path[-1] - rate * 2 * path[-1])
        path = np.array(path)
        ax.plot(path, path ** 2, color=T["orange"], linewidth=1.4,
                marker="o", markersize=5)
        ax.set_title(title, color=T["ink"], fontsize=12)
        ax.set_xlabel(note, color=T["soft"], fontsize=10.5)
        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-0.15, 3.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("loss", color=T["soft"], fontsize=11)
    return figure


def training_loop():
    figure, ax = canvas(8.6, 2.4)
    diagram_axes(ax, (-1.6, 14.0), (-1.0, 2.4))
    steps = (("forward pass", "predict"), ("loss", "measure"),
             ("backward pass", "blame"), ("update parameters", "improve"))
    centres = (0, 3.3, 6.8, 11.2)
    widths = (2.7, 1.5, 3.0, 3.9)
    for x, width, (name, note) in zip(centres, widths, steps):
        box(ax, x, 0, name, width=width)
        label(ax, x, -0.7, note, size=10, color=T["soft"])
    for index in range(3):
        arrow(ax, (centres[index] + widths[index] / 2 + 0.08, 0),
              (centres[index + 1] - widths[index + 1] / 2 - 0.08, 0))
    ax.plot([11.2, 11.2, 0], [0.4, 1.7, 1.7], color=T["soft"], linewidth=1.4)
    arrow(ax, (0, 1.7), (0, 0.42))
    label(ax, 5.6, 2.0, "repeat", size=10, color=T["soft"])
    return figure


def overfitting():
    figure, ax = canvas(6.4, 3.3)
    epochs = np.linspace(0, 100, 300)
    train = 0.08 + 0.92 * np.exp(-epochs / 18)
    validation = 0.3 + 0.75 * np.exp(-epochs / 14) + 0.0000165 * epochs ** 2.2
    turn = epochs[np.argmin(validation)]
    ax.axvline(turn, color=T["grid"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.plot(epochs, train, color=T["blue"], linewidth=2)
    ax.plot(epochs, validation, color=T["orange"], linewidth=2)
    label(ax, 101, validation[-1], "validation", ha="left", color=T["ink"], size=10.5)
    label(ax, 101, train[-1], "training", ha="left", color=T["ink"], size=10.5)
    label(ax, turn + 2, 1.0, "overfitting starts here", ha="left",
          color=T["soft"], size=10)
    plot_axes(ax, "epochs", "loss")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.1)
    return figure


# --- chapter 3 -------------------------------------------------------------

def xor():
    figure, ax = canvas(3.9, 3.1)
    ax.scatter([0, 1], [1, 0], s=190, color=T["blue"], label="output 1", zorder=3)
    ax.scatter([0, 1], [0, 1], s=190, facecolors="none", edgecolors=T["orange"],
               linewidths=2, label="output 0", zorder=3)
    plot_axes(ax, "x1", "x2")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.4, 1.4)
    ax.set_aspect("equal")
    ax.grid(color=T["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=T["ink"], fontsize=10,
              loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=1.0)
    return figure


# --- chapter 13 ------------------------------------------------------------

def loss_shapes():
    figure, axes = canvas(8.6, 4.6, 2, 3)
    epochs = np.linspace(0, 1, 200)
    decay = 0.1 + 0.9 * np.exp(-epochs * 7)
    panels = axes.ravel()

    def panel(index, title):
        ax = panels[index]
        ax.set_title(title, color=T["ink"], fontsize=11.5)
        plot_axes(ax, "epoch", "loss")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.15)
        return ax

    panel(0, "healthy").plot(epochs, decay, color=T["blue"], linewidth=2)
    panel(1, "flat: not learning").plot(
        epochs, np.full_like(epochs, 0.8), color=T["blue"], linewidth=2)

    ax = panel(2, "NaN")
    cut = epochs < 0.55
    ax.plot(epochs[cut], decay[cut], color=T["blue"], linewidth=2)
    ax.plot(epochs[cut][-1], decay[cut][-1], marker="x", markersize=10,
            markeredgewidth=2.4, color=T["orange"])
    label(ax, 1.03, 0.34, "loss becomes NaN", ha="right", size=9.5, color=T["soft"])

    ax = panel(3, "overfitting")
    ax.plot(epochs, decay, color=T["blue"], linewidth=2)
    ax.plot(epochs, 0.3 + 0.75 * np.exp(-epochs * 9) + 0.5 * epochs ** 2.2,
            color=T["orange"], linewidth=2)
    label(ax, 0.98, 0.93, "validation", ha="right", size=9.5)
    label(ax, 0.98, 0.2, "training", ha="right", size=9.5)

    ax = panel(4, "unstable: rate too high")
    ax.plot(epochs, 0.55 + 0.3 * np.sin(epochs * 38) * (0.5 + epochs),
            color=T["blue"], linewidth=2)
    panels[5].axis("off")
    figure.subplots_adjust(hspace=0.55, wspace=0.3)
    return figure


# --- chapter 14 ------------------------------------------------------------

def transformer_block():
    figure, ax = canvas(4.6, 6.4)
    diagram_axes(ax, (-2.6, 3.6), (-0.5, 9.5))
    rows = ((9, "input"), (8, "RMSNorm"), (7, "attention\nRoPE + causal mask"),
            (5.8, "+"), (4.6, "RMSNorm"), (3.6, "SwiGLU"), (2.4, "+"), (1.2, "output"))
    half = {}
    for y, name in rows:
        if name == "+":
            node(ax, 0, y, radius=0.26, filled=False, color=T["soft"])
            label(ax, 0, y, "+", size=13)
            half[y] = 0.26
        elif name in ("input", "output"):
            label(ax, 0, y, name)
            half[y] = 0.25
        else:
            tall = "\n" in name
            box(ax, 0, y, name, width=3.4, height=0.95 if tall else 0.62, size=10.5)
            half[y] = 0.475 if tall else 0.31
    heights = [y for y, _ in rows]
    for upper, lower in zip(heights, heights[1:]):
        arrow(ax, (0, upper - half[upper] - 0.04), (0, lower + half[lower] + 0.04))
    for top, bottom in ((8.55, 5.8), (5.2, 2.4)):
        ax.plot([0, 2.5, 2.5], [top, top, bottom], color=T["orange"], linewidth=1.4)
        arrow(ax, (2.5, bottom), (0.3, bottom), color=T["orange"])
        label(ax, 2.65, (top + bottom) / 2, "residual", ha="left", size=10,
              color=T["soft"], rotation=90, rotation_mode="anchor", va="top")
    return figure


# --- chapter 15 ------------------------------------------------------------

def experts():
    figure, ax = canvas(8.2, 4.6)
    diagram_axes(ax, (-0.8, 13.2), (-1.0, 6.6))
    label(ax, 0, 6.2, "dense", ha="left", color=T["soft"], size=10)
    label(ax, 0.4, 5.5, "token")
    arrow(ax, (1.0, 5.5), (2.0, 5.5))
    box(ax, 5.6, 5.5, "one big FFN", width=7.0)
    arrow(ax, (9.2, 5.5), (10.2, 5.5))
    label(ax, 10.9, 5.5, "output")

    label(ax, 0, 4.3, "mixture of experts", ha="left", color=T["soft"], size=10)
    middle = 1.9
    label(ax, 0.4, middle, "token")
    arrow(ax, (1.0, middle), (1.7, middle))
    box(ax, 2.6, middle, "router", width=1.6)
    box(ax, 9.3, middle, "weighted sum", width=3.0)
    arrow(ax, (10.9, middle), (11.6, middle))
    label(ax, 12.3, middle, "output")
    for index in range(8):
        y = 3.71 - index * 0.517
        active = index + 1 in (3, 7)
        box(ax, 5.6, y, f"expert {index + 1}", width=1.9, height=0.42, size=9.5,
            color=T["blue"] if active else T["grid"],
            text_color=T["ink"] if active else T["soft"])
        if active:
            arrow(ax, (3.45, middle), (4.6, y), color=T["blue"])
            arrow(ax, (6.6, y), (7.75, middle), color=T["blue"])
    label(ax, 5.6, -0.7, "two experts run, six stay idle", color=T["soft"], size=10)
    return figure


def router():
    figure, ax = canvas(6.0, 3.0)
    scores = [0.02, 0.61, 0.05, 0.28, 0.01, 0.01, 0.01, 0.01]
    colors = [T["blue"] if score > 0.2 else T["grid"] for score in scores]
    ax.bar(range(8), scores, width=0.62, color=colors)
    for index, score in enumerate(scores):
        if score > 0.2:
            label(ax, index, score + 0.04, f"{score:.2f}", size=10.5)
    plot_axes(ax, "expert", "router probability")
    ax.set_xticks(range(8))
    ax.set_yticks([0, 0.25, 0.5])
    ax.set_ylim(0, 0.72)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color=T["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    label(ax, 7.4, 0.62, "top 2 run", ha="right", color=T["soft"], size=10)
    return figure


FIGURES = {
    "layer": layer,
    "network": network,
    "relu": activation(lambda z: np.maximum(0, z), [0, 2, 4]),
    "sigmoid": activation(lambda z: 1 / (1 + np.exp(-z)), [0, 0.5, 1]),
    "tanh": activation(np.tanh, [-1, 0, 1]),
    "learning_rate": learning_rate,
    "training_loop": training_loop,
    "overfitting": overfitting,
    "xor": xor,
    "loss_shapes": loss_shapes,
    "transformer_block": transformer_block,
    "experts": experts,
    "router": router,
}

if __name__ == "__main__":
    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none",
                         "svg.hashsalt": "rusting_brain"})
    for theme, colors in THEMES.items():
        T = colors
        for name, draw in FIGURES.items():
            figure = draw()
            figure.savefig(OUT / f"{name}_{theme}.svg", transparent=True,
                           bbox_inches="tight", pad_inches=0.12,
                           metadata={"Date": None})
            plt.close(figure)
