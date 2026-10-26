"""The three headline figures -> imgs/*.png. Model-free: reads the landscape
and clone JSONs under runs/.

    uv run plot.py

  well.png      V(z) at T = 1 for base vs instruct, and the well bottom vs T
  survival.png  token-level -log S(t) on Qwen2.5-7B (straight = fixed hazard),
                and the fraction of completions ending above the midpoint vs T
  cloning.png   cloning vs brute force on gemma-2-2b-it, and the collapse of
                the population as k grows
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import jspace

MODELS = [  # (model id, label, categorical slot) — fixed order, never cycled
    ("google/gemma-2-2b", "gemma-2-2b (base)", "#2a78d6"),
    ("google/gemma-2-2b-it", "gemma-2-2b-it", "#eb6834"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "#1baf7a"),
]
TEMPS = [0.7, 1.0, 1.3]
RAMP = {0.7: "#86b6ef", 1.0: "#2a78d6", 1.3: "#104281"}  # ordinal, one hue
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OUT = Path("imgs")


def landscape(model: str, T: float) -> dict:
    return json.loads((jspace.model_dir(model) / "traj" / f"all-advice-diffmean-T{T}.landscape.json").read_text())


def z_of(r: dict, s) -> float:
    """Position in units of the class gap: 0 = safe mean, 1 = harmful mean."""
    return (s - r["mu_safe"]) / (r["mu_harm"] - r["mu_safe"])


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
        "axes.titlesize": 10, "axes.titlelocation": "left", "axes.titlecolor": INK2,
        "figure.titlesize": 12, "figure.titleweight": "bold",
    })


def figure(message: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    fig.suptitle(message, x=0.01, ha="left")
    return fig, axes


def well() -> None:
    fig, (a, b) = figure("Safety tuning digs the well below the safe-class mean; temperature lifts it back out")
    for model, label, c in MODELS:
        r = landscape(model, 1.0)
        edges = np.array(r["potential"]["edges"])
        z = z_of(r, (edges[:-1] + edges[1:]) / 2)
        V = np.array([np.nan if v is None else v for v in r["potential"]["pooled"]])
        ok = ~np.isnan(V)
        a.plot(z[ok], V[ok], color=c, label=label)
    a.set_ylim(-0.7, a.get_ylim()[1])
    for x, t in ((0, "safe mean"), (1, "harmful mean")):
        a.axvline(x, color=AXIS, linewidth=0.8)
        a.text(x, -0.62, t, ha="center", va="bottom", fontsize=8, color=MUTED)
    a.set_xlabel("position along the safety axis  (class-gap units)")
    a.set_ylabel("V(s) / T  =  −log ρ(s)")
    a.set_title("Potential along the axis at T = 1")
    a.legend(loc="upper center")

    for model, label, c in MODELS:
        zs = [z_of(landscape(model, T), landscape(model, T)["ou"]["pooled"]["s_eq"]) for T in TEMPS]
        b.plot(TEMPS, zs, color=c, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        b.annotate(label, (TEMPS[-1], zs[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    for y, t in ((0, "safe mean"), (1, "harmful mean")):
        b.axhline(y, color=AXIS, linewidth=0.8)
        b.text(TEMPS[-1], y, t, ha="right", va="bottom", fontsize=8, color=MUTED)
    b.set_xticks(TEMPS)
    b.set_xlim(0.6, 1.75)
    b.set_xlabel("sampling temperature T")
    b.set_ylabel("well bottom  (class-gap units)")
    b.set_title("Well bottom vs temperature (shape barely changes)")
    b.legend(loc="upper left")
    fig.savefig(OUT / "well.png", dpi=200)


def survival() -> None:
    fig, (a, b) = figure("The hazard is constant along a response (a fixed well); temperature multiplies it")
    model = MODELS[2][0]
    for T in TEMPS:
        r = landscape(model, T)
        S = np.array(r["survival"]["S"])
        y = -np.log(S)
        a.plot(np.arange(len(S)), y, color=RAMP[T], label=f"T = {T}")
        a.annotate(f"T = {T}", (len(S) - 1, y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    a.set_xlim(0, 185)
    a.set_xlabel("generated tokens t")
    a.set_ylabel("−log S(t),  S = P[no crossing of s* up to t]")
    a.set_title("Qwen2.5-7B-Instruct, token-level first passage")
    a.legend(loc="upper left")

    floor = 1 / 2048
    for model, label, c in MODELS:
        f = np.array([landscape(model, T)["survival_cummean"]["final_above"] for T in TEMPS])
        shown = np.maximum(f, floor)
        b.plot(TEMPS, shown, color=c, label=label)
        for T, v, s in zip(TEMPS, f, shown, strict=True):
            # Open marker: zero hits in 2048, drawn at the detection floor.
            b.plot([T], [s], marker="o", markersize=7, linestyle="none", markeredgewidth=2,
                   markerfacecolor=c if v > 0 else SURFACE, markeredgecolor=SURFACE if v > 0 else c)
        b.annotate(label, (TEMPS[-1], shown[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    b.axhline(floor, color=AXIS, linewidth=0.8)
    b.text(TEMPS[-1], floor, "none of 2048 (open marker)", ha="right", va="bottom", fontsize=8, color=MUTED)
    b.set_yscale("log")
    b.set_xticks(TEMPS)
    b.set_xlim(0.6, 1.8)
    b.set_xlabel("sampling temperature T")
    b.set_ylabel("fraction of completions ending above s*")
    b.set_title("Response-level failure rate vs temperature")
    b.legend(loc="center left")
    fig.savefig(OUT / "survival.png", dpi=200)


def cloning() -> None:
    d = jspace.model_dir("google/gemma-2-2b-it") / "clone"
    runs = json.loads((d / "p3-mid.json").read_text())["runs"]
    ref = json.loads((d / "p3-ref.json").read_text())["runs"]
    a_key = "-25.0"
    p = lambda r: r["p_mean"] if isinstance(r["p_mean"], float) else r["p_mean"][a_key]  # noqa: E731
    fr = lambda r: r["frac_mean"] if isinstance(r["frac_mean"], float) else r["frac_mean"][a_key]  # noqa: E731
    ks = sorted({r["k"] for r in runs})
    ref_p = np.mean([p(r) for r in ref])

    fig, (a, b) = figure("Cloning matches the brute-force reference at a quarter of its budget, then over-tilts into collapse")
    c = MODELS[1][2]
    x = np.arange(len(ks))
    for i, k in enumerate(ks):
        vals = np.array([p(r) for r in runs if r["k"] == k])
        a.plot(np.full(len(vals), i), np.maximum(vals, 1e-11), linestyle="none", marker="o", markersize=7,
               markerfacecolor=SURFACE, markeredgecolor=c, markeredgewidth=2)
        a.plot([i], [max(vals.mean(), 1e-11)], linestyle="none", marker="o", markersize=8, color=c,
               markeredgecolor=SURFACE, markeredgewidth=2)
    a.axhline(ref_p, color=AXIS, linewidth=0.8)
    a.text(x[-1] + 0.35, ref_p, f"brute force, 2048 completions: {ref_p:.2g}", ha="right", va="top", fontsize=8, color=MUTED)
    a.set_yscale("log")
    a.set_xticks(x, [f"k = {k:g}" if k else "k = 0\n(brute force)" for k in ks])
    a.set_xlabel("tilt k   (each run: 256 walkers × 8 blocks × 16 tokens)")
    a.set_ylabel("estimate of P[mean score ≥ s*]")
    a.set_title("gemma-2-2b-it: single runs (open), mean of 3 (filled)")
    a.annotate("0 hits", (0, 1e-11), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)

    for key, label, col in (("min_parent_frac", "parents kept at the worst block", "#2a78d6"), (None, "walkers with mean ≥ s* (raw)", "#eb6834")):
        ys = [np.mean([(r[key] if key else fr(r)) for r in runs if r["k"] == k]) for k in ks]
        b.plot(x, ys, color=col, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    b.axhline(0.5, color=AXIS, linewidth=0.8)
    b.text(x[-1], 0.5, "event typical", ha="right", va="bottom", fontsize=8, color=MUTED)
    b.set_xticks(x, [f"k = {k:g}" for k in ks])
    b.set_ylim(0, 1.05)
    b.set_ylabel("fraction")
    b.set_title("Population diagnostics (mean of 3 runs)")
    b.legend(loc="center left", bbox_to_anchor=(0.0, 0.3))
    fig.savefig(OUT / "cloning.png", dpi=200)


if __name__ == "__main__":
    style()
    OUT.mkdir(exist_ok=True)
    well()
    survival()
    cloning()
    print("wrote", ", ".join(str(p) for p in sorted(OUT.glob("*.png"))))
