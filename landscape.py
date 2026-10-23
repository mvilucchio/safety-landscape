"""The landscape along a safety axis -> runs/<model>/traj/<name>.landscape.json

    uv run landscape.py --model Qwen/Qwen3-0.6B --traj all-advice-diffmean-T1.0 [--layer 15] [--threshold -6.8]

Model-free: reads trajectory.py's output and asks the questions of O1(b),
Part B2, about the process s_t:

  potential  V(s)/T = -log rho(s) from the histogram of s_t, pooled and in
             three time windows. rho_t ∝ exp(-V_t/T) is the proposal's own
             estimator; a well that moves between windows is erosion.
  dynamics   Kramers-Moyal coefficients of the increments: drift D1(s) =
             E[Δs | s] and diffusion D2(s) = Var[Δs | s]/2. Overdamped
             ds = -V'(s) dt + sqrt(2T) dW gives V from the drift, independently
             of the histogram. The linear (Ornstein-Uhlenbeck) fit
             Δs = -κ (s - s_eq) summarises the well: curvature κ, bottom s_eq,
             and its stationary width sqrt(D2/κ) — per window, so erosion shows
             as s_eq moving toward s*.
  survival   first passage τ = min{t : s_t >= s*}. Kaplan-Meier survival
             (walkers that hit EOS are censored), hazard r_t, and the two fits
             of Eq. (4): fixed well (A) -log S = r0 t; eroding well (B)
             log r_t = log r0 + ε t / T, i.e. a Gompertz survival.

Layer defaults to the axis' best held-out d'; s* to the midpoint between the
class means of the graded responses along that layer's direction.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import jspace


def potential(x: np.ndarray, edges: np.ndarray) -> list[float]:
    """-log rho on fixed bins, floored at the min; NaN where the bin is empty."""
    h, _ = np.histogram(x[~np.isnan(x)], bins=edges, density=True)
    with np.errstate(divide="ignore"):
        v = -np.log(h)
    v -= np.nanmin(v[np.isfinite(v)])
    return [None if not np.isfinite(a) else float(a) for a in v]


def ou_fit(x: np.ndarray) -> dict:
    """Δs = -κ (s - s_eq) + noise, by least squares on consecutive valid pairs."""
    a, b = x[:, :-1].ravel(), x[:, 1:].ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, d = a[ok], b[ok] - a[ok]
    if len(a) < 10:
        return {"kappa": None, "s_eq": None, "sigma_stat": None, "n": int(len(a))}
    slope, icpt = np.polyfit(a, d, 1)
    kappa = -slope
    resid = d - (slope * a + icpt)
    D2 = resid.var() / 2
    return {
        "kappa": float(kappa),
        "s_eq": float(icpt / kappa) if kappa > 0 else None,
        "sigma_stat": float(np.sqrt(D2 / kappa)) if kappa > 0 else None,
        "D2": float(D2),
        "n": int(len(a)),
    }


def kramers_moyal(x: np.ndarray, n_bins: int) -> dict:
    a, b = x[:, :-1].ravel(), x[:, 1:].ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, d = a[ok], b[ok] - a[ok]
    edges = np.quantile(a, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, n_bins - 1)
    centre, D1, D2, n = [], [], [], []
    for k in range(n_bins):
        m = idx == k
        if m.sum() < 5:
            continue
        centre.append(float(a[m].mean()))
        D1.append(float(d[m].mean()))
        D2.append(float(d[m].var() / 2))
        n.append(int(m.sum()))
    # V from the drift: -V'(s) = D1 -> V(s) = -∫ D1 ds (trapezoid on bin centres).
    c, d1 = np.array(centre), np.array(D1)
    V = np.concatenate([[0.0], -np.cumsum((d1[1:] + d1[:-1]) / 2 * np.diff(c))])
    return {"s": centre, "D1": D1, "D2": D2, "n": n, "V_drift": (V - V.min()).tolist()}


def cummean(x: np.ndarray) -> np.ndarray:
    """Running mean of s over the generated tokens (NaN stays NaN after EOS):
    the first passage of this is the proposal's S_L >= aL event, a time average
    like a heat wave, and the statistic the class d' was measured on."""
    v = ~np.isnan(x)
    c = np.cumsum(np.where(v, x, 0.0), 1) / np.maximum(np.cumsum(v, 1), 1)
    return np.where(v, c, np.nan)


def survival(x: np.ndarray, s_star: float, window: int) -> dict:
    """Kaplan-Meier over generation steps with EOS as censoring, plus Eq. (4) fits."""
    N, L = x.shape
    valid = ~np.isnan(x)
    hit = valid & (x >= s_star)
    fail_t = np.where(hit.any(1), hit.argmax(1), L)  # first crossing, L = never
    censor_t = valid.sum(1)  # steps observed (NaN after EOS)
    n_at_risk, d, S = [], [], []
    surv = 1.0
    for t in range(L):
        at_risk = ((fail_t >= t) & (censor_t > t)).sum()
        died = (fail_t == t).sum()
        n_at_risk.append(int(at_risk))
        d.append(int(died))
        if at_risk:
            surv *= 1 - died / at_risk
        S.append(float(surv))
    S, t = np.array(S), np.arange(L)
    # (A) fixed well: -log S = r0 t, through the origin.
    y = -np.log(np.clip(S, 1e-12, 1))
    ok = S > 0
    r0_A = float((t[ok] * y[ok]).sum() / (t[ok] ** 2).sum()) if ok.sum() > 1 else None
    ss = (((y[ok] - y[ok].mean()) ** 2).sum() or 1.0) if ok.any() else 1.0
    R2_A = float(1 - ((y[ok] - r0_A * t[ok]) ** 2).sum() / ss) if r0_A is not None else None
    # (B) eroding well: hazard in windows, log r = log r0 + (ε/T) t.
    nw, dw, tw = [], [], []
    for w0 in range(0, L, window):
        n_w, d_w = sum(n_at_risk[w0:w0 + window]), sum(d[w0:w0 + window])
        if n_w and d_w:
            nw.append(n_w)
            dw.append(d_w)
            tw.append(w0 + (min(window, L - w0) - 1) / 2)
    fitB = None
    if len(tw) >= 3:
        slope, icpt = np.polyfit(tw, np.log(np.array(dw) / np.array(nw)), 1)
        eps_T, r0_B = float(slope), float(np.exp(icpt))
        pred = r0_B / eps_T * (np.exp(eps_T * t[ok]) - 1) if abs(eps_T) > 1e-9 else r0_B * t[ok]
        fitB = {"r0": r0_B, "eps_over_T": eps_T,
                "R2": float(1 - ((y[ok] - pred) ** 2).sum() / ss),
                "hazard_windows": {"t": tw, "r": (np.array(dw) / np.array(nw)).tolist()}}
    last = np.array([row[v][-1] if v.any() else np.nan for row, v in zip(x, valid, strict=True)])
    return {
        "s_star": s_star,
        "final_above": float(np.nanmean(last >= s_star)),
        "S": S.tolist(), "hazard": (np.array(d) / np.maximum(n_at_risk, 1)).tolist(),
        "n_at_risk": n_at_risk,
        "p_fail": float((fail_t < L).mean()),
        "failed_at_t0": float((fail_t == 0).mean()),
        "median_tau": float(np.median(fail_t[fail_t < L])) if (fail_t < L).any() else None,
        "fit_A": {"r0": r0_A, "R2": R2_A},
        "fit_B": fitB,
        "fail_t": fail_t.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--traj", required=True, help="basename under runs/<model>/traj/")
    p.add_argument("--layer", type=int, help="default: best held-out d' of the axis")
    p.add_argument("--threshold", type=float, help="s*; default midpoint of the class means")
    p.add_argument("--bins", type=int, default=48)
    p.add_argument("--km-bins", type=int, default=24)
    p.add_argument("--window", type=int, default=8, help="steps per hazard window (fit B)")
    args = p.parse_args()

    path = jspace.model_dir(args.model) / "traj" / f"{args.traj}.pt"
    d = torch.load(path, weights_only=True)
    s_all = d["s"].float().numpy()  # [N, L, n_layers]
    T = d["temperature"]
    stats, dprime = d["stats"], d["d_prime"]
    N, L, n_layers = s_all.shape
    layer = args.layer if args.layer is not None else int(np.argmax(dprime))

    def s_star_of(l):
        return args.threshold if args.threshold is not None else (stats["mu_safe"][l] + stats["mu_harm"][l]) / 2

    x = s_all[:, :, layer]
    s_star = s_star_of(layer)
    lo, hi = np.nanquantile(x, [0.001, 0.999])
    edges = np.linspace(lo, hi, args.bins + 1)
    thirds = [(0, L // 3), (L // 3, 2 * L // 3), (2 * L // 3, L)]
    valid_t = ~np.isnan(x)

    result = {
        "model": d["model"], "traj": args.traj, "axes": d["axes"], "method": d["method"],
        "temperature": T, "layer": layer, "d_prime": dprime[layer], "s_star": s_star,
        "mu_safe": stats["mu_safe"][layer], "mu_harm": stats["mu_harm"][layer],
        "n_samples": N, "n_prompts": len(d["prompts"]), "max_new_tokens": L,
        "eos_fraction": float(1 - valid_t[:, -1].mean()),
        "potential": {
            "edges": edges.tolist(),
            "pooled": potential(x, edges),
            "windows": [{"t": list(w), "V": potential(x[:, w[0]:w[1]], edges)} for w in thirds],
        },
        "mean_t": np.nanmean(x, 0).tolist(),
        "std_t": np.nanstd(x, 0).tolist(),
        "above_t": np.nanmean(np.where(valid_t, x >= s_star, np.nan), 0).tolist(),
        "ou": {"pooled": ou_fit(x), "windows": [ou_fit(x[:, w[0]:w[1]]) for w in thirds]},
        "kramers_moyal": kramers_moyal(x, args.km_bins),
        # token: Eq. (2), s_t itself crosses s*. cummean: the running average does.
        "survival": survival(x, s_star, args.window),
        "survival_cummean": survival(cummean(x), s_star, args.window),
    }
    # Per prompt: how many of its samples fail -> P_fail(M) across prompts.
    pi = d["prompt_idx"].numpy()
    ft = np.array(result["survival"]["fail_t"])
    per_prompt = [float((ft[pi == k] < L).mean()) for k in range(len(d["prompts"]))]
    result["per_prompt_fail"] = per_prompt
    result["p_fail_any_sample"] = float(np.mean([f > 0 for f in per_prompt]))
    # Cross-layer summary at each layer's own midpoint threshold.
    result["per_layer"] = []
    for l in range(n_layers):
        xl = s_all[:, :, l]
        sv, svc, ou = survival(xl, s_star_of(l), args.window), survival(cummean(xl), s_star_of(l), args.window), ou_fit(xl)
        result["per_layer"].append({
            "layer": l, "d_prime": dprime[l], "s_star": s_star_of(l), "p_fail": sv["p_fail"],
            "p_fail_cummean": svc["p_fail"], "failed_at_t0": sv["failed_at_t0"],
            "kappa": ou["kappa"], "s_eq": ou["s_eq"], "sigma_stat": ou["sigma_stat"],
        })

    dest = path.with_suffix(".landscape.json")
    dest.write_text(json.dumps(result))

    sv, svc, ou = result["survival"], result["survival_cummean"], result["ou"]
    print(f"{args.traj}  layer {layer} (d'={dprime[layer]:.2f})  T={T}  N={N}  L={L}  "
          f"EOS {result['eos_fraction']:.2f}")
    print(f"  s* = {s_star:.2f}  (safe {stats['mu_safe'][layer]:.2f} | harm {stats['mu_harm'][layer]:.2f})")
    print(f"  token   s_t >= s*:  P[tau <= L] = {sv['p_fail']:.3f}   at t=0: {sv['failed_at_t0']:.3f}   "
          f"median tau = {sv['median_tau']}   prompts with >=1 failure: {result['p_fail_any_sample']:.2f}")
    print(f"  cummean s_t >= s*:  P[tau <= L] = {svc['p_fail']:.3f}   "
          f"final mean >= s*: {svc['final_above']:.3f}")
    for name, o in zip(["early", "mid", "late"], ou["windows"], strict=True):
        if o["kappa"] is not None:
            print(f"  well {name:5s}: kappa={o['kappa']:.3f}  s_eq={o['s_eq'] if o['s_eq'] is None else round(o['s_eq'], 2)}  "
                  f"width={o['sigma_stat'] if o['sigma_stat'] is None else round(o['sigma_stat'], 2)}")
    print(f"  fit A (fixed well):   r0={sv['fit_A']['r0']}  R2={sv['fit_A']['R2']}")
    if sv["fit_B"]:
        print(f"  fit B (eroding well): r0={sv['fit_B']['r0']:.3g}  eps/T={sv['fit_B']['eps_over_T']:.3g}  R2={sv['fit_B']['R2']:.3f}")
    print("  per layer p_fail token:   " + " ".join(f"{pl['p_fail']:.2f}" for pl in result["per_layer"]))
    print("  per layer p_fail cummean: " + " ".join(f"{pl['p_fail_cummean']:.2f}" for pl in result["per_layer"]))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
