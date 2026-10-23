"""Structural analysis of the fitted J matrices -> runs/<model>/analysis.json.

    uv run analyze.py --model Qwen/Qwen3-0.6B

Model-free: reads only lens.pt and gram.pt. The JSON is small on purpose —
it is the thing you hand to a Claude artifact.

Per layer:
  fro          ||J_l||_F / sqrt(d)          overall transport gain
  identity_cos <J_l, I> / (||J_l||_F sqrt(d))  how close to pure pass-through
  stable_rank  ||J||_F^2 / ||J||_2^2        rank robust to small singular values
  eff_rank     (sum s)^2 / sum s^2          participation ratio of the spectrum
  sv           top singular values, /s_0

Across layers:
  cka[l][m]    linear CKA between the J-lens vector geometries (rows of W_U J).
               The paper's layer-similarity analysis; sensory / workspace /
               motor zones show up as blocks on this matrix.
"""

from __future__ import annotations

import argparse
import json

import torch

import jspace


def cka_matrix(J: dict[int, torch.Tensor], G: torch.Tensor, device: str) -> list[list[float]]:
    """Linear CKA between rows of W_U @ J_l, for every layer pair.

    Centering over the vocab axis is idempotent, so
    (C W_U J_l)^T (C W_U J_m) = J_l^T G J_m with G = W_U^T C W_U. Never
    materialises anything vocab-sized.
    """
    layers = sorted(J)
    G = G.to(device, torch.float32)
    Js = [J[l].to(device, torch.float32) for l in layers]
    GJ = [G @ x for x in Js]  # [d, d] each
    n = len(layers)
    cross = torch.zeros(n, n, device=device)
    for a in range(n):
        for b in range(a, n):
            cross[a, b] = cross[b, a] = (Js[a].T @ GJ[b]).norm() ** 2
    self_norm = cross.diagonal().sqrt()
    return (cross / torch.outer(self_norm, self_norm)).cpu().tolist()


#: Histogram support for log10(sigma_k / sigma_0), and its bin count.
SV_LOG_RANGE, SV_BINS = (-4.0, 0.0), 64


def sv_histogram(s: torch.Tensor) -> list[int]:
    """Counts of the *whole* spectrum in fixed log bins, so the bulk is visible
    as a distribution rather than as a curve. Fixed bins keep every layer and
    every lens on the same axis."""
    lo, hi = SV_LOG_RANGE
    x = (s / s[0]).clamp_min(10**lo).log10()
    return torch.histc(x, bins=SV_BINS, min=lo, max=hi).int().cpu().tolist()


def eigenvalues(A: torch.Tensor, top: int) -> tuple[list[list[float]], float]:
    """Top eigenvalues by modulus, normalised by the largest.

    J_l is square (d_model x d_model — it maps a residual to a residual) but far
    from symmetric, so its eigenvalues are genuinely complex. This is a
    different object from the singular values plotted elsewhere: those are real
    for *any* matrix, since they are the square roots of the eigenvalues of the
    PSD matrix J^T J. Run on CPU — LAPACK's general eig has no CUDA path here.
    """
    ev = torch.linalg.eigvals(A.cpu().float())
    order = ev.abs().argsort(descending=True)[:top]
    ev = ev[order]
    scale = ev.abs().max().item()
    return [[round(z.real.item() / scale, 4), round(z.imag.item() / scale, 4)] for z in ev], scale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="base", help="which fitted lens (subdir of runs/<model>/)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-sv", type=int, default=0,
                   help="keep only the top N singular values; 0 = the whole spectrum")
    p.add_argument("--no-cka", action="store_true", help="skip CKA (O(n_layers^2) matmuls)")
    p.add_argument("--no-eig", action="store_true", help="skip eigenvalues (CPU-bound)")
    p.add_argument("--top-eig", type=int, default=128, help="eigenvalues kept, by modulus")
    args = p.parse_args()

    import jlens

    out = jspace.run_dir(args.model, args.tag)
    lens = jlens.JacobianLens.load(str(out / "lens.pt"))
    J = lens.jacobians
    layers = sorted(J)
    d = lens.d_model
    dev = args.device if torch.cuda.is_available() else "cpu"

    per_layer = []
    for l in layers:
        A = J[l].to(dev, torch.float32)
        s = torch.linalg.svdvals(A)
        per_layer.append(
            {
                "layer": l,
                "fro": (A.norm() / d**0.5).item(),
                "identity_cos": (A.trace() / (A.norm() * d**0.5)).item(),
                "stable_rank": ((s**2).sum() / s[0] ** 2).item(),
                "eff_rank": (s.sum() ** 2 / (s**2).sum()).item(),
                # All d_model of them by default. Truncating at 64 hides the
                # part of the decay that actually distinguishes the layers —
                # at index 63 a Qwen3-0.6B layer is still at ~0.16 of sigma_0.
                "sv": ((s[: args.top_sv] if args.top_sv else s) / s[0]).cpu().tolist(),
                "sv_hist": sv_histogram(s),
            }
        )
        if not args.no_eig:
            ev, scale = eigenvalues(A, args.top_eig)
            per_layer[-1]["eig"] = ev
            per_layer[-1]["eig_absmax"] = scale
        print(f"L{l:3d}  fro={per_layer[-1]['fro']:.3f}  eff_rank={per_layer[-1]['eff_rank']:.1f}")

    result = {
        "model": args.model,
        "d_model": d,
        "n_layers": max(layers) + 2,  # source layers stop below the target
        "n_prompts": lens.n_prompts,
        "layers": layers,
        "sv_log_range": list(SV_LOG_RANGE),
        "per_layer": per_layer,
    }

    gram_path = jspace.model_dir(args.model) / "gram.pt"
    if not args.no_cka and gram_path.exists():
        result["cka"] = cka_matrix(J, torch.load(gram_path, weights_only=True), dev)
    elif not args.no_cka:
        print(f"no {gram_path}; skipping CKA (re-run fit.py to write it)")

    (out / "analysis.json").write_text(json.dumps(result))
    print(f"wrote {out / 'analysis.json'}")


if __name__ == "__main__":
    main()
