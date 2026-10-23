"""Safety axes: a unit vector u_l per layer, pointing toward the unsafe class.
-> runs/<model>/axes/<set>.pt (vectors + class statistics) and .json (scores)

    uv run axes.py --model Qwen/Qwen3-0.6B --data data/contrast/all-advice.jsonl

Three constructions, all scored on the same paired safe/harmful responses
(make_data.py), so they can be compared on one footing:

  diffmean  mean response residual of the harmful class minus the safe one,
            per layer (Arditi et al.; the default of the proposal, Part B2).
  pullback  a_l = J_l^T w: the final-layer diffmean w pulled back through the
            web-text lens (Part B2, Fig. 2d). Needs runs/<model>/base/lens.pt.
  contrast  top right-singular vector of J_diff[l] — the layer-l residual
            direction whose transport differs most between the classes.
            Needs runs/<model>/contrast/lens.pt. Sign aligned to diffmean.

"Measurable" is decided per layer from the per-response mean projection:
d' between the classes, AUC, and the fraction of pairs ordered correctly
(harmful above safe) — scored on held-out pairs (odd pair ids), with the
directions rebuilt from the even ones. The saved vectors use every pair. The
lenses behind pullback/contrast were themselves fitted on all pairs, so for
those two the split only guards the diffmean part. The .pt carries
mu_safe / mu_harm / sigma per layer so trajectory.py and landscape.py can
place the threshold s*.

Residual indexing follows jlens: layer l is the output of block l, so the
final layer is n_layers - 1 (what the lens transports *to*).
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from jlens import ActivationRecorder

import jspace


def response_means(model, rows: list[dict], max_seq_len: int) -> torch.Tensor:
    """[n_rows, n_layers, d]: residual averaged over each row's response tokens."""
    layers = list(range(model.n_layers))
    H = torch.zeros(len(rows), model.n_layers, model.d_model)
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        ids = model.encode(row["text"], max_length=max_seq_len)
        # Response starts where the shared prefix ends; the prefix is the
        # request and must not enter the class means (same rule as contrast.py).
        p0 = model.encode(row["prefix"], max_length=max_seq_len).shape[1] if row.get("prefix") else 1
        if ids.shape[1] <= p0:
            raise SystemExit(f"row {i}: truncated to the prefix, raise --max-seq-len")
        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(ids)
            for l in layers:
                H[i, l] = rec.activations[l][0, p0:].float().mean(0).cpu()
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}  {time.perf_counter() - t0:.0f}s")
    return H


def separation(score: torch.Tensor, y: torch.Tensor, pair: torch.Tensor) -> dict:
    """How well a 1-D score tells harmful (y=-1) from safe (y=+1) responses."""
    h, s = score[y == -1], score[y == 1]
    d_prime = (h.mean() - s.mean()) / ((h.var() + s.var()) / 2).sqrt()
    auc = (h[:, None] > s[None, :]).float().mean()
    # Paired accuracy: the harmful member above the safe one, pair by pair.
    pairs = {}
    for v, lab, pid in zip(score.tolist(), y.tolist(), pair.tolist(), strict=True):
        pairs.setdefault(pid, {})[lab] = v
    ordered = [p[-1] > p[1] for p in pairs.values() if len(p) == 2]
    return {
        "d_prime": d_prime.item(),
        "auc": auc.item(),
        "pair_acc": sum(ordered) / max(len(ordered), 1),
        "mu_safe": s.mean().item(),
        "mu_harm": h.mean().item(),
        "sigma": ((h.var() + s.var()) / 2).sqrt().item(),
    }


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build(model, H: torch.Tensor, y: torch.Tensor, model_id: str) -> dict[str, torch.Tensor]:
    """{method: [n_layers, d] unit vectors} from the response means of one split."""
    from jlens import JacobianLens

    n_layers, last = model.n_layers, model.n_layers - 1
    # u points toward the unsafe class: label -1 is harmful (make_data.py).
    methods = {"diffmean": unit(H[y == -1].mean(0) - H[y == 1].mean(0))}

    # Lenses are optional: without them only diffmean is built.
    base = jspace.run_dir(model_id, "base") / "lens.pt"
    if base.exists():
        lens = JacobianLens.load(str(base))  # ponytail: reloaded per call, 2 calls
        w = methods["diffmean"][last]
        U = torch.stack([
            lens.jacobians[l].T @ w if l in lens.jacobians else w for l in range(n_layers)
        ])
        methods["pullback"] = unit(U)

    contrast = jspace.run_dir(model_id, "contrast") / "lens.pt"
    if contrast.exists():
        diff = JacobianLens.load(str(contrast))
        U = torch.zeros(n_layers, model.d_model)
        for l in range(n_layers):
            if l not in diff.jacobians:
                U[l] = methods["diffmean"][l]
                continue
            _, _, Vh = torch.linalg.svd(diff.jacobians[l], full_matrices=False)
            v = Vh[0]
            # The SVD fixes (u_k, v_k) only up to a shared sign.
            U[l] = v if v @ methods["diffmean"][l] >= 0 else -v
        methods["contrast"] = unit(U)

    return methods


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/contrast/all-advice.jsonl")
    p.add_argument("--name", help="output basename (default: the data file's stem)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--limit", type=int, help="first N rows (quick check)")
    args = p.parse_args()

    rows = jspace.load_jsonl(args.data)[: args.limit]
    name = args.name or args.data.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    out = jspace.model_dir(args.model) / "axes"
    out.mkdir(exist_ok=True)

    model, _, _ = jspace.load_model(args.model, device=args.device, dtype=args.dtype)
    print(f"{model} | {len(rows)} responses from {args.data}")
    H = response_means(model, rows, args.max_seq_len)
    y = torch.tensor([int(r["label"]) for r in rows])
    pair = torch.tensor([int(r.get("pair", i)) for i, r in enumerate(rows)])
    n_layers = model.n_layers

    methods = build(model, H, y, args.model)
    # Held-out scoring: directions from even pairs, separation on odd pairs.
    train, test = pair % 2 == 0, pair % 2 == 1
    held = build(model, H[train], y[train], args.model)

    report, stats = {}, {}
    for m, U in methods.items():
        per_layer = [separation(H[test, l] @ held[m][l], y[test], pair[test]) for l in range(n_layers)]
        report[m] = per_layer
        # Class statistics along the *saved* (all-pair) vectors: what s* is set from.
        full = [separation(H[:, l] @ U[l], y, pair) for l in range(n_layers)]
        stats[m] = {k: [pl[k] for pl in full] for k in ("mu_safe", "mu_harm", "sigma")}
        best = max(range(n_layers), key=lambda l: per_layer[l]["d_prime"])
        print(f"{m:9s} best layer {best:2d} (held-out): d'={per_layer[best]['d_prime']:.2f} "
              f"auc={per_layer[best]['auc']:.3f} pair_acc={per_layer[best]['pair_acc']:.3f}")
        print("          d' by layer: " + " ".join(f"{pl['d_prime']:.1f}" for pl in per_layer))

    names = list(methods)
    cos = {
        f"{a}/{b}": [(methods[a][l] @ methods[b][l]).item() for l in range(n_layers)]
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    for k, v in cos.items():
        print(f"cos {k}: " + " ".join(f"{c:+.2f}" for c in v))

    torch.save(
        {"model": args.model, "data": args.data, "n_rows": len(rows),
         "layers": list(range(n_layers)),
         "u": methods, "stats": stats},
        out / f"{name}.pt",
    )
    (out / f"{name}.json").write_text(json.dumps(
        {"model": args.model, "data": args.data, "n_rows": len(rows),
         "n_layers": n_layers, "separation": report, "cosine": cos}
    ))
    print(f"wrote {out / name}.pt / .json")


if __name__ == "__main__":
    main()
