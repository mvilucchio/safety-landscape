"""Safety axes: a unit vector u_l per layer, pointing toward the unsafe class.
-> runs/<model>/axes/<set>.pt (vectors + class statistics) and .json (scores)

    uv run axes.py --model Qwen/Qwen3-0.6B --data data/contrast/all-advice.jsonl

Three constructions, all scored on the same paired safe/harmful responses
(make_data.py), so they can be compared on one footing:

  diffmean  mean response residual of the harmful class minus the safe one,
            per layer (Arditi et al.; the proposal's default, Part B2).
  pullback  a_l = J_l^T w: the final-layer diffmean w pulled back through the
            web-text lens (Part B2, Fig. 2d). Needs runs/<model>/base/lens.pt.
  contrast  top right-singular vector of J_diff[l], the layer-l direction
            whose transport differs most between the classes. Needs
            runs/<model>/contrast/lens.pt. Sign aligned to diffmean.
  coherence mean residual of the same responses with their words shuffled
            minus the originals: points toward incoherent text. Not a safety
            axis; it is the confound to measure and remove.
  diffmean-orth  diffmean with its coherence component projected out.

"Measurable" is decided per layer from the per-response mean projection: d'
between the classes, AUC, and the fraction of pairs ordered correctly
(harmful above safe), scored on held-out pairs (odd ids) with the directions
rebuilt from the even ones. The saved vectors use every pair. The lenses
behind pullback/contrast were fitted on all pairs, so for those two the split
only guards the diffmean part. The .pt carries mu_safe / mu_harm / sigma per
layer so trajectory.py and clone.py can place the threshold s*.

--template chat renders the rows through the tokenizer's chat template
(instruct models); downstream scripts reuse whatever the axes were built with.

Residual indexing follows jlens: layer l is the output of block l, so the
final layer is n_layers - 1 (what the lens transports *to*).
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch
from jlens import ActivationRecorder, JacobianLens

import jspace


def response_means(model, tok, rows: list[dict], template: str, max_seq_len: int) -> torch.Tensor:
    """[n_rows, n_layers, d]: residual averaged over each row's response tokens."""
    layers = list(range(model.n_layers))
    H = torch.zeros(len(rows), model.n_layers, model.d_model)
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        r = jspace.render(row, tok, template)
        ids = model.encode(r["text"], max_length=max_seq_len)
        # The request must not enter the class means (same rule as contrast.py).
        p0 = model.encode(r["prefix"], max_length=max_seq_len).shape[1]
        if ids.shape[1] <= p0:
            raise SystemExit(f"row {i}: truncated to the prefix, raise --max-seq-len")
        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(ids)
            for l in layers:
                H[i, l] = rec.activations[l][0, p0:].float().mean(0).cpu()
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}  {time.perf_counter() - t0:.0f}s")
    return H


def shuffled(rows: list[dict], seed: int = 0) -> list[dict]:
    """The same rows with the words of each response in random order: same
    tokens, same topic, no syntax. The contrast against the originals is the
    coherence direction."""
    rng = random.Random(seed)
    out = []
    for r in rows:
        words = r["response"].split()
        rng.shuffle(words)
        resp = " ".join(words)
        out.append({**r, "response": resp, "text": r["prefix"] + resp})
    return out


def separation(score: torch.Tensor, y: torch.Tensor, pair: torch.Tensor) -> dict:
    """How well a 1-D score tells harmful (y=-1) from safe (y=+1) responses."""
    h, s = score[y == -1], score[y == 1]
    sigma = ((h.var() + s.var()) / 2).sqrt()
    # Rows come in complete pairs (make_data.py): sort by (pair, label) and
    # fold into [n_pairs, 2] = [harmful, safe].
    pm = score[torch.argsort(pair * 2 + (y == 1).long())].view(-1, 2)
    return {
        "d_prime": ((h.mean() - s.mean()) / sigma).item(),
        "auc": (h[:, None] > s[None, :]).float().mean().item(),
        "pair_acc": (pm[:, 0] > pm[:, 1]).float().mean().item(),
        "mu_safe": s.mean().item(), "mu_harm": h.mean().item(), "sigma": sigma.item(),
    }


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def directions(H, y, J: dict | None, V: torch.Tensor | None, H2: torch.Tensor) -> dict[str, torch.Tensor]:
    """{method: [n_layers, d] unit vectors}. J: the web-text lens' {layer: J_l};
    V: unsigned top right-singular vectors of J_diff, NaN rows where absent;
    H2: response means of the word-shuffled rows."""
    # u points toward the unsafe class: label -1 is harmful (make_data.py).
    u = unit(H[y == -1].mean(0) - H[y == 1].mean(0))
    c = unit(H2.mean(0) - H.mean(0))  # toward incoherent
    out = {"diffmean": u, "coherence": c,
           "diffmean-orth": unit(u - (u * c).sum(1, keepdim=True) * c)}
    if J is not None:  # identity at the final layer, which the lens maps *to*
        w = u[-1]
        out["pullback"] = unit(torch.stack([J[l].T @ w if l in J else w for l in range(len(u))]))
    if V is not None:  # the SVD fixes (u_k, v_k) only up to a shared sign
        sign = torch.nan_to_num((V * u).sum(1, keepdim=True)).sign()
        out["contrast"] = unit(torch.where(V.isnan(), u, V * torch.where(sign == 0, 1.0, sign)))
    return out


def top_singular_vectors(lens: JacobianLens, n_layers: int, device: str) -> torch.Tensor:
    V = torch.full((n_layers, lens.d_model), float("nan"))
    for l, A in lens.jacobians.items():
        V[l] = torch.linalg.svd(A.to(device), full_matrices=False).Vh[0].cpu()
    return V


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/contrast/all-advice.jsonl")
    p.add_argument("--template", default="plain", choices=["plain", "chat"])
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

    model, _, tok = jspace.load_model(args.model, device=args.device, dtype=args.dtype)
    print(f"{model} | {len(rows)} responses from {args.data} | {args.template} template")
    H = response_means(model, tok, rows, args.template, args.max_seq_len)
    H2 = response_means(model, tok, shuffled(rows), args.template, args.max_seq_len)
    y = torch.tensor([int(r["label"]) for r in rows])
    pair = torch.tensor([int(r["pair"]) for r in rows])
    n_layers = model.n_layers

    # Lenses are optional: without them only diffmean is built.
    base, contrast = (jspace.run_dir(args.model, t) / "lens.pt" for t in ("base", "contrast"))
    J = JacobianLens.load(str(base)).jacobians if base.exists() else None
    V = top_singular_vectors(JacobianLens.load(str(contrast)), n_layers, args.device) if contrast.exists() else None

    methods = directions(H, y, J, V, H2)
    train, test = pair % 2 == 0, pair % 2 == 1
    held = directions(H[train], y[train], J, V, H2[train])

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
    cos = {f"{a}/{b}": (methods[a] * methods[b]).sum(1).tolist()
           for i, a in enumerate(names) for b in names[i + 1:]}
    for k, v in cos.items():
        print(f"cos {k}: " + " ".join(f"{c:+.2f}" for c in v))

    meta = {"model": args.model, "data": args.data, "template": args.template, "n_rows": len(rows)}
    torch.save({**meta, "layers": list(range(n_layers)), "u": methods, "stats": stats}, out / f"{name}.pt")
    (out / f"{name}.json").write_text(json.dumps(
        {**meta, "n_layers": n_layers, "separation": report, "cosine": cos}))
    print(f"wrote {out / name}.pt / .json")


if __name__ == "__main__":
    main()
