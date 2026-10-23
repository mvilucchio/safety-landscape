"""Lens quality on the paper's prompt sets -> runs/<model>/eval-<set>.json.

    uv run eval.py --model Qwen/Qwen3-0.6B --set multihop
    uv run eval.py --model Qwen/Qwen3-0.6B --set all

Metric (data/README.md): read out at the final prompt token across all layers;
pass@k = mean over items of the fraction of `intermediates` whose
min-over-layers lens rank is <= k. The per-layer curve is the interesting part
— it is where the workspace band shows up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import jspace

SETS = [p.stem.replace("lens-eval-", "") for p in sorted(Path("data").glob("lens-eval-*.json"))]
KS = (1, 5, 10, 50)


def token_variants(word: str, tok) -> list[int]:
    """Single-token ids for a word, over spacing/case variants.

    ponytail: order-ops also wants digit<->word synonyms ("5"/"five"); add a
    synonym table here if that set's numbers look under-counted.
    """
    ids = set()
    for form in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        enc = tok.encode(form, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


def run_set(name: str, model, lens, tok, limit: int | None) -> dict:
    items = json.loads(Path(f"data/lens-eval-{name}.json").read_text())["items"]
    if limit:
        items = items[:limit]
    layers = lens.source_layers

    # ranks[layer] -> list of per-intermediate ranks, aligned across layers.
    per_layer_ranks: dict[int, list[int]] = {l: [] for l in layers}
    n_skipped = 0
    for item in items:
        lens_logits, _, _ = lens.apply(model, item["prompt"], positions=[-1])
        targets = [token_variants(w, tok) for w in item["intermediates"]]
        for l in layers:
            logits = lens_logits[l][0]
            for ids in targets:
                if not ids:
                    continue
                best = min(int((logits > logits[i]).sum()) for i in ids)
                per_layer_ranks[l].append(best)
        n_skipped += sum(1 for ids in targets if not ids)

    R = torch.tensor([per_layer_ranks[l] for l in layers])  # [n_layers, n_targets]
    if R.numel() == 0:
        raise SystemExit(f"{name}: no single-token intermediates found")
    best_over_layers = R.min(dim=0).values
    return {
        "set": name,
        "n_items": len(items),
        "n_targets": R.shape[1],
        "n_skipped_multitoken": n_skipped,
        "layers": layers,
        "pass_at_k": {k: (best_over_layers <= k).float().mean().item() for k in KS},
        # Fraction of targets at rank <= k, layer by layer.
        "per_layer_frac": {k: (R <= k).float().mean(dim=1).tolist() for k in KS},
        "median_rank_per_layer": R.float().median(dim=1).values.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="base", help="which fitted lens (subdir of runs/<model>/)")
    p.add_argument("--set", default="multihop", help=f"{', '.join(SETS)}, or 'all'")
    p.add_argument("--limit", type=int, help="first N items only (quick check)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    from jlens import JacobianLens

    out = jspace.run_dir(args.model, args.tag)
    lens = JacobianLens.load(str(out / "lens.pt"))
    model, _, tok = jspace.load_model(args.model, device=args.device, dtype=args.dtype)

    for name in SETS if args.set == "all" else [args.set]:
        result = run_set(name, model, lens, tok, args.limit)
        (out / f"eval-{name}.json").write_text(json.dumps(result))
        pk = result["pass_at_k"]
        print(
            f"{name:14s} n={result['n_targets']:4d}  "
            + "  ".join(f"pass@{k}={pk[k]:.2f}" for k in KS)
        )


if __name__ == "__main__":
    main()
