"""Fit a contrast lens: J_diff = mean(J | label +1) - mean(J | label -1).

    uv run make_data.py --set medical-advice
    uv run contrast.py --model Qwen/Qwen3-0.6B --data data/contrast/medical-advice.jsonl \
        --shard 0/2 --device cuda:0 &
    uv run contrast.py --model Qwen/Qwen3-0.6B --data data/contrast/medical-advice.jsonl \
        --shard 1/2 --device cuda:1 &
    wait && uv run contrast.py --model Qwen/Qwen3-0.6B --merge

Writes runs/<model>/contrast/{lens,lens.pos,lens.neg}.pt. `lens.pt` is the
difference and is a normal JacobianLens, so analyze.py / spectrum.py / report.py
all take it with --tag contrast.

Two things this does that a plain jlens.fit cannot:

1. **Signed accumulation.** Each prompt's Jacobian enters weighted by its label,
   normalised per class, so one pass yields J_pos, J_neg and their difference.
2. **Response-only positions.** Both members of a pair share a request; the
   Jacobian average is restricted to positions after that shared prefix, so the
   request cannot contribute to the difference. This is why the data carries a
   `prefix` field.

The per-prompt Jacobian itself is jlens.jacobian_for_prompt, unmodified.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import jlens
import torch
from jlens.fitting import SKIP_FIRST_N_POSITIONS

import jspace

logger = logging.getLogger("contrast")


def prefix_len(model, row: dict, max_seq_len: int) -> int:
    """Token position where this prompt's own (unshared) content begins.

    Falls back to the jlens default when a row has no prefix — a plain
    unpaired contrast set still works, it just averages over everything.
    """
    if not row.get("prefix"):
        return SKIP_FIRST_N_POSITIONS
    n = model.encode(row["prefix"], max_length=max_seq_len).shape[1]
    return max(n, SKIP_FIRST_N_POSITIONS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", help="labelled .jsonl from make_data.py")
    p.add_argument("--tag", default="contrast", help="output subdirectory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="responses are long; 128 would truncate away the contrast")
    p.add_argument("--dim-batch", type=int, default=32)
    p.add_argument("--layer-stride", type=int, default=1)
    p.add_argument("--limit", type=int, help="first N prompts (quick check)")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--merge", action="store_true", help="combine shard sums and exit")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = jspace.run_dir(args.model, args.tag)
    i, n = (int(x) for x in args.shard.split("/"))

    if args.merge:
        return merge(out)
    if not args.data:
        raise SystemExit("--data is required unless --merge")

    rows = jspace.load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    # Shard by pair, never by row: the two members of a pair are consecutive, so
    # rows[i::n] would hand one shard every positive and the other every negative.
    rows = [r for k, r in enumerate(rows) if r.get("pair", k) % n == i]

    model, _, _ = jspace.load_model(args.model, device=args.device, dtype=args.dtype)
    layers = list(range(0, model.n_layers - 1, args.layer_stride))
    d = model.d_model
    logger.info("%s | %d prompts | %d layers | shard %d/%d", model, len(rows), len(layers), i, n)

    ckpt = out / f"sums.shard{i}.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        logger.info("  resuming at prompt %d", state["next_idx"])
    else:
        state = {
            "sum": {1: {l: torch.zeros(d, d) for l in layers},
                    -1: {l: torch.zeros(d, d) for l in layers}},
            "count": {1: 0, -1: 0},
            "next_idx": 0,
            "layers": layers,
        }

    def save():
        tmp = f"{ckpt}.tmp.{os.getpid()}"
        torch.save(state, tmp)
        os.replace(tmp, ckpt)

    for idx, row in enumerate(rows):
        if idx < state["next_idx"]:
            continue
        label = int(row["label"])
        if label not in (1, -1):
            raise SystemExit(f"row {idx}: label must be +1 or -1, got {label!r}")
        skip = prefix_len(model, row, args.max_seq_len)
        t0 = time.perf_counter()
        try:
            J, seq_len, n_valid = jlens.jacobian_for_prompt(
                model, row["text"], layers,
                dim_batch=args.dim_batch, max_seq_len=args.max_seq_len, skip_first=skip,
            )
        except ValueError as exc:  # prompt truncated to at-or-below the prefix
            logger.warning("  skipping %d: %s", idx, exc)
            state["next_idx"] = idx + 1
            continue

        for l in layers:
            state["sum"][label][l] += J[l]
        state["count"][label] += 1
        state["next_idx"] = idx + 1
        logger.info("  %d/%d  label=%+d  seq_len=%d resp_positions=%d  %.0fs  (+%d/-%d)",
                    idx + 1, len(rows), label, seq_len, n_valid,
                    time.perf_counter() - t0, state["count"][1], state["count"][-1])
        if (idx + 1) % 10 == 0:
            save()

    save()
    logger.info("shard done: %d positive, %d negative", state["count"][1], state["count"][-1])
    if n == 1:
        merge(out)


def merge(out) -> None:
    """Combine shard sums into the pos/neg means and their difference."""
    shards = sorted(out.glob("sums.shard*.pt"))
    if not shards:
        raise SystemExit(f"no sums.shard*.pt in {out}")
    states = [torch.load(s, map_location="cpu", weights_only=True) for s in shards]
    layers = states[0]["layers"]

    means, counts = {}, {}
    for label in (1, -1):
        counts[label] = sum(s["count"][label] for s in states)
        if counts[label] == 0:
            raise SystemExit(f"no prompts with label {label:+d}")
        means[label] = {
            l: sum(s["sum"][label][l] for s in states) / counts[label] for l in layers
        }

    diff = {l: means[1][l] - means[-1][l] for l in layers}
    n_total = counts[1] + counts[-1]
    for name, J, n_p in (("lens.pos", means[1], counts[1]),
                         ("lens.neg", means[-1], counts[-1]),
                         ("lens", diff, n_total)):
        # float32: the difference is small relative to either mean, and the
        # default float16 save would eat a chunk of it.
        jlens.JacobianLens(jacobians=J, n_prompts=n_p, d_model=J[layers[0]].shape[0]).save(
            str(out / f"{name}.pt"), dtype=torch.float32
        )

    ratio = {l: (diff[l].norm() / means[1][l].norm()).item() for l in layers}
    peak = max(ratio, key=ratio.get)
    print(f"merged {len(shards)} shards: {counts[1]} positive, {counts[-1]} negative")
    print(f"wrote {out / 'lens.pt'} (difference), lens.pos.pt, lens.neg.pt")
    print(f"‖J_diff‖/‖J_pos‖ peaks at layer {peak} ({ratio[peak]:.3f})")

    split_half(states, layers, out)


def split_half(states, layers, out) -> None:
    """Noise floor. Shards hold disjoint *pairs*, so each gives an independent
    estimate of the same difference matrix. If the difference were sampling
    noise the two would be uncorrelated; a cosine near 0 means the layer's
    contrast is not yet resolved at this sample size. Report it — a difference
    matrix always *looks* like something.
    """
    if len(states) < 2:
        print("split-half agreement: needs >= 2 shards, skipped")
        return
    import json

    agree = {}
    for l in layers:
        d = []
        for s in states:
            if s["count"][1] and s["count"][-1]:
                d.append((s["sum"][1][l] / s["count"][1] - s["sum"][-1][l] / s["count"][-1]).flatten())
        agree[l] = torch.nn.functional.cosine_similarity(d[0], d[1], dim=0).item()

    (out / "split_half.json").write_text(json.dumps(agree))
    best = max(agree, key=agree.get)
    print(f"split-half agreement: max {agree[best]:.3f} at layer {best}, "
          f"median {sorted(agree.values())[len(agree) // 2]:.3f}  (0 = pure noise)")


if __name__ == "__main__":
    main()
