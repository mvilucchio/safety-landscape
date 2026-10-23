"""Fit the Jacobian lens for a model: runs/<model>/lens.pt (+ gram.pt).

    uv run fit.py --model Qwen/Qwen3-0.6B --n-prompts 50

Two GPUs: fitting is a mean over prompts, so shard and merge (jlens.fit is
per-prompt sequential and single-device).

    uv run fit.py --model M --n-prompts 100 --shard 0/2 --device cuda:0 &
    uv run fit.py --model M --n-prompts 100 --shard 1/2 --device cuda:1 &
    wait && uv run fit.py --model M --merge
"""

from __future__ import annotations

import argparse
import hashlib
import logging

import jlens
import torch

import jspace


def _fingerprint(prompts: list[str]) -> str:
    return hashlib.sha1("\0".join(prompts).encode()).hexdigest()[:8]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument("--tag", default="base", help="which fitted lens (subdir of runs/<model>/)")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--prompts", help="local file, blank-line separated (default wikitext)")
    p.add_argument("--device", default="cuda:0", help="cuda:0 | cuda:1 | auto | cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--dim-batch", type=int, default=32, help="J rows per backward pass")
    p.add_argument("--layer-stride", type=int, default=1, help="fit every Nth layer")
    p.add_argument("--shard", default="0/1", help="i/n — fit on prompts[i::n]")
    p.add_argument("--merge", action="store_true", help="merge shard lenses and exit")
    args = p.parse_args()

    jlens.configure_logging(logging.INFO)
    out = jspace.run_dir(args.model, args.tag)
    i, n = (int(x) for x in args.shard.split("/"))

    if args.merge:
        shards = sorted(out.glob("lens.shard*.pt"))
        if not shards:
            raise SystemExit(f"no lens.shard*.pt in {out}")
        merged = jlens.JacobianLens.merge([jlens.JacobianLens.load(str(s)) for s in shards])
        merged.save(str(out / "lens.pt"))
        print(f"merged {len(shards)} shards -> {out / 'lens.pt'} ({merged.n_prompts} prompts)")
        return

    model, _, _ = jspace.load_model(args.model, device=args.device, dtype=args.dtype)
    prompts = jspace.corpus(args.n_prompts, args.prompts)[i::n]
    layers = list(range(0, model.n_layers - 1, args.layer_stride))
    print(f"{model} | {len(prompts)} prompts | {len(layers)} layers | shard {i}/{n}")

    lens = jlens.fit(
        model,
        prompts,
        source_layers=layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        # Fingerprint the corpus into the name: resuming is only ever correct
        # against the same prompt list.
        checkpoint_path=str(out / f"ckpt.{_fingerprint(prompts)}.shard{i}.pt"),
        checkpoint_every=5,
    )
    name = "lens.pt" if n == 1 else f"lens.shard{i}.pt"
    lens.save(str(out / name))
    # Model-level, not tag-level: the Gram depends only on W_U.
    torch.save(jspace.unembed_gram(model), jspace.model_dir(args.model) / "gram.pt")
    print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
