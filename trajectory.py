"""Sample completions at temperature T and record s_t = <u_l, h_t> at every
generated token, for every layer -> runs/<model>/traj/<name>.pt

    uv run trajectory.py --model Qwen/Qwen3-0.6B --axes all-advice --method diffmean \\
        --n-prompts 128 --samples 16 --temperature 1.0 --max-new-tokens 160

The stochastic process of Part B1, Eq. (2): the last-token residual at
generation step t, projected on a safety direction, under the model's own
sampling at temperature T (pure temperature: top-k/top-p are switched off,
otherwise the chain is not the one the theory describes).

Prompts are the unique requests of the contrast set, rendered the way the
axes were built (plain or chat template), so s* from axes.py refers to the
same distribution. s[n, t, l] is the projection at the context position that
*produced* generated token t; t = 0 is the end of the prompt. Positions after
a walker's EOS are NaN.
"""

from __future__ import annotations

import argparse
import time

import torch

import jspace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--axes", default="all-advice", help="basename under runs/<model>/axes/")
    p.add_argument("--method", default="diffmean", help="diffmean | pullback | contrast")
    p.add_argument("--data", default="data/contrast/all-advice.jsonl", help="prompts = its unique requests")
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--samples", type=int, default=16, help="completions per prompt")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--batch", type=int, default=16, help="prompts per generate call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", help="output basename (default <axes>-<method>-T<temperature>)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    U, stats, d_prime, template = jspace.load_axes(args.model, args.axes, args.method)
    name = args.name or f"{args.axes}-{args.method}-T{args.temperature}"
    out = jspace.model_dir(args.model) / "traj"
    out.mkdir(exist_ok=True)

    model, hf, tok = jspace.load_model(args.model, device=args.device, dtype=args.dtype)
    prompts = jspace.prompts(jspace.load_jsonl(args.data), tok, template)[: args.n_prompts]
    eos, pad = jspace.generation_setup(hf, tok)
    L, M, n_layers = args.max_new_tokens, args.samples, model.n_layers
    proj = jspace.Projector(model, U, list(range(n_layers)))
    torch.manual_seed(args.seed)

    S, T, P = [], [], []
    t0 = time.perf_counter()
    for b in range(0, len(prompts), args.batch):
        batch = prompts[b : b + args.batch]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.input_device)
        proj.clear()
        with torch.no_grad():
            seqs = hf.generate(**enc, num_return_sequences=M,
                               **jspace.sampling(args.temperature, L, pad)).sequences
        toks = seqs[:, enc.input_ids.shape[1]:].cpu()  # [B*M, L'] generated, L' <= L
        s = proj.stack()  # [B*M, L', n_layers]
        # Token g was sampled from the residual recorded at call g; after the
        # first EOS the inputs are padding and the residuals meaningless.
        after = torch.arange(toks.shape[1])[None, :] > jspace.first_eos(toks, eos)[:, None]
        s[after], toks[after] = float("nan"), -1
        pad_to = L - toks.shape[1]
        S.append(torch.nn.functional.pad(s, (0, 0, 0, pad_to), value=float("nan")).half())
        T.append(torch.nn.functional.pad(toks, (0, pad_to), value=-1).int())
        P.append(torch.arange(b, b + len(batch)).repeat_interleave(M))
        print(f"  {b + len(batch)}/{len(prompts)} prompts  {after.any(1).float().mean():.2f} hit EOS  "
              f"{time.perf_counter() - t0:.0f}s")
    proj.remove()

    torch.save(
        {"model": args.model, "axes": args.axes, "method": args.method, "template": template,
         "temperature": args.temperature, "seed": args.seed, "layers": list(range(n_layers)),
         "prompts": prompts, "prompt_idx": torch.cat(P), "s": torch.cat(S), "tokens": torch.cat(T),
         "stats": stats, "d_prime": d_prime},
        out / f"{name}.pt",
    )
    print(f"wrote {out / name}.pt  s{tuple(torch.cat(S).shape)}")


if __name__ == "__main__":
    main()
