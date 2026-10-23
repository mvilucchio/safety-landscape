"""Sample completions at temperature T and record s_t = <u_l, h_t> at every
generated token, for every layer -> runs/<model>/traj/<name>.pt

    uv run trajectory.py --model Qwen/Qwen3-0.6B --axes all-advice --method diffmean \\
        --n-prompts 128 --samples 16 --temperature 1.0 --max-new-tokens 160

This is the stochastic process of Part B2, Eq. (2) of Part I: the last-token
residual at generation step t, projected on a safety direction, under the
model's own sampling at temperature T (pure temperature — top-k/top-p are
switched off, otherwise the chain is not the one the theory describes).

Prompts are the unique requests of the contrast set (the `prefix` field),
so s* from axes.py refers to the same distribution. s[n, t, l] is the
projection at the context position that *produced* generated token t;
t = 0 is the end of the prompt. Positions after a walker's EOS are NaN.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import jspace


def eos_ids(hf, tok) -> list[int]:
    e = hf.generation_config.eos_token_id
    ids = [e] if isinstance(e, int) else list(e or [])
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return sorted(set(ids))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--axes", default="all-advice", help="basename under runs/<model>/axes/")
    p.add_argument("--method", default="diffmean", help="diffmean | pullback | contrast")
    p.add_argument("--data", default="data/contrast/all-advice.jsonl", help="prompts = its unique prefixes")
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

    axes_dir = jspace.model_dir(args.model) / "axes"
    axes = torch.load(axes_dir / f"{args.axes}.pt", weights_only=True)
    sep = json.loads((axes_dir / f"{args.axes}.json").read_text())["separation"][args.method]
    name = args.name or f"{args.axes}-{args.method}-T{args.temperature}"
    out = jspace.model_dir(args.model) / "traj"
    out.mkdir(exist_ok=True)

    # ponytail: the data's "Assistant: " prefix ends in a space; stripped so the
    # model's first token carries its own leading space as in training.
    rows = jspace.load_jsonl(args.data)
    prompts = list(dict.fromkeys(r["prefix"].rstrip() for r in rows))[: args.n_prompts]

    model, hf, tok = jspace.load_model(args.model, device=args.device, dtype=args.dtype)
    U = axes["u"][args.method].to(model.input_device)  # [n_layers, d]
    n_layers, L, M = model.n_layers, args.max_new_tokens, args.samples
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos = eos_ids(hf, tok)
    torch.manual_seed(args.seed)

    # One forward hook per block: project the last position and keep the scalar.
    rec: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}

    def make_hook(l):
        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            rec[l].append((h[:, -1, :].float() @ U[l]).cpu())
        return hook

    handles = [blk.register_forward_hook(make_hook(l)) for l, blk in enumerate(model.layers)]

    S, T, P = [], [], []
    t0 = time.perf_counter()
    for b in range(0, len(prompts), args.batch):
        batch = prompts[b : b + args.batch]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.input_device)
        for l in rec:
            rec[l].clear()
        with torch.no_grad():
            gen = hf.generate(
                **enc, do_sample=True, temperature=args.temperature, top_k=0, top_p=1.0,
                max_new_tokens=L, num_return_sequences=M, pad_token_id=tok.pad_token_id,
                return_dict_in_generate=True,
            )
        toks = gen.sequences[:, enc.input_ids.shape[1]:].cpu()  # [B*M, L'] generated
        s = torch.stack([torch.stack(rec[l], 1) for l in range(n_layers)], -1)  # [B*M, L', n_layers]
        n, Lp = toks.shape
        assert s.shape[1] == Lp, (s.shape, toks.shape)
        # Token g was sampled from the residual recorded at call g; after the
        # first EOS the inputs are padding and the residuals meaningless.
        is_eos = torch.isin(toks, torch.tensor(eos))
        first = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((n,), Lp))
        t_idx = torch.arange(Lp)[None, :]
        s[t_idx > first[:, None]] = float("nan")
        toks[t_idx > first[:, None]] = -1
        pad = L - Lp
        S.append(torch.nn.functional.pad(s, (0, 0, 0, pad), value=float("nan")).half())
        T.append(torch.nn.functional.pad(toks, (0, pad), value=-1).int())
        P.append(torch.arange(b, b + len(batch)).repeat_interleave(M))
        done = b + len(batch)
        print(f"  {done}/{len(prompts)} prompts  {done * M} samples  "
              f"{(first < Lp).float().mean():.2f} hit EOS  {time.perf_counter() - t0:.0f}s")
    for h in handles:
        h.remove()

    torch.save(
        {"model": args.model, "axes": args.axes, "method": args.method,
         "temperature": args.temperature, "seed": args.seed,
         "layers": list(range(n_layers)), "prompts": prompts,
         "prompt_idx": torch.cat(P), "s": torch.cat(S), "tokens": torch.cat(T),
         "stats": axes["stats"][args.method],
         "d_prime": [pl["d_prime"] for pl in sep]},
        out / f"{name}.pt",
    )
    print(f"wrote {out / name}.pt  s{tuple(torch.cat(S).shape)}")


if __name__ == "__main__":
    main()
