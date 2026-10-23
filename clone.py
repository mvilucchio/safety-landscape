"""Rare-event sampling by cloning (GKTL; Part B2, Eqs. 7-9)
-> runs/<model>/clone/<name>.json

    uv run clone.py --model Qwen/Qwen3-0.6B --axes all-advice --method diffmean \\
        --prompt 3 --walkers 64 --blocks 8 --block-tokens 16 --k 0 0.5 1.0 --repeats 4

Giardinà-Kurchan-Tailleur-Lecomte cloning as Ragone, Wouters & Bouchet ran it
for heat waves. N walkers (completions of one prompt) advance block by block;
after each block, walker n is cloned in proportion to exp(k A_n,t) / R_t, with
A_t its score on the block and R_t the population mean of exp(k A). The final
population is tilted by exp(k S_L), S_L = sum_t A_t, and the tilt is undone
exactly:

    P[S_L >= a L] = E_k[ 1{S_L >= a L} exp(-k S_L) ] * prod_t R_t          (8)

k = 0 is brute force with the same code, so both sit on one budget. The score
is the open-weight one of the proposal: A_t = mean over the block of
s_t = <u_l, h_t>, a time average like a heat wave; the event is the block mean
over the whole response reaching a (default: the class midpoint s* of
axes.py). The running max of s_t is kept too, so P[max_t s_t >= s*] comes out
of the same population. Judge-based scores and the committor probe are not
implemented.

The cloning step (`gktl`, `estimate`) is model-free; test_clone.py checks it
against a Gaussian walk whose tail probability is known in closed form.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import jspace


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """N children indices with E[count_i] = N w_i / sum(w), low variance."""
    N = len(w)
    c = np.cumsum(w / w.sum())
    u = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(c, u), N - 1)


def gktl(states: list, step, *, k: float, n_blocks: int, rng: np.random.Generator) -> dict:
    """Run the cloning dynamics.

    `step(states) -> (new_states, A)` advances every walker one block and
    returns its score for that block (array [N]). States are never mutated
    here — clones share the object, so `step` must return fresh ones.
    Returns the final (resampled) population: S = S_L per walker, logR =
    sum_t log R_t, A = [N, L] block scores along each survivor's ancestry,
    distinct = fraction of parents kept at each resampling.

    Choosing k: the event should be *typical* under the tilt — aim for a raw
    fraction near 1/2 and a parent fraction that stays well above 1/N. Sweep
    k from 0 (brute force) upward; a jump of the raw fraction to 1.0 with the
    estimate scattering over decades means k overshot.
    """
    N = len(states)
    S, logR, hist, distinct = np.zeros(N), 0.0, np.zeros((N, 0)), []
    for _ in range(n_blocks):
        states, A = step(states)
        A = np.asarray(A, dtype=float)
        S = S + A
        hist = np.concatenate([hist, A[:, None]], 1)
        shift = k * A.max()  # R_t = mean exp(kA), computed stably
        w = np.exp(k * A - shift)
        logR += shift + np.log(w.mean())
        idx = systematic_resample(w, rng) if k else np.arange(N)
        # Collapse diagnostic: how many parents the next generation descends
        # from. Near 1/N the tilt is too strong and the estimate is one lineage.
        distinct.append(len(set(idx.tolist())) / N)
        states, S, hist = [states[i] for i in idx], S[idx], hist[idx]
    return {"S": S, "logR": logR, "states": states, "A": hist, "distinct": distinct}


def estimate(res: dict, event: np.ndarray, k: float) -> float:
    """Eq. (8): unbiased P[event] from a run tilted by k."""
    return float(np.mean(event * np.exp(-k * res["S"] + res["logR"])))


def llm_step(hf, tok, model, u: torch.Tensor, layer: int, *, block_tokens: int,
             temperature: float, eos: list[int]):
    """A block of generation for every walker. State = dict(ids, smax, done, last).

    Finished walkers (EOS seen) are frozen: their block score repeats the last
    one they earned. ponytail: their ids still get fed to generate (wasted but
    harmless); mask them out if block count × walkers ever hurts.
    """
    rec: list[torch.Tensor] = []
    pad = tok.pad_token_id

    def hook(module, inputs, output):
        h = output if torch.is_tensor(output) else output[0]
        rec.append((h[:, -1, :].float() @ u).cpu())

    handle = model.layers[layer].register_forward_hook(hook)

    def step(states):
        ids = torch.stack([s["ids"] for s in states]).to(model.input_device)
        rec.clear()
        with torch.no_grad():
            gen = hf.generate(
                input_ids=ids, attention_mask=(ids != pad).long(), do_sample=True,
                temperature=temperature, top_k=0, top_p=1.0, max_new_tokens=block_tokens,
                pad_token_id=pad, return_dict_in_generate=True,
            )
        new_ids = gen.sequences.cpu()
        toks = new_ids[:, ids.shape[1]:]
        s = torch.stack(rec, 1).numpy()  # [N, L'] projection that produced each token
        is_eos = torch.isin(toks, torch.tensor(eos)).numpy()
        out, A = [], np.zeros(len(states))
        for n, st in enumerate(states):
            if st["done"]:
                out.append({**st, "ids": new_ids[n]})
                A[n] = st["last"]
                continue
            e = int(is_eos[n].argmax()) if is_eos[n].any() else toks.shape[1] - 1
            vals = s[n, : e + 1]
            A[n] = vals.mean()
            out.append({"ids": new_ids[n], "smax": max(st["smax"], float(vals.max())),
                        "done": bool(is_eos[n].any()), "last": float(A[n])})
        return out, A

    step.remove = handle.remove
    return step


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--axes", default="all-advice")
    p.add_argument("--method", default="diffmean")
    p.add_argument("--layer", type=int, help="default: best held-out d' of the axis")
    p.add_argument("--data", default="data/contrast/all-advice.jsonl")
    p.add_argument("--prompt", type=int, default=0, help="index into the unique requests")
    p.add_argument("--walkers", type=int, default=64)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--block-tokens", type=int, default=16)
    p.add_argument("--k", type=float, nargs="+", default=[0.0, 0.5], help="tilt strengths; 0 = brute force")
    p.add_argument("--a", type=float, help="event: block-mean score >= a (default: class midpoint s*)")
    p.add_argument("--repeats", type=int, default=4, help="independent runs per k (spread = error)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    axes_dir = jspace.model_dir(args.model) / "axes"
    axes = torch.load(axes_dir / f"{args.axes}.pt", weights_only=True)
    sep = json.loads((axes_dir / f"{args.axes}.json").read_text())["separation"][args.method]
    layer = args.layer if args.layer is not None else int(np.argmax([pl["d_prime"] for pl in sep]))
    st = axes["stats"][args.method]
    s_star = (st["mu_safe"][layer] + st["mu_harm"][layer]) / 2
    a = args.a if args.a is not None else s_star
    rows = jspace.load_jsonl(args.data)
    prompt = list(dict.fromkeys(r["prefix"].rstrip() for r in rows))[args.prompt]

    model, hf, tok = jspace.load_model(args.model, device=args.device, dtype=args.dtype)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    e = hf.generation_config.eos_token_id
    eos = sorted({*([e] if isinstance(e, int) else list(e or [])), tok.eos_token_id})
    u = axes["u"][args.method][layer].to(model.input_device)
    ids0 = model.encode(prompt)[0].cpu()
    step = llm_step(hf, tok, model, u, layer, block_tokens=args.block_tokens,
                    temperature=args.temperature, eos=eos)
    print(f"{model} | layer {layer} d'={sep[layer]['d_prime']:.2f} | s*={s_star:.2f} a={a:.2f} | "
          f"N={args.walkers} L={args.blocks}x{args.block_tokens} tokens | prompt {args.prompt}")

    runs, samples = [], []
    for k in args.k:
        for r in range(args.repeats):
            rng = np.random.default_rng(args.seed * 1000 + r)
            torch.manual_seed(args.seed * 1000 + r + int(1e6 * k))
            t0 = time.perf_counter()
            init = [{"ids": ids0, "smax": -np.inf, "done": False, "last": 0.0} for _ in range(args.walkers)]
            res = gktl(init, step, k=k, n_blocks=args.blocks, rng=rng)
            smax = np.array([s["smax"] for s in res["states"]])
            runs.append({
                "k": k, "repeat": r,
                "p_mean": estimate(res, res["S"] >= a * args.blocks, k),
                "p_max": estimate(res, smax >= s_star, k),
                "frac_mean": float((res["S"] >= a * args.blocks).mean()),
                "frac_max": float((smax >= s_star).mean()),
                "S_mean": float(res["S"].mean() / args.blocks),
                "eos": float(np.mean([s["done"] for s in res["states"]])),
                "min_parent_frac": float(min(res["distinct"])),
                "seconds": time.perf_counter() - t0,
            })
            top = int(np.argmax(res["S"]))
            samples.append({"k": k, "repeat": r, "S_over_L": float(res["S"][top] / args.blocks),
                            "text": tok.decode(res["states"][top]["ids"][len(ids0):].tolist(),
                                               skip_special_tokens=True)})
            print(f"  k={k:<4} run {r}: p_mean={runs[-1]['p_mean']:.3e} p_max={runs[-1]['p_max']:.3e} "
                  f"(raw {runs[-1]['frac_mean']:.2f}/{runs[-1]['frac_max']:.2f})  "
                  f"<S/L>={runs[-1]['S_mean']:.2f}  parents>={runs[-1]['min_parent_frac']:.2f}  "
                  f"eos={runs[-1]['eos']:.2f}  {runs[-1]['seconds']:.0f}s")
    step.remove()

    summary = {}
    for k in args.k:
        pm = np.array([r["p_mean"] for r in runs if r["k"] == k])
        px = np.array([r["p_max"] for r in runs if r["k"] == k])
        summary[str(k)] = {"p_mean": pm.mean(), "p_mean_std": pm.std(ddof=1) if len(pm) > 1 else None,
                           "p_max": px.mean(), "p_max_std": px.std(ddof=1) if len(px) > 1 else None}
        # p_max stays in the JSON only: it shares the time-average tilt, which is
        # the wrong tilt for a max event, so it is only readable when k is small.
        print(f"k={k}: P[mean >= a] = {pm.mean():.3e} ± {summary[str(k)]['p_mean_std']}")

    out = jspace.model_dir(args.model) / "clone"
    out.mkdir(exist_ok=True)
    name = args.name or f"{args.axes}-{args.method}-L{layer}-p{args.prompt}-T{args.temperature}"
    (out / f"{name}.json").write_text(json.dumps({
        "model": args.model, "axes": args.axes, "method": args.method, "layer": layer,
        "s_star": s_star, "a": a, "prompt": prompt, "temperature": args.temperature,
        "walkers": args.walkers, "blocks": args.blocks, "block_tokens": args.block_tokens,
        "completions_per_run": args.walkers * args.blocks * args.block_tokens,
        "runs": runs, "summary": summary, "top_samples": samples,
    }))
    print(f"wrote {out / name}.json")


if __name__ == "__main__":
    main()
