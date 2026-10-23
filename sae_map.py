"""The J-lens in an SAE feature basis -> runs/<model>/<tag>/sae.json.

    uv run sae_map.py --model openai-community/gpt2

For a source layer l and the lens' target layer, compose the transport with the
SAE frames at each end:

    M_l = W_enc,targetᵀ · J_l · W_dec,lᵀ            [F_target, F_l]

    M[j,i] = how much one unit of feature i at layer l moves the
             pre-activation of feature j at the target layer

That is the paper's readout with the SAE dictionary in place of W_U. The
decoder is affine so its derivative is exactly W_dec; the encoder's is
diag(gate)·W_encᵀ, and the gate is dropped here — M is an effect on
*pre-activations*. J_l is already an expectation over a corpus, and folding a
second corpus-average (E[diag(gate)]) into it makes the object harder to read.

Two facts make this cheap, both checked in test_sae_map.py:

  rank(M) <= d_model. M factors through J, so lifting 768 -> 24576 adds no
  directions. The dictionary is an overcomplete frame, not a bigger space; M's
  spectrum measures J *in the SAE geometry*.

  The spectrum needs no F x F matrix. With thin QRs W_encᵀ = Q_E R_E and
  (J W_decᵀ)ᵀ = Q_X R_X, the orthonormal factors drop out of the SVD:
  svdvals(M) = svdvals(R_E R_Xᵀ), a d x d problem. The singular vectors come
  back as U = Q_E U_core, V = Q_X V_core — still in feature space.
"""

from __future__ import annotations

import argparse
import json

import torch

import jspace

#: How a model's lens layers map onto a published SAE suite. `source` is
#: formatted with the jlens source-layer index; jlens layer l is the *output* of
#: block l, which is the input ("resid_pre") of block l+1.
#: Gemma Scope ships one SAE per (layer, width, sparsity); these are the width-16k
#: L0 values nearest 100, layer 0..25. Regenerate by listing the repo.
_GEMMA_L0 = [105, 102, 141, 59, 124, 68, 70, 69, 71, 73, 77, 80, 82, 84, 84,
             78, 78, 77, 74, 73, 71, 70, 72, 75, 73, 116]

SUITES = {
    "openai-community/gpt2": {
        "repo": "jbloom/GPT2-Small-SAEs-Reformatted",
        "source": "blocks.{}.hook_resid_pre/sae_weights.safetensors",
        "offset": 1,
        "target": "blocks.11.hook_resid_post/sae_weights.safetensors",
    },
    # Gemma Scope's `res` SAEs sit on the residual stream at each layer's
    # *output*, which is what jlens records — so offset 0. Verified with --check
    # at layers 3/12/20: FVU 0.16/0.28/0.23 against 0.44/0.55/0.53 for offset 1.
    # google/gemma-2-2b is gated; the SAEs themselves are not.
    "google/gemma-2-2b": {
        "repo": "google/gemma-scope-2b-pt-res",
        "source": "layer_{}/width_16k/average_l0_%d/params.npz",
        "offset": 0,
        "target": f"layer_25/width_16k/average_l0_{_GEMMA_L0[25]}/params.npz",
        "l0": _GEMMA_L0,
    },
}


def sae_path(suite: dict, layer: int) -> str:
    """Source-SAE path for a jlens source layer."""
    idx = layer + suite["offset"]
    path = suite["source"].format(idx)
    return path % suite["l0"][idx] if "l0" in suite else path


def load_sae_full(repo: str, path: str, device: str) -> dict[str, torch.Tensor]:
    """All of an SAE's parameters, from safetensors or npz."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo, path)
    if path.endswith(".npz"):
        import numpy as np

        z = np.load(local)
        w = {k: torch.from_numpy(z[k]) for k in z.files}
    else:
        from safetensors.torch import load_file

        w = load_file(local)
    return {k: v.to(device, torch.float32) for k, v in w.items()}


def load_sae(repo: str, path: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (W_enc [d, F], W_dec [F, d]) for one SAE."""
    w = load_sae_full(repo, path, device)
    return w["W_enc"], w["W_dec"]


def reconstruct(w: dict[str, torch.Tensor], h: torch.Tensor) -> torch.Tensor:
    """SAE round-trip. JumpReLU when the SAE ships a threshold, else ReLU."""
    pre = h @ w["W_enc"] + w["b_enc"]
    f = pre * (pre > w["threshold"]) if "threshold" in w else pre.relu()
    return f @ w["W_dec"] + w["b_dec"]


#: Long enough that dropping the leading sink positions still leaves plenty.
_CHECK_TEXT = (
    "The capital of France is Paris, and the city is known for its museums, its "
    "architecture, and the river that runs through it. Visitors arrive all year "
    "round, though the summer months are by far the busiest, and the queues "
    "outside the larger galleries can stretch around the block by mid-morning. "
    "Away from the centre the streets are quieter, and the cafes there serve the "
    "same coffee for half the price."
)


def check_hook(model, suite: dict, layer: int, text: str) -> None:
    """Which SAE index lines up with the residual jlens records at `layer`?

    jlens records the *output* of block l. A suite indexed on block inputs needs
    offset 1, one indexed on block outputs offset 0 — and getting it wrong
    silently corrupts every number downstream. Reconstruction FVU settles it:
    the right index reconstructs, the wrong one does not.
    """
    from jlens.fitting import SKIP_FIRST_N_POSITIONS
    from jlens.hooks import ActivationRecorder

    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(model.encode(text, max_length=256))
        h = rec.activations[layer][0].float()

    # Drop the leading positions before scoring. BOS is an attention sink whose
    # residual norm runs an order of magnitude above everything else — on
    # gemma-2-2b, 1946 against a median of 147 — and no SAE reconstructs it. Left
    # in, it dominates both the error and the variance and reports FVU ~8 for a
    # correctly-matched SAE. This is the same reason jlens skips them when fitting.
    h = h[SKIP_FIRST_N_POSITIONS:]
    var = (h - h.mean(0)).pow(2).sum()
    print(f"residual at jlens layer {layer}: {tuple(h.shape)} "
          f"(first {SKIP_FIRST_N_POSITIONS} positions dropped)")
    scored = {}
    for off in (0, 1):
        probe = dict(suite, offset=off)
        try:
            w = load_sae_full(suite["repo"], sae_path(probe, layer), h.device)
        except Exception as exc:
            print(f"  offset {off}: unavailable ({type(exc).__name__})")
            continue
        scored[off] = (((h - reconstruct(w, h)).pow(2).sum() / var).item(),
                       sae_path(probe, layer))
    # Mark the winner, not everything under some cutoff — a mismatched SAE still
    # reconstructs partially (neighbouring layers are similar), so an absolute
    # threshold flags both.
    best = min(scored, key=lambda o: scored[o][0]) if scored else None
    for off, (fvu, path) in scored.items():
        print(f"  offset {off} -> {path}\n"
              f"      FVU = {fvu:.4f}   {'<-- best' if off == best else ''}")


def composed_spectrum(J: torch.Tensor, W_enc_t: torch.Tensor, W_dec_s: torch.Tensor):
    """Singular values and feature-space singular vectors of W_enc_tᵀ J W_dec_sᵀ.

    Never materialises the [F, F] matrix — see the module docstring.
    """
    Q_E, R_E = torch.linalg.qr(W_enc_t.T)          # [F, d], [d, d]
    Q_X, R_X = torch.linalg.qr(W_dec_s @ J.T)      # (J W_dec_sᵀ)ᵀ = W_dec_s Jᵀ
    U_c, S, Vh_c = torch.linalg.svd(R_E @ R_X.T)
    return S, Q_E @ U_c, Q_X @ Vh_c.T              # U: target features, V: source


def gloss(W_dec: torch.Tensor, feat: int, model, top: int) -> list[str]:
    """Name a feature by pushing its decoder direction through the unembedding —
    the SAE suites ship no labels, and this needs nothing but the model."""
    logits = model.unembed(W_dec[feat].unsqueeze(0).to(model.input_device))[0].float()
    return [model.tokenizer.decode([t]) for t in logits.topk(top).indices.tolist()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="base")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--modes", type=int, default=3, help="singular modes glossed per layer")
    p.add_argument("--feats", type=int, default=5, help="features named per mode side")
    p.add_argument("--top-tokens", type=int, default=6)
    p.add_argument("--no-gloss", action="store_true", help="skip token naming (no model load)")
    p.add_argument("--check", type=int, metavar="LAYER",
                   help="verify which SAE index matches this lens layer, then exit")
    args = p.parse_args()

    from jlens import JacobianLens

    suite = SUITES.get(args.model)
    if suite is None:
        raise SystemExit(f"no SAE suite registered for {args.model}; add one to SUITES")

    if args.check is not None:
        model, _, _ = jspace.load_model(args.model, device=args.device, dtype=args.dtype)
        check_hook(model, suite, args.check, _CHECK_TEXT)
        return

    out = jspace.run_dir(args.model, args.tag)
    lens = JacobianLens.load(str(out / "lens.pt"))
    layers = lens.source_layers
    dev = args.device if torch.cuda.is_available() else "cpu"

    W_enc_t, W_dec_t = load_sae(suite["repo"], suite["target"], dev)
    d, F = W_enc_t.shape
    print(f"target SAE: d={d} F={F} ({F // d}x)")

    model = None
    if not args.no_gloss:
        model, _, _ = jspace.load_model(args.model, device=args.device, dtype=args.dtype)

    per_layer = []
    for l in layers:
        W_enc_s, W_dec_s = load_sae(suite["repo"], sae_path(suite, l), dev)
        J = lens.jacobians[l].to(dev, torch.float32)
        S, U, V = composed_spectrum(J, W_enc_t, W_dec_s)
        energy = (S**2).sum()

        row = {
            "layer": l,
            "d_model": d,
            "n_features": F,
            "fro": (S**2).sum().sqrt().item(),
            "eff_rank": (S.sum() ** 2 / energy).item(),
            "stable_rank": (energy / S[0] ** 2).item(),
            "sv": (S / S[0]).cpu().tolist(),
        }
        if model is not None:
            row["modes"] = []
            for k in range(args.modes):
                src = V[:, k].abs().topk(args.feats).indices.tolist()
                tgt = U[:, k].abs().topk(args.feats).indices.tolist()
                row["modes"].append({
                    "sigma": (S[k] / S[0]).item(),
                    "share": (S[k] ** 2 / energy).item(),
                    "source": [{"f": i, "tokens": gloss(W_dec_s, i, model, args.top_tokens)}
                               for i in src],
                    "target": [{"f": j, "tokens": gloss(W_dec_t, j, model, args.top_tokens)}
                               for j in tgt],
                })
        per_layer.append(row)
        print(f"L{l:3d}  eff_rank={row['eff_rank']:7.1f}  stable_rank={row['stable_rank']:6.1f}"
              f"  (max possible {d})")
        del W_enc_s, W_dec_s, J, S, U, V

    (out / "sae.json").write_text(json.dumps({
        "model": args.model,
        "repo": suite["repo"],
        "d_model": d,
        "n_features": F,
        "layers": layers,
        "per_layer": per_layer,
    }))
    print(f"wrote {out / 'sae.json'}")


if __name__ == "__main__":
    main()
