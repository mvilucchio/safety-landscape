"""Layer-by-layer spectrum of a contrast lens -> runs/<model>/<tag>/spectrum.json.

    uv run spectrum.py --model Qwen/Qwen3-0.6B --tag contrast

`analyze.py` describes a J matrix as a transport map. A *difference* matrix is
not a transport map, so the questions change:

  where     ‖J_diff‖ / ‖J_pos‖ per layer — at which depth do the two classes
            actually diverge?
  how many  effective rank and top-k energy — is the difference a couple of
            directions, or diffuse?
  what      each top singular triple (σ_k, u_k, v_k): v_k is the layer-l
            residual direction whose transport differs most between classes,
            u_k the final-basis direction it lands on. Decoding u_k through the
            model's unembedding names the mode in tokens — both signs, since
            the SVD fixes the pair only up to a shared sign flip.
"""

from __future__ import annotations

import argparse
import json

import torch

import jspace


def decode(model, vec: torch.Tensor, top: int) -> list[str]:
    logits = model.unembed(vec.unsqueeze(0).to(model.input_device))[0].float()
    return [model.tokenizer.decode([t]) for t in logits.topk(top).indices.tolist()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="contrast")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--modes", type=int, default=4, help="singular directions decoded per layer")
    p.add_argument("--top-tokens", type=int, default=8)
    p.add_argument("--top-sv", type=int, default=64)
    args = p.parse_args()

    from jlens import JacobianLens

    out = jspace.run_dir(args.model, args.tag)
    lens = JacobianLens.load(str(out / "lens.pt"))
    layers = lens.source_layers
    dev = args.device if torch.cuda.is_available() else "cpu"

    ref_path = out / "lens.pos.pt"
    ref = JacobianLens.load(str(ref_path)) if ref_path.exists() else None

    model, _, _ = jspace.load_model(args.model, device=args.device, dtype=args.dtype)

    per_layer = []
    for l in layers:
        A = lens.jacobians[l].to(dev, torch.float32)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        energy = (S**2).sum()
        row = {
            "layer": l,
            "fro": (A.norm() / lens.d_model**0.5).item(),
            "eff_rank": (S.sum() ** 2 / energy).item(),
            "stable_rank": (energy / S[0] ** 2).item(),
            # How concentrated the contrast is: 1.0 would mean a single direction
            # carries the whole difference between the classes.
            "top4_energy": ((S[:4] ** 2).sum() / energy).item(),
            "sv": (S[: args.top_sv] / S[0]).cpu().tolist(),
            "modes": [
                {
                    "sigma": (S[k] / S[0]).item(),
                    # This mode's share of the whole difference — for k=0 it says
                    # how much of the class separation one direction carries.
                    "share": (S[k] ** 2 / energy).item(),
                    "pos": decode(model, U[:, k], args.top_tokens),
                    "neg": decode(model, -U[:, k], args.top_tokens),
                }
                for k in range(min(args.modes, len(S)))
            ],
        }
        if ref is not None:
            row["rel_to_pos"] = (A.norm() / ref.jacobians[l].to(dev).norm()).item()
        per_layer.append(row)
        print(f"L{l:3d}  fro={row['fro']:.3f}  eff_rank={row['eff_rank']:6.1f}  "
              f"top4={row['top4_energy']:.2f}"
              + (f"  ‖diff‖/‖pos‖={row['rel_to_pos']:.3f}" if ref is not None else ""))
        print(f"      +{row['modes'][0]['pos'][:6]}")
        print(f"      -{row['modes'][0]['neg'][:6]}")

    result = {
        "model": args.model,
        "tag": args.tag,
        "d_model": lens.d_model,
        "n_prompts": lens.n_prompts,
        "layers": layers,
        "per_layer": per_layer,
    }
    (out / "spectrum.json").write_text(json.dumps(result))
    print(f"wrote {out / 'spectrum.json'}")


if __name__ == "__main__":
    main()
