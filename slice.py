"""Apply a fitted lens to a prompt: interactive layer x position page + JSON.

    uv run slice.py --model Qwen/Qwen3-0.6B --example multihop
    uv run slice.py --model Qwen/Qwen3-0.6B --prompt "The capital of France is"

Writes runs/<model>/slices/<name>.html (self-contained, openable in a browser)
and .json with the top-k lens tokens per (layer, position) for artifacts.

`--example list` shows the paper's bundled prompts.
"""

from __future__ import annotations

import argparse
import json

import jspace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="base", help="which fitted lens (subdir of runs/<model>/)")
    p.add_argument("--example", help="slug from jlens.examples, or 'list'")
    p.add_argument("--prompt")
    p.add_argument("--name", help="output basename (default: example slug / 'prompt')")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--layer-stride", type=int, default=1)
    args = p.parse_args()

    from jlens import JacobianLens, examples, vis

    if args.example == "list":
        for e in examples.EXAMPLES:
            print(f"{e.slug:24s} {e.section}")
        return
    if not (args.example or args.prompt):
        raise SystemExit("pass --prompt or --example (--example list to see them)")

    out = jspace.run_dir(args.model, args.tag)
    lens = JacobianLens.load(str(out / "lens.pt"))
    model, _, tok = jspace.load_model(args.model, device=args.device, dtype=args.dtype)

    if args.example:
        ex = next(e for e in examples.EXAMPLES if e.slug == args.example)
        prompt, title, desc = examples.resolve_prompt(ex, tok), ex.section, ex.description
    else:
        prompt, title, desc = args.prompt, "Custom prompt", ""
    name = args.name or (args.example or "prompt")

    data = vis.compute_slice(
        model, lens, prompt, top_n=args.top_n, layer_stride=args.layer_stride
    )
    html, _, _ = vis.build_page(data, prompt, title=title, description=desc)

    slices = out / "slices"
    slices.mkdir(exist_ok=True)
    (slices / f"{name}.html").write_text(html, encoding="utf-8")

    # Compact readout for artifacts: top-k tokens per (layer, position).
    (slices / f"{name}.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt": prompt,
                "layers": data.layers,
                "tokens": data.context_token_strs[data.ctx_offset :],
                "top": [
                    [[tok.decode([t]) for t in cell] for cell in row]
                    for row in data.top_ids.tolist()
                ],
            }
        ),
        encoding="utf-8",
    )
    print(f"wrote {slices / f'{name}.html'}")

    # Sanity check: what the lens reads out at the final position, by layer.
    last = -1
    for li, layer in enumerate(data.layers):
        top5 = [tok.decode([t]) for t in data.top_ids[last, li, :5]]
        print(f"L{layer:3d}  {top5}")


if __name__ == "__main__":
    main()
