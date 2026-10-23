"""Bundle every lens fitted so far into ONE self-contained HTML report.

    uv run report.py --models Qwen/Qwen3-0.6B openai-community/gpt2 \
                     --tags base contrast

Reads runs/<model>/<tag>/{analysis,spectrum,split_half,sae,eval-*}.json for
every model x tag that exists and inlines them into report.tmpl.html. The page
gets a model switcher; within a model, lenses sit side by side. Missing
combinations are skipped with a warning. No network, no CDN — publishable as a
Claude artifact as-is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jspace

TEMPLATE = Path(__file__).parent / "report.tmpl.html"


def r(x, nd=4):
    """Round floats (incl. nested) — the page never needs full precision, and
    it roughly halves the inlined payload."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [r(v, nd) for v in x]
    return x


def log_grid(n: int, samples: int = 150) -> list[int]:
    """Log-spaced indices over [0, n). The spectrum spans d_model values but its
    shape is all in the first decade or two, so a linear sample would spend most
    of its columns on a flat tail. Sampling log-spaced shows the whole spectrum
    at a readable width; the axis is labelled with real indices."""
    if n <= samples:
        return list(range(n))
    out, prev = [], -1
    for k in range(samples):
        i = round((n - 1) ** (k / (samples - 1))) - 1 if k else 0
        i = min(n - 1, max(0, i))
        if i > prev:
            out.append(i)
            prev = i
    return out


def load_lens(out: Path, tag: str) -> dict | None:
    analysis = out / "analysis.json"
    if not analysis.exists():
        return None
    a = json.loads(analysis.read_text())
    grid = log_grid(len(a["per_layer"][0]["sv"]))
    lens = {
        "tag": tag,
        "n_prompts": a["n_prompts"],
        "layers": a["layers"],
        "profile": {
            key: r([pl[key] for pl in a["per_layer"]])
            for key in ("fro", "identity_cos", "eff_rank", "stable_rank")
        },
        # Every layer, whole spectrum, log-sampled down to a drawable width.
        "sv_index": grid,
        "sv": r([[pl["sv"][i] for i in grid] for pl in a["per_layer"]], 5),
        # Histogram of the full spectrum + the leading values kept exactly, so
        # the page can draw the bulk as a distribution and the outliers as spikes.
        "sv_log_range": a.get("sv_log_range", [-4, 0]),
        "sv_hist": [pl.get("sv_hist", []) for pl in a["per_layer"]],
        "sv_top": r([pl["sv"][:24] for pl in a["per_layer"]], 5),
        "eig": [pl.get("eig", []) for pl in a["per_layer"]],
        "eig_absmax": r([pl.get("eig_absmax", 0) for pl in a["per_layer"]], 4),
        "cka": r(a.get("cka", []), 3),
        "evals": {
            f.stem.replace("eval-", ""): {
                k: r(v, 3) if isinstance(v, list) else v
                for k, v in json.loads(f.read_text()).items()
                if k in ("pass_at_k", "per_layer_frac", "n_targets", "n_items",
                         "median_rank_per_layer")
            }
            for f in sorted(out.glob("eval-*.json"))
        },
    }

    # A contrast lens carries extra structure the base lens has no analogue for:
    # signed singular modes, and a split-half noise floor for them.
    # The same transport read out in an SAE feature basis instead of through W_U.
    sae = out / "sae.json"
    if sae.exists():
        s = json.loads(sae.read_text())
        lens["sae"] = {
            "repo": s["repo"],
            "n_features": s["n_features"],
            "layers": s["layers"],
            "eff_rank": r([pl["eff_rank"] for pl in s["per_layer"]], 2),
            "stable_rank": r([pl["stable_rank"] for pl in s["per_layer"]], 3),
            # M's own spectrum stays in sae.json; the page only plots ranks, and
            # inlining d_model values per layer per lens was most of the payload.
            "modes": [pl.get("modes", [])[:1] for pl in s["per_layer"]],
        }

    spectrum = out / "spectrum.json"
    if spectrum.exists():
        s = json.loads(spectrum.read_text())
        split = out / "split_half.json"
        agree = json.loads(split.read_text()) if split.exists() else {}
        lens["contrast"] = {
            "rel_to_pos": r([pl.get("rel_to_pos", 0) for pl in s["per_layer"]]),
            "top4_energy": r([pl["top4_energy"] for pl in s["per_layer"]]),
            "agreement": r([agree.get(str(l), 0) for l in s["layers"]]),
            "modes": [
                {"layer": pl["layer"], "share": r(pl["modes"][0]["share"]),
                 "pos": pl["modes"][0]["pos"][:6], "neg": pl["modes"][0]["neg"][:6]}
                for pl in s["per_layer"]
            ],
        }
    return lens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True, help="in display order")
    p.add_argument("--tags", nargs="+", default=["base", "contrast"],
                   help="lenses to include per model, in display order")
    p.add_argument("--out", default="runs/report.html")
    args = p.parse_args()

    models = {}
    for model in args.models:
        lenses = {}
        for tag in args.tags:
            out = jspace.run_dir(model, tag)
            lens = load_lens(out, tag)
            if lens is None:
                print(f"  skipping {model} / {tag}: no analysis.json")
            else:
                lenses[tag] = lens
        if not lenses:
            continue
        a = json.loads(
            (jspace.run_dir(model, next(iter(lenses))) / "analysis.json").read_text()
        )
        models[model] = {"d_model": a["d_model"], "n_layers": a["n_layers"], "lenses": lenses}
        print(f"  {model}: {', '.join(lenses)}")
    if not models:
        raise SystemExit("nothing to report")

    data = {"models": models}
    html = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__MODEL__", " · ".join(models))  # <title> names the artifact
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
    )
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    print(f"wrote {dest}  ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
