"""Build a labelled contrast set -> data/contrast/<name>.jsonl.

    uv run make_data.py --set medical-advice

One JSON object per line:

    {"text": <full prompt>, "label": +1|-1, "prefix": <shared part>, "pair": <id>}

`label` is the contrast sign: +1 is the class whose mean Jacobian is added,
-1 the one subtracted. `prefix` is the part of the text both members of a pair
share — contrast.py restricts the Jacobian average to the positions *after* it,
so the shared request cannot contribute to the difference.

Sources are matched pairs: same user request, one low-harm and one high-harm
expert-graded response. Unpaired classes would make the difference matrix
mostly a topic detector.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

OUT = Path("data/contrast")
# Plain-text roles keep the file model-agnostic; jspace.render() re-renders
# `request`/`response` through a tokenizer's chat template when asked.
TEMPLATE = "User: {request}\n\nAssistant: {response}"


def harmful_advice(category: str | None, safe_max: int, harm_min: int) -> list[dict]:
    """ai-safety-institute/harmful-advice-dataset, paired by user request.

    `harm_level` is the graded harm of the *response* (0 safe … 4 harmful).
    We keep requests that carry both a low- and a high-harm response, and emit
    the pair. label +1 = safe advice, -1 = harmful advice.
    """
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset("ai-safety-institute/harmful-advice-dataset")
    d = concatenate_datasets([ds["train"], ds["test"]])
    if category:
        d = d.filter(lambda r: r["category"] == category)

    by_request = collections.defaultdict(list)
    for row in d:
        by_request[row["user_request"]].append(row)

    out = []
    for pair_id, (request, rows) in enumerate(sorted(by_request.items())):
        safe = [r for r in rows if r["harm_level"] <= safe_max]
        harmful = [r for r in rows if r["harm_level"] >= harm_min]
        if not safe or not harmful:
            continue
        # Most extreme member of each side, so the contrast is as clean as the
        # grading allows.
        pick = [(min(safe, key=lambda r: r["harm_level"]), 1),
                (max(harmful, key=lambda r: r["harm_level"]), -1)]
        for row, label in pick:
            out.append({
                "text": TEMPLATE.format(request=request, response=row["response"]),
                "label": label,
                "prefix": TEMPLATE.format(request=request, response=""),
                # Raw parts, so a chat template can be applied at run time.
                "request": request,
                "response": row["response"],
                "pair": pair_id,
                "harm_level": row["harm_level"],
                "category": row["category"],
            })
    return out


SETS = {
    "medical-advice": lambda a: harmful_advice("medical", a.safe_max, a.harm_min),
    "all-advice": lambda a: harmful_advice(None, a.safe_max, a.harm_min),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", default="medical-advice", choices=sorted(SETS))
    p.add_argument("--safe-max", type=int, default=1, help="harm_level <= this counts as safe")
    p.add_argument("--harm-min", type=int, default=3, help="harm_level >= this counts as harmful")
    args = p.parse_args()

    rows = SETS[args.set](args)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.set}.jsonl"
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    n_pos = sum(1 for r in rows if r["label"] == 1)
    print(f"wrote {dest}: {len(rows)} prompts, {n_pos} pairs "
          f"(+1 safe / -1 harmful, harm_level <= {args.safe_max} vs >= {args.harm_min})")


if __name__ == "__main__":
    main()
