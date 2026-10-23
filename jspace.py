"""Shared plumbing: model loading, corpus, output paths.

Everything heavy lives in `jlens` (the paper's reference implementation);
this module only wires it to local conventions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

OUT = Path(os.environ.get("JSPACE_OUT", "runs"))


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def model_dir(model_id: str) -> Path:
    """Artifacts that depend on the model but not on which lens was fitted."""
    d = OUT / slug(model_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(model_id: str, tag: str = "base") -> Path:
    """One lens and everything derived from it. `base` is the plain web-text
    lens; `contrast` is a labelled difference lens. Any downstream script takes
    --tag, so a new kind of lens costs nothing on the analysis side."""
    d = model_dir(model_id) / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_model(model_id: str, *, device: str = "cuda:0", dtype: str = "bfloat16"):
    """Return (jlens model, hf model, tokenizer).

    device="auto" shards across both GPUs (needed above ~30B here); anything
    else is a single device. Compile is left off — it fights device_map="auto"
    and only pays off for long fits.
    """
    import jlens
    import transformers

    kwargs = {"dtype": getattr(torch, dtype)}
    if device == "auto":
        kwargs["device_map"] = "auto"
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "auto":
        hf = hf.to(device)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    return jlens.from_hf(hf, tok), hf, tok


def corpus(n: int, path: str | None = None) -> list[str]:
    """Fitting prompts. Default: WikiText-103 (pretraining-like, as in §9.3).

    `path` overrides with a local file of blank-line-separated prompts, so a
    fit can be repeated on a domain-specific distribution.
    """
    if path:
        chunks = Path(path).read_text(encoding="utf-8").split("\n\n")
        return [c.strip() for c in chunks if c.strip()][:n]
    from jlens.examples import load_wikitext_prompts

    return load_wikitext_prompts(n)


def unembed_gram(model) -> torch.Tensor:
    """G = W_U^T C W_U, the vocab-centered Gram of the unembedding [d, d].

    The paper's J-lens vectors are the rows of W_U @ J_l. Linear CKA between
    two layers' lens geometries only ever needs J_l^T G J_m — so caching this
    one d x d matrix makes every downstream layer comparison model-free (and
    GPU-free). Computed once at fit time.
    """
    W = model._lm_head.weight.detach().float()  # [vocab, d]
    W = W - W.mean(dim=0, keepdim=True)  # center over vocab
    return (W.T @ W).cpu()
