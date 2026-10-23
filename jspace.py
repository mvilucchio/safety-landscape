"""Shared plumbing: model loading, data rendering, output paths, generation
helpers. Everything heavy lives in `jlens`; this only wires it to local
conventions.
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
    lens; `contrast` is a labelled difference lens."""
    d = model_dir(model_id) / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_model(model_id: str, *, device: str = "cuda:0", dtype: str = "bfloat16"):
    """Return (jlens model, hf model, tokenizer).

    device="auto" shards across GPUs (needed when one does not fit); anything
    else is a single device. `from_hf` freezes the HF model in place.
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
    """Lens-fitting prompts: WikiText-103 by default, or a local file of
    blank-line-separated prompts."""
    if path:
        chunks = Path(path).read_text(encoding="utf-8").split("\n\n")
        return [c.strip() for c in chunks if c.strip()][:n]
    from jlens.examples import load_wikitext_prompts

    return load_wikitext_prompts(n)


def unembed_gram(model) -> torch.Tensor:
    """G = W_U^T C W_U, the vocab-centered Gram of the unembedding [d, d]: all
    that linear CKA between layers' lens geometries needs (analyze.py)."""
    W = model._lm_head.weight.detach().float()  # [vocab, d]
    W = W - W.mean(dim=0, keepdim=True)
    return (W.T @ W).cpu()


# --- contrast data -> text --------------------------------------------------


def render(row: dict, tok, template: str) -> dict:
    """{prefix, text, prompt} for one contrast row (make_data.py).

    plain  the data's own `User: … / Assistant:` rendering, model-agnostic.
    chat   the tokenizer's chat template around the same request/response, so
           instruct models see their training format. A BOS the template emits
           is stripped: encode() adds it once.

    `prefix` precedes the response (position masks), `text` is the whole
    thing, `prompt` is what generation starts from.
    """
    if template == "plain":
        # The prefix ends in "Assistant: "; generation starts without the
        # trailing space so the first token carries its own, as in training.
        return {"prefix": row["prefix"], "text": row["text"], "prompt": row["prefix"].rstrip()}
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": row["request"]}], tokenize=False, add_generation_prompt=True
    )
    if tok.bos_token and prefix.startswith(tok.bos_token):
        prefix = prefix[len(tok.bos_token):]
    return {"prefix": prefix, "text": prefix + row["response"], "prompt": prefix}


def prompts(rows: list[dict], tok, template: str) -> list[str]:
    """The unique generation prompts of a contrast set, in data order."""
    return list(dict.fromkeys(render(r, tok, template)["prompt"] for r in rows))


def load_axes(model_id: str, name: str, method: str):
    """(U [n_layers, d], class stats, held-out d' per layer, template) from axes.py."""
    d = model_dir(model_id) / "axes"
    pt = torch.load(d / f"{name}.pt", weights_only=True)
    sep = json.loads((d / f"{name}.json").read_text())["separation"][method]
    return pt["u"][method], pt["stats"][method], [pl["d_prime"] for pl in sep], pt.get("template", "plain")


# --- generation -------------------------------------------------------------


def generation_setup(hf, tok) -> tuple[list[int], int]:
    """(eos ids, pad id). Left-pads for batched prompts; pads with EOS if the
    tokenizer has no pad token."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    e = hf.generation_config.eos_token_id
    ids = {e} if isinstance(e, int) else set(e or [])
    ids.add(tok.eos_token_id)
    return sorted(i for i in ids if i is not None), tok.pad_token_id


def sampling(temperature: float, max_new_tokens: int, pad: int) -> dict:
    """generate() kwargs for a pure temperature-T chain: top-k/top-p off,
    which the model's generation_config would otherwise switch on."""
    return {"do_sample": True, "temperature": temperature, "top_k": 0, "top_p": 1.0,
            "max_new_tokens": max_new_tokens, "pad_token_id": pad, "return_dict_in_generate": True}


def first_eos(toks: torch.Tensor, eos: list[int]) -> torch.Tensor:
    """Index of the first EOS in each generated row, or the row length if none."""
    hit = torch.isin(toks, torch.tensor(eos))
    return torch.where(hit.any(1), hit.int().argmax(1), torch.full((len(toks),), toks.shape[1]))


class Projector:
    """Forward hooks recording <u_i, h> at the last position of every block
    call. generate() makes one call per step, so after a call `stack()` is
    [batch, steps, len(layers)]; step 0 is the end of the prompt."""

    def __init__(self, model, U: torch.Tensor, layers: list[int]) -> None:
        self.rec: list[list[torch.Tensor]] = [[] for _ in layers]
        self.handles = [
            model.layers[l].register_forward_hook(self._hook(i, U[i].to(model.input_device)))
            for i, l in enumerate(layers)
        ]

    def _hook(self, i: int, u: torch.Tensor):
        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            self.rec[i].append((h[:, -1, :].float() @ u).cpu())
        return hook

    def clear(self) -> None:
        for r in self.rec:
            r.clear()

    def stack(self) -> torch.Tensor:
        return torch.stack([torch.stack(r, 1) for r in self.rec], -1)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
