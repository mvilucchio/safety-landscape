# The lens stack (directions pillar)

This is the repo's first life, kept as-is: the Jacobian lens of
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Anthropic, July 2026), fitted for open-weight models. In the TAILS reading
(`CLAUDE.md`) it is the construction of Part B2, Eq. (5): `J_l` is the
linearised section `l -> final`, `sae_map.py` is the transfer matrix
`T_{l->l'}` in an SAE basis, and `axes.py` uses the lenses for two of its
three safety directions (`pullback` = `J_l^T w`, `contrast` = top singular
vector of `J_diff[l]`).

The maths lives in the paper's reference implementation,
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens),
pinned as a git dependency (`jlens`). **Do not reimplement fitting here.**

## The idea in four lines

```
lens_l(h) = unembed( J_l @ h ),   J_l = E[ ∂h_final / ∂h_l ]
```

The expectation is over prompts, source positions, and all current-and-future
target positions of a web-text corpus. `J_l` transports a residual at layer `l`
into the final-layer basis; the model's own unembedding then decodes it into
ranked vocabulary tokens. The rows of `W_U J_l` are the **J-lens vectors**, one
per token; J-space is the set of sparse nonnegative combinations of them.

The paper's headline structure: sensory (early) → **workspace** (middle,
~layers 38–92 of 100) → motor (late) zones, visible as blocks in the layer ×
layer similarity of the lens geometry.

## Layout

| File | What |
|---|---|
| `jspace.py` | model loading, corpus, run paths, unembedding Gram |
| `fit.py` | fit `J_l` on web text → `lens.pt` |
| `make_data.py` | build a labelled contrast set → `data/contrast/*.jsonl` |
| `contrast.py` | fit a **difference** lens `J_pos − J_neg` |
| `analyze.py` | structure of the `J` matrices → `analysis.json` |
| `spectrum.py` | contrast-lens spectrum + decoded modes → `spectrum.json` |
| `sae_map.py` | the lens in an SAE feature basis → `sae.json` |
| `eval.py` | lens quality on the paper's prompt sets → `eval-<set>.json` |
| `slice.py` | apply the lens to a prompt → interactive HTML + JSON |
| `report.py` + `report.tmpl.html` | bundle everything into one self-contained page |
| `data/` | the paper's six eval prompt sets (Apache-2.0, vendored) |
| `test_cka.py` | the one non-obvious bit of maths, checked |

Every output lives under `runs/<model>/<tag>/`. `base` is the plain web-text
lens, `contrast` a difference lens; every downstream script takes `--tag`, so a
new kind of lens costs nothing on the analysis side. `gram.pt` sits one level up
because it depends only on `W_U`.

`runs/` is gitignored. Override with `JSPACE_OUT=/path` on shared machines.

## Workflow

```bash
uv sync                                    # once

# Fit. Several GPUs: shard over prompts, one process per GPU, then merge.
uv run fit.py --model Qwen/Qwen3-0.6B --n-prompts 100 --shard 0/2 --device cuda:0 &
uv run fit.py --model Qwen/Qwen3-0.6B --n-prompts 100 --shard 1/2 --device cuda:1 &
wait
uv run fit.py --model Qwen/Qwen3-0.6B --merge

uv run analyze.py --model Qwen/Qwen3-0.6B          # J structure -> analysis.json
uv run eval.py    --model Qwen/Qwen3-0.6B --set all  # lens quality -> eval-*.json
uv run report.py  --model Qwen/Qwen3-0.6B --tags base   # -> report.html, publish as an artifact

uv run slice.py   --model Qwen/Qwen3-0.6B --example multihop
uv run slice.py   --model Qwen/Qwen3-0.6B --example list   # paper's prompt set
```

Single GPU is just `--shard 0/1` (the default) and no merge step.

## Contrast lenses

A lens fitted on labelled data, where each prompt carries `label ∈ {+1, −1}`:

```
J_diff = mean(J_i | label +1) − mean(J_i | label −1)
```

```bash
uv run make_data.py --set medical-advice                 # -> data/contrast/*.jsonl
uv run contrast.py --model M --data data/contrast/medical-advice.jsonl --shard 0/2 --device cuda:0 &
uv run contrast.py --model M --data data/contrast/medical-advice.jsonl --shard 1/2 --device cuda:1 &
wait && uv run contrast.py --model M --merge
uv run spectrum.py --model M --tag contrast
uv run analyze.py  --model M --tag contrast
uv run report.py   --model M --tags base contrast     # one page, both lenses
```

`contrast.py` accumulates per-prompt Jacobians with a signed weight in a single
pass, writing `lens.pt` (the difference), `lens.pos.pt` and `lens.neg.pt`. The
per-prompt Jacobian is `jlens.jacobian_for_prompt`, unmodified — only the
accumulation is ours, because `jlens.fit` offers neither signed weights nor
per-prompt position masks.

**Three things that decide whether the result means anything:**

1. **Pair the classes.** `data/contrast/*.jsonl` rows carry a `pair` id: the same
   user request with a safe and a harmful expert-graded response. Unpaired
   classes make `J_diff` a topic detector, not a quality detector.
2. **Exclude the shared prefix.** Each row's `prefix` is the part both members
   share; `contrast.py` sets `skip_first` per prompt so only response positions
   enter the average. Without this the request dominates both means and mostly
   cancels, diluting the signal.
3. **Shard by pair, never by row.** Pair members are consecutive lines, so
   `rows[i::n]` would give one shard every positive and the other every negative.
   `contrast.py` shards on `pair % n`.

**Check the noise floor before believing a layer.** Shards hold disjoint pairs,
so `merge` writes `split_half.json`: the cosine between two independent
estimates of the same difference, per layer. Near 0 means that layer's contrast
is unresolved at this sample size — a difference matrix always *looks* like
something. Raise the pair count before interpreting such a layer.

`spectrum.py` is the analysis counterpart to `analyze.py`. A difference matrix
is not a transport map, so it asks different questions: *where* the classes
diverge (`‖J_diff‖/‖J_pos‖` by layer), *how concentrated* the difference is
(effective rank, top-4 energy), and *what* it says — each top singular triple's
left vector decoded through `W_U`, at both signs, since the SVD fixes the pair
only up to a shared sign flip.

### Datasets

`ai-safety-institute/harmful-advice-dataset` (3650 rows, expert harm grades
0–4 from two graders per item, categories: medical / mental_health / careers /
relationships). Requests recur across harm levels, so matched pairs can be
built: **148 medical pairs, 731 across all categories** at the default
`harm_level ≤ 1` vs `≥ 3` thresholds. `--set all-advice` uses every category.

Other candidates, none adopted yet: `microsoft/PatientSafetyBench` (questions
with judge scores, no good/bad response pairs), `ImperialCollegeLondon/health_fact`
(PUBHEALTH — 4-way veracity claims, unpaired), `PKU-Alignment/PKU-SafeRLHF` and
`Anthropic/hh-rlhf` (paired chosen/rejected, large, not medical).

To add a source: write a loader in `make_data.py` emitting
`{text, label, prefix, pair}` and register it in `SETS`. Nothing downstream
changes.

## The lens in an SAE feature basis

`sae_map.py` composes the transport with a sparse-autoencoder dictionary at
each end:

```
M_l = W_enc,target^T . J_l . W_dec,l^T          [F_target, F_l]
M[j,i] = effect of one unit of feature i at layer l
         on the pre-activation of feature j at the target layer
```

The paper's readout with the SAE dictionary in place of `W_U`. The decoder is
affine so `B = W_dec` exactly; the encoder's derivative is `diag(gate).W_enc^T`
and **the gate is dropped** — `J` is already a corpus expectation and folding a
second one (`E[diag(gate)]`) in makes the object harder to read. Hence
"pre-activation".

Two facts, both asserted in `test_sae_map.py`:

- **`rank(M) <= d_model`.** `M` factors through `J`, so a 32x dictionary adds no
  directions. It is an overcomplete frame, not a bigger space; `M`'s spectrum
  measures `J` *in the SAE geometry*.
- **The spectrum needs no `F x F` matrix.** With thin QRs `W_enc^T = Q_E R_E`
  and `(J W_dec^T)^T = Q_X R_X`, the orthonormal factors cancel:
  `svdvals(M) = svdvals(R_E R_X^T)`, a `d x d` problem. Singular vectors come
  back in feature space as `U = Q_E U_core`, `V = Q_X V_core`.

### Verify the hook before trusting anything

`sae_map.py --model M --check LAYER` reconstructs the residual jlens records at
that layer with the SAE at offset 0 and at offset 1, and reports FVU. A wrong
offset silently corrupts every number downstream, so run it once per new suite.

**Drop the leading positions when scoring.** The check does this
(`SKIP_FIRST_N_POSITIONS`) because BOS is an attention sink whose residual norm
runs an order of magnitude above everything else — on gemma-2-2b, **1946 against
a median of 147**. Left in, it dominates both the error and the variance and
reports FVU ~8 for a *correctly* matched SAE, which reads as "neither offset
works". It also marks the argmin rather than everything under a fixed cutoff:
a neighbouring-layer SAE still reconstructs partially, so an absolute threshold
flags both.

### Registering a suite

`SUITES` in `sae_map.py` maps lens layers to SAE files. jlens layer `l` is the
*output* of block `l`, i.e. the input of block `l+1` — hence `offset: 1` and
`hook_resid_pre` for sources, `blocks.11.hook_resid_post` for the target.
Registered: `openai-community/gpt2` (jbloom, 32x, 24576 features, all 12
layers, `offset 1` — that suite indexes block *inputs*) and `google/gemma-2-2b`
(Gemma Scope, 7x, 16384 features, all 26 layers, npz not safetensors,
`offset 0` — verified, FVU 0.16/0.28/0.23 at L3/12/20 against 0.44/0.55/0.53
for offset 1). The Gemma SAEs are ungated but **`google/gemma-2-2b` itself is
gated**: accept the licence, then `uv run hf auth login`.

### GPT-2 runs on disk

`base` is 1000 wikitext prompts (the paper's spec), `contrast` is the full
`all-advice` set — 731 matched pairs across medical / mental_health / careers /
relationships, 1462 prompts. Both fit in ~3 and ~6 minutes on two 48 GB GPUs.
Split-half agreement on the contrast is **0.982 median** (vs 0.96 for the
148-pair Qwen medical run), so the extra pairs buy real resolution.

The gpt2 contrast lens is far more legible than Qwen's. Its leading mode
through layers 5-8 is a clean safety axis: one sign lands on
`focus / help / work / support / calm / appropriately`, the other on
`fascist / racist / extortion / thugs / criminality / Evil`. Sign flips between
adjacent layers, as always. `||J_diff||/||J_pos||` peaks at 0.196 (L1), well
below Qwen's 0.44.

`sae_map.py` is run on **both** tags. The contrast map's stable rank is ~3.8
against the transport map's ~2.0, i.e. the difference is less dominated by the
outlier direction than the transport is.

### The GPT-2 outlier is not universal

The composed map behaves completely differently on the two models:

| | gpt2 (32x) | gemma-2-2b (7x) |
|---|---|---|
| `M` stable rank | **~2** at every layer | **16 - 95** |
| `sigma_1/sigma_0` | ~0.17 | 0.75 - 0.94 |
| `M` eff_rank vs `J`'s | roughly halved | barely reduced (254 vs 285 at L0, 1962 vs 2070 at L24) |

So the GPT-2 result below is an artifact of that model's rogue residual
dimension, not a property of composing with an SAE frame. On gemma-2-2b there
is no dominant direction and the SAE basis leaves the rank essentially intact.
**Do not generalise the GPT-2 finding.**

### Result on GPT-2 small

`M`'s effective rank runs ~114 (L0) to ~338 (L8) against `J`'s own 218-490, so
the SAE basis *concentrates* the map. But its **stable rank sits at ~2 at every
layer** (`sigma_1/sigma_0 ~ 0.17`), and mode 1 carries ~50% of the energy with the
same target feature `#24008` (glosses to `"\nPosted"` + non-breaking spaces) on
top at nearly every layer, alongside `#5463` (`"000000000"`) and `#14536`
(`"SNK"`). Those are GPT-2's known high-norm / attention-sink features, so the
dominant mode looks like the outlier-dimension artifact rather than semantics.
**Read modes >= 2, not mode 1.** Worth rechecking on gemma-2-2b, whose residual
norms behave differently.

## Singular values vs eigenvalues

`J_l` is **square** — `d_model x d_model`, because it maps a residual to a
residual — but it is nowhere near symmetric (`||J - J^T|| / ||J||` ~ 1.2 on
Qwen3-0.6B), so the two spectra are different objects and the report shows both:

- **Singular values** (`svdvals`) are real and non-negative for *any* matrix,
  square or not, being the square roots of the eigenvalues of the PSD `J^T J`.
  Their realness says nothing about squareness. This is what `sv`, the heatmap,
  the overlay and the histogram all plot.
- **Eigenvalues** (`eigvals`) exist only because the matrix is square, and are
  genuinely complex here — 968 of 1024 have a non-negligible imaginary part at
  L13. `analyze.py` keeps the top `--top-eig` by modulus, normalised by
  `|lambda|max`, and runs them on CPU (LAPACK general eig has no CUDA path).
  Skip with `--no-eig` if the CPU time hurts.

Measured at L13: the transport lens' eigenvalues sit in a narrow cone in the
right half-plane with one large real outlier (`|lambda|max` 6.88); the contrast
lens' fill a disk symmetric about the real axis with its largest eigenvalue
real and *negative* (`|lambda|max` 1.41). A difference matrix behaves much more
like a random matrix than a transport does.

## Evaluations

`data/lens-eval-*.json` are the paper's six synthetic prompt sets. Each item
hides an intermediate concept the prompt never states. Metric (from
`data/README.md`): read out at the final prompt token across all layers,
`pass@k` = mean over items of the fraction of hidden concepts reaching lens
rank ≤ k at *some* layer. The per-layer curve is the more interesting output —
a mid-network hump is the workspace signature.

gemma-2-2b, 400-prompt lens: typo 0.41 / multihop 0.31 / multilingual 0.24 /
poetry 0.21 / order-ops 0.15 / association 0.02. gpt2 with a 1000-prompt lens
is near the floor everywhere (typo 0.04, multihop 0.01, order-ops 0.00) — a
124M model does not carry these concepts; its contrast and SAE results still
stand.

Baseline, Qwen3-0.6B with a 100-prompt lens (`pass@1` / peak layer): typo
0.51 / L13, multihop 0.27 / L26, poetry 0.26 / L26, order-ops 0.25 / L19,
multilingual 0.20 / L22, association 0.00 (best median rank ~2300 — a 0.6B
model simply doesn't carry the unnamed concept; not a bug).

## The report page

```bash
uv run report.py --model M --tags base contrast
```

**One page per model, not per lens** — every tag goes in side by side, so
`runs/<model>/report.html` is the single artifact to publish. Tags without an
`analysis.json` are skipped with a warning, so a base-only model still reports.
Adding a third lens is `--tags base contrast <new>`; nothing in the template
is per-tag.

It inlines the JSONs into `report.tmpl.html` — no CDN, no fetch, so it
publishes as a Claude artifact unchanged. Only the data blob (`__DATA__`) and
title (`__MODEL__`) are substituted.

Two things worth not breaking:

- Colours are emitted as `var(--…)`, never resolved hex, so the page follows
  the viewer's light/dark toggle *after* load. The sequential ramp is also
  reversed under dark, since on a dark ground near-zero has to recede into the
  surface rather than off it.
- The spectrum heatmaps and overlays share one scale across lenses. Per-panel
  autoscaling shows more detail in each but makes the comparison meaningless,
  which is the whole point of the section.
- `analyze.py` stores the **whole** spectrum (all `d_model` singular values),
  not a top-k. It used to keep 64, which cut the curve off at ~0.16 of sigma_0 —
  before any of the decay that separates the layers. `report.py` log-samples it
  to ~150 columns for drawing and labels the axis with real indices; the page
  floors the log scale at 1e-3 of sigma_0.

## What `analysis.json` contains

Small by design — it is the thing you hand to a Claude artifact.

Per layer: `fro` (transport gain, `‖J‖_F/√d`), `identity_cos` (how close to
pass-through), `stable_rank`, `eff_rank` (spectral participation ratio), `sv`
(top-64 singular values, normalised).

Across layers: `cka[l][m]`, linear CKA between the J-lens geometries of layers
`l` and `m` — the paper's zone analysis.

**The Gram trick.** CKA is defined on `W_U J_l`, which is vocab-sized. Because
vocab-centering is idempotent, `(C W_U J_l)ᵀ(C W_U J_m) = J_lᵀ G J_m` with
`G = W_UᵀC W_U` — one `d × d` matrix, cached at fit time as `gram.pt`. So
`analyze.py` never loads the model and never materialises anything vocab-sized.
`test_cka.py` checks this against the explicit definition.

## Things that will bite you

- **Cost scales with `d_model`, not depth**: `ceil(d_model / dim_batch)`
  backward passes per prompt. A wide 7B model is far more expensive than a deep
  small one. Raise `--dim-batch` until GPU memory complains (it replicates the
  prompt along batch, total FLOPs unchanged).
- **Checkpoints are `n_layers · d_model² · 4` bytes.** Fine at 0.6B (113 MB),
  4 GB+ at 13B — raise `checkpoint_every` in `fit.py` there. Use
  `--layer-stride` to fit a subset of layers when exploring.
- **Resume is fingerprinted on the prompt list** (`ckpt.<sha>.shardN.pt`), so
  changing `--n-prompts` or `--prompts` starts fresh instead of silently
  resuming a different corpus. Deleting `runs/<model>/ckpt.*` is always safe.
- **`from_hf` mutates the HF model in place** (`requires_grad_(False)`, may set
  `add_bos_token`). Don't reuse that object for anything else.
- **Quality saturates fast.** The paper uses 1000 × 128-token sequences; ~100
  is usable (§9.3). Start at 50 when iterating.
- `jlens` only fits `source_layers < target_layer`, so the last layer never has
  a `J` — `analyze.py`'s `layers` list stops one short of `n_layers`.

## Not implemented yet

- **Sparse J-space decomposition** (gradient pursuit for the sparsest
  nonnegative combination of ~25 J-lens vectors). The paper describes it; the
  reference repo does not ship it. Needed for "which concepts are active in
  this activation" as opposed to "what is this activation disposed to say".
- Causal work: ablating top-k J-lens directions, patching lens coordinates.

## Conventions

- `uv run <script>`, never a bare `python`.
- One flat script per verb at repo root; no package, no framework. Adding a new
  experiment means adding one script that loads `lens.pt` and writes JSON.
- Analyses that only need `J` must stay model-free (read `lens.pt`/`gram.pt`
  only) so they run on a laptop.
