# safety-landscape

Testbed for **TAILS** (*Tail Analysis of Intelligence, Learning & Safety*,
M. Biroli's ERC Starting Grant proposal; Part B1 = `documents/main_v8.pdf`,
Part B2 = `documents/main_v6.pdf`): the statistics of *rare* unsafe behaviour
in open-weight LLMs, treated as a stochastic process with the toolkit of
extreme values, first passage and rare-event sampling. Three pillars, three
kinds of script:

| Pillar | In the proposal | Scripts | Question |
|---|---|---|---|
| **Directions** | WP1, secondary objective (Eq. 5, BBP) | `axes.py` + the lens stack (`LENS.md`) | which residual directions carry "unsafe", and how well can each be measured? |
| **Process & landscape** | WP1, O1(b): B1 Eq. (2), B2 Eq. (4) | `trajectory.py`, `landscape.py` | along such an axis, what is the well `V_t(s)`, its dynamics, the first passage across `s*`? |
| **Rare events** | WP2: GKTL cloning, B2 Eqs. (7)–(9) | `clone.py` | how probable is a failure brute force cannot see, at what cost? |

The lens machinery (`fit.py`, `contrast.py`, `analyze.py`, `spectrum.py`,
`sae_map.py`, `report.py`, …) is the repo's first life and is kept unchanged:
it is the transfer-matrix construction of Part B2 and it feeds two of the
three axes. Everything about it, including its results, lives in `LENS.md`.

## The idea in four lines

```
u_l                       unit vector at layer l, pointing toward the unsafe class      axes.py
s_t = <u_l, h_t>          last-token residual at generation step t, projected           trajectory.py
rho_t(s) ∝ e^{-V_t(s)/T}  the landscape;  tau = min{t : s_t >= s*}  the failure         landscape.py
P[S_L >= aL] = E_k[1{S_L >= aL} e^{-k S_L}] Π_t R_t   cloning, undoing the tilt          clone.py
```

Fixed well vs eroding well (B2, Eq. 4): hazard constant in `t` (exponential
survival) against growing as `e^{εt/T}` (Gompertz). `landscape.py` fits both.

## Layout

| File | What |
|---|---|
| `jspace.py` | model loading, run paths, contrast-row rendering (plain or chat template), axis loading, generation helpers: pure-T sampling kwargs, EOS masking, projection hooks |
| `make_data.py` | paired safe/harmful responses → `data/contrast/*.jsonl`: the labels behind every axis |
| `axes.py` | per-layer directions `diffmean`, `pullback` (needs `base/lens.pt`), `contrast` (needs `contrast/lens.pt`); held-out d', AUC, pair accuracy; class means for `s*`; `--template plain\|chat` → `axes/<set>.{pt,json}` |
| `trajectory.py` | sample at temperature T, record `s[n, t, l]` for every layer → `traj/<name>.pt` |
| `landscape.py` | model-free: potential, Kramers–Moyal drift/diffusion, Ornstein–Uhlenbeck well per time window, Kaplan–Meier survival, fits A/B → `traj/<name>.landscape.json` |
| `clone.py` | GKTL cloning on one prompt, `k = 0` is brute force on the same budget → `clone/<name>.json` |
| `test_clone.py` | the estimator against a Gaussian walk whose tail is known in closed form |
| `fit.py` `contrast.py` `analyze.py` `spectrum.py` `sae_map.py` `eval.py` `slice.py` `report.py` `test_cka.py` `test_sae_map.py` | the lens stack — see `LENS.md` |
| `documents/` | the proposal (gitignored) |

Outputs go to `runs/<model>/{axes,traj,clone}/`, next to the lens dirs
`runs/<model>/<tag>/`. `runs/` is gitignored; `JSPACE_OUT=/path` overrides.

## Workflow

```bash
uv sync
uv run make_data.py --set all-advice                        # 731 pairs, once
uv run axes.py --model Qwen/Qwen3-0.6B                       # 25 s; pullback/contrast appear if the lenses exist
uv run trajectory.py --model Qwen/Qwen3-0.6B --n-prompts 128 --samples 16 \
    --max-new-tokens 160 --temperature 1.0                   # 70 s per 2048 completions
uv run landscape.py --model Qwen/Qwen3-0.6B --traj all-advice-diffmean-T1.0
uv run clone.py --model Qwen/Qwen3-0.6B --prompt 3 --walkers 256 --blocks 8 \
    --block-tokens 16 --a -3.25 --k 0 0.1 0.2 --repeats 3    # 6 s per run
uv run test_clone.py
```

Every script is single-device (`--device`); with several GPUs run different
temperatures, prompts or shards as separate processes. Lens fitting shards
over prompts (`fit.py --shard i/n`, then `--merge`), see `LENS.md`.

Measured on one 48 GB Ada-class GPU, Qwen3-0.6B in bf16: `axes.py` on 1462
responses 25 s; `trajectory.py` 128 prompts × 16 samples × 160 tokens 71 s;
one `clone.py` run of 256 walkers × 8 blocks × 16 tokens 6 s; `landscape.py`
seconds on CPU. These are generation, not backward passes: cost scales with
tokens × batch, and `d_model` matters far less than for fitting a lens.

## What decides whether a number means anything

- **The axis points toward unsafe.** `label -1` (harmful) minus `label +1`
  (safe): `s*` is crossed upward and a "fixed well" sits *below* `s*`. Every
  script assumes this sign.
- **Score an axis held out.** `axes.py` rebuilds the directions on even pairs
  and scores odd ones. The lenses behind `pullback` and `contrast` were fitted
  on all pairs, so for those two only the diffmean part is held out. Say so
  when quoting their d'.
- **Token-level crossings are not rare.** `s_t` fluctuates with width ≈ 3.8
  around its well while the class means are 7 apart, so `s_t >= s*` at the
  midpoint happens in every completion (58% of them already at `t = 0`). The
  rare events are *time averages*: `landscape.py` reports the token first
  passage (Eq. 2) and the running-mean one side by side; `clone.py` tilts the
  block mean, as the heat-wave estimator does.
- **Choose `k` by the raw fraction.** The event must be typical under the
  tilt: raw fraction near 1/2, parent fraction well above `1/N`. `k = 0` is
  the reference. An estimate scattering over decades with raw fraction 1.0
  means `k` overshot.
- **Same rendering everywhere.** `axes.py --template plain|chat` decides how
  the contrast rows are rendered and the other scripts reuse what the axes
  were built with, so `s*` refers to one distribution. Plain is the data's
  `User: … / Assistant:` text (Qwen3 then runs without `<think>`); chat is the
  tokenizer's template, the right choice for instruct models.
  `--temperature` is pure (top-k/top-p off), as the theory requires.
- **Coherence is a confound.** Degenerate text sits on the unsafe side of the
  diffmean axis (the T = 1.3 row below). Before reading an axis as "safety",
  read what the high-`s` completions say; `clone.py` saves the top walker of
  every run for exactly this.

## First results: Qwen3-0.6B, `all-advice`, 2026-10-23

**Axes** (held-out d' at the best layer): diffmean 3.21 (L15), pullback 2.70
(L17), contrast top singular vector 2.1 (L13). Pairwise cosines stay ≤ 0.3 at
the layers that matter: three different directions, each separating the
classes. One direction is not the whole story, which is the proposal's own
de-risking remark on the 1-D reduction.

**Landscape** along diffmean/L15, 128 prompts × 16 samples × 160 tokens,
`s* = −6.78` (class means: safe −10.3, harmful −3.25):

| T | well bottom `s_eq` | curvature κ per token | width | final running mean ≥ `s*` | ε/T (fit B) |
|---|---|---|---|---|---|
| 0.7 | −9.4 | 0.21 | 3.6 | 0.12 | −0.005 |
| 1.0 | −7.5 | 0.20 | 3.8 | 0.38 | −0.011 |
| 1.3 | −1.8, still rising | 0.22 | 4.1 | 0.96 | −0.023 |

Relaxation time ≈ 5 tokens, nearly independent of T; the well moves *up*
with T while its shape stays, which is the deployment knob the proposal
expects. Within 160 tokens the hazard *falls* (ε < 0): it is the prompt-end
transient (`s_0 = −6.5`) relaxing into the well, not erosion. Eq. (4)'s
question needs multi-turn contexts. T = 1.3 is the confound case: the text is
gibberish and the "well" sits past the harmful mean.

**Cloning** on prompt 3 (a vaccination side-effect question), event "mean
score over 128 tokens ≥ μ_harm": brute force 3.0e-2 ± 0.2e-2 (3 × 256
walkers); cloning at `k = 0.1` gives 3.2e-2 ± 0.6e-2 with the event typical
(raw 0.57); `k = 0.5` collapses (parent fraction 0.5, estimates spread over
two decades). Agreement, no speedup yet: this target is not rare. Cloning
pays below p ≈ 1e-4, so raise `--a` or lengthen `--blocks`.

## Not implemented yet

- **Scores beyond the projection**: a judge on the transcript (the only
  option for closed models) and the committor probe. Judge false positives
  are the proposal's central risk; synthetic, program-checked tasks are its
  answer.
- **Multi-turn and long context.** The erosion test needs conversations.
  `clone.py`'s block is a token block; a turn is the same loop with a user
  message inserted between blocks.
- **Adaptive multilevel splitting** for single-threshold crossings.
- **Best-of-N grids** (the proof-of-concept fit, B1 Eq. 1) and the
  energetic-vs-entropic recipe comparison across post-training checkpoints.
- **BBP null** for the SAE transfer matrices: `sae_map.py` builds `T_{l→l'}`,
  the random-init clone and the outlier test are not written.
- **Figures.** `report.py` renders the lens page only; the landscape and
  clone JSONs have no page. No plotting library is installed.
- **A safety-tuned model above 0.6B.** `google/gemma-2-2b` on disk is the
  *base* model (useful as the no-safety-well control); `-it` is not
  downloaded. Qwen2.5/Llama-3 instruct families are the proposal's targets.

## Conventions

- `uv run <script>`, never a bare `python`.
- One flat script per verb at repo root; no package, no framework. A new
  experiment is one script that reads `axes/*.pt`, `traj/*.pt` or `lens.pt`
  and writes JSON.
- Analyses that only need saved tensors (`landscape.py`, `analyze.py`,
  `test_clone.py`) stay model-free so they run on a laptop.
- `jlens` layer `l` is the output of block `l`; the final layer is
  `n_layers − 1`. `axes.py` and `trajectory.py` index layers the same way.
