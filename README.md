# DIFFerentiable EntRopic Process LOss (DIFF-ERO)

Next Activity Prediction (NAP) for process event logs using a differentiable Entropic Relevance loss.

## Key idea

Standard NAP models are trained with cross-entropy loss on the next token. This repo experiments with replacing or mixing in an **Entropic Relevance (ER) loss** that evaluates predicted activity distributions against an SDFA (stochastic DFA) built from the trace suffixes — encoding how well the model's output aligns with the process model implied by the data.

## Models

| Model | Description |
|---|---|
| `SDFA_NAP_model` | Transformer encoder + SDFA projector; used when ER loss is active |
| `NAP_model` | Transformer encoder without SDFA projection; used as baseline |

## Experiment grid (`start_differo_nap_experiments.py`)

The script sweeps the following dimensions:

| Parameter | Values |
|---|---|
| Loss | ER (`er_loss_use=True`) vs CE / rank loss baseline |
| `d_model` | 16, 32, 64 |
| `mix_lambda` | 0, 0.1, 0.2, 0.5 (weight of ER vs CE) |
| `sdfa_ce` | True / False (use CE on SDFA projection) |
| `local` | True / False (local per-step SDFA target vs global trace SDFA) |
| `batch_size` | 16, 32, 64, 128 (local) or 32, 128 (baseline) |
| `rank_loss` | True / False (baseline only) |
| Seeds | 10 seeds (56, 8, 15, 76, 23, 42, 4, 108, 16, 1089) |

Results per dataset are appended to `results_ns_nap_prediction_<dataset>.csv`.

## Datasets

BPI12, BPI13 (CP, Incidents), BPI15, BPI17, BPI19, BPI20 (PrepaidTravelCosts, RequestForPayment, TravelPermitData), Sepsis, BPI11 — loaded via `skpm` (https://github.com/raseidi/skpm).

## Setup

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt
python start_differo_nap_experiments_15.py
```

Requires CUDA for reasonable runtime; falls back to CPU automatically.
