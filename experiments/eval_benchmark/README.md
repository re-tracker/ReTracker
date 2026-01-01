# Eval Benchmark (TAP-Vid DAVIS First)

This folder contains a lightweight evaluation benchmark for **TAP-Vid DAVIS** using the **first-query protocol**.

Goals:
- Evaluate **ReTracker** on TAP-Vid DAVIS (first).
- Support multi-GPU parallelism across videos (one process per GPU).
- Save per-video prediction caches (`.npz`) and support resume.
- Write per-video metrics (`.json`) + aggregated result (`.json`).

## Dataset

Expected dataset layout under `--dataset-root`:

```
<dataset-root>/
  tapvid_davis/
    tapvid_davis.pkl
```

The pickle is expected to be a dict:
- `video`: `(T,H,W,3)` uint8, or a list of JPEG bytes
- `points`: `(N,T,2)` float32 normalized `(x,y)` in `[0,1]`
- `occluded`: `(N,T)` bool (True = occluded)

## Run

Minimal example:

```bash
python experiments/eval_benchmark/run_tapvid_davis_first.py \
  --dataset-root /path/to/tapvid_val \
  --ckpt /path/to/retracker_b1.7.ckpt \
  --gpus 0
```

## Multi-method suite (subprocess conda envs)

If you have multiple trackers installed into separate conda environments, you can
evaluate them under a unified suite runner:

```bash
python experiments/eval_benchmark/run_tapvid_davis_first_suite.py \
  --dataset-root /path/to/tapvid_val \
  --methods retracker,trackon2,cotracker3_offline,cotracker3_online,tapir,tapnext \
  --gpus 0,1
```

Notes:
- Each method runs via its own conda env (`--retracker-env`, `--trackon-env`, `--tapir-env`, `--tapnext-env`).
- Per-sequence inputs are cached as NPZ under `.../tapvid_davis_first/cache/` and
  reused across methods.

### Convenience wrapper (bash)

If you prefer the same "one-liner" UX as `scripts/evaluation/*.sh`, you can use:

```bash
METHODS="retracker,trackon2,cotracker3_offline,cotracker3_online,tapir,tapnext" \
  GPUS="0,1" \
  ./scripts/evaluation/eval_benchmark_davis_first_suite.sh /path/to/retracker_b1.7.ckpt
```

Multi-GPU:

```bash
python experiments/eval_benchmark/run_tapvid_davis_first.py \
  --dataset-root /path/to/tapvid_val \
  --ckpt /path/to/retracker_b1.7.ckpt \
  --gpus 0,1,2,3
```

CPU-only:

```bash
python experiments/eval_benchmark/run_tapvid_davis_first.py \
  --dataset-root /path/to/tapvid_val \
  --ckpt /path/to/retracker_b1.7.ckpt \
  --gpus ""
```

## Resume / overwrite

By default, the runner resumes if a per-video cache exists:
- `--resume` (default): reuse `predictions/pred_<seq>.npz` if present
- `--no-resume`: always recompute predictions
- `--overwrite`: recompute predictions even if cached

## Outputs

By default, outputs are written under `outputs/eval_benchmark/tapvid_davis_first/retracker/`:

```
manifest.json
predictions/
  pred_<seq>.npz
metrics/
  metrics_<seq>.json
result_eval_tapvid_davis_first.json
```

The suite runner writes per-method subfolders:

```
outputs/eval_benchmark/tapvid_davis_first/
  cache/
  retracker/
  trackon2/
  cotracker3_offline/
  cotracker3_online/
  tapir/
  tapnext/
  compare_methods.json
```

## Smoke test (optional)

If your machine has the dataset/checkpoint available, you can run:

```bash
CKPT=/abs/path/to/retracker_b1.7.ckpt \
  bash experiments/eval_benchmark/scripts/smoke_test_davis_first.sh
```
