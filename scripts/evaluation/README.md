# Evaluation Scripts

These scripts are thin wrappers around `python -m retracker.evaluation.cli` for common datasets.

## Install

Evaluation uses Hydra for configuration:

```bash
python -m pip install -e ".[eval]"
```

## Usage

Each script takes:

1. `<run_id | run_name_prefix | checkpoint_path>`
2. Optional `dataset_root` override

Example (TAP-Vid DAVIS):

```bash
bash scripts/evaluation/retracker_davis_first.sh my_run_abcd1234 data/tapvid
```

## Environment Variables

- `RETRACKER_OUTPUT_ROOT`: training output root (default: `./outputs/training`)
- `RETRACKER_EVAL_OUTPUT_ROOT`: evaluation output root (default: `./outputs/eval`)
- `RETRACKER_DATA_ROOT`: (legacy) if set, scripts will use `dataset_root=$RETRACKER_DATA_ROOT/<default_dataset_path>`
- `RETRACKER_EVAL_DRY_RUN`: if set to `1`, prints resolved paths + creates the output directory, but
  does not run the Python evaluation.
- `RETRACKER_EVAL_OUT_DIR`: (multi-GPU launcher only) force the output directory so interrupted runs can resume
  into the same folder (used by `scripts/evaluation/retracker_davis_first_multigpu.sh`).
- `RETRACKER_EVAL_SAVE_VIDEO`: (multi-GPU launcher only) set to `true/false` to enable/disable per-sequence mp4 output.
- `RETRACKER_EVAL_VISUALIZE_EVERY`: (multi-GPU launcher only) visualize every N sequences (default: `1`).

## Using configs/paths_local.yaml (recommended)

If you want to avoid passing `dataset_root` every time, configure machine-local paths:

1. Copy the example:

```bash
cp configs/paths_local.yaml.example configs/paths_local.yaml
```

2. Set (at least) `paths.TAPVID_ROOT`, e.g.:

```yaml
paths:
  TAPVID_ROOT: "data/tapvid"
```

Now you can run:

```bash
bash scripts/evaluation/retracker_davis_first.sh ./weights/retracker_b1.7.ckpt
```

## Notes

- These scripts do **not** activate conda environments or create symlinks. Use your own environment
  management and set `RETRACKER_*` variables as needed.
- If you pass a `run_name_prefix` (e.g. `my_run`), it will be resolved to a unique run directory
  like `my_run_<gitsha>` under `RETRACKER_OUTPUT_ROOT`. If multiple matches exist, the script will
  error and list candidates.
