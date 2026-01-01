# Training Scripts

This folder contains small, user-facing launchers for training ReTracker.

The project uses LightningCLI under `retracker/training/cli.py`. The shell
wrappers here are intentionally thin and avoid machine-specific assumptions.

## Prerequisites

Install training dependencies:

```bash
python -m pip install -e ".[train]"
```

Configure your dataset paths (recommended):

```bash
cp configs/paths_local.yaml.example configs/paths_local.yaml
$EDITOR configs/paths_local.yaml
```

## Stage4 (Unified) Training

The canonical entrypoint is:

```bash
bash scripts/train/train.sh stage4 <run_name> [extra LightningCLI args...]
```

Examples:

```bash
# Basic run
bash scripts/train/train.sh stage4 my_exp

# Fast sanity check (1 train + 1 val batch)
bash scripts/train/train.sh stage4 debug_run --trainer.fast_dev_run=1 --trainer.devices=1

# Override output root (default: ./outputs/training)
bash scripts/train/train.sh stage4 my_exp --output_root ./outputs/training

# Disable auto-resume
bash scripts/train/train.sh stage4 my_exp --no_resume
```

Notes:
- The script auto-resumes from `outputs/training/<run_id>/version_shared_ckpt/last.ckpt`
  if it exists.
- If `<run_name>` contains `debug`, the script will merge in
  `configs/train/debug/stage4_debug.yaml` (if present).

## Smoke Test (No External Datasets)

If you just want to verify the training stack works (GPU, Lightning, imports),
use:

```bash
bash scripts/smoke/train_stage4_dummy.sh
```

