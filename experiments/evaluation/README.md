# Evaluation

This folder contains **detailed** notes for running ReTracker evaluation on public benchmarks.

If you only want a quick command to run evaluation, start here instead:

- `scripts/evaluation/README.md` (wrapper scripts + environment variables)

## What is evaluated

The current evaluation entrypoint is:

- `python -m retracker.evaluation.cli` (Hydra config in `configs/eval/`)

Most configs target TAP-Vid variants:

- DAVIS: `tapvid_davis_first`, `tapvid_davis_strided`
- Kinetics: `tapvid_kinetics_first`
- RoboTAP: `tapvid_robotap_first`
- RGB stacking: `tapvid_stacking_first`
- PointOdyssey "retrack": `tapvid_retrack_first`

Wrapper scripts exist for common "first frame" settings:

- `scripts/evaluation/retracker_davis_first.sh`
- `scripts/evaluation/retracker_kinetics_first.sh`
- `scripts/evaluation/retracker_robotap_first.sh`
- `scripts/evaluation/retracker_rgbstacking_first.sh`
- `scripts/evaluation/retracker_retrack_first.sh`

## Install

Evaluation uses Hydra:

```bash
python -m pip install -e ".[eval]"
```

## Dataset layout (expected)

Evaluation code constructs dataset paths under `dataset_root` based on `dataset_name`.
Below is what `dataset_root` should contain.

TAP-Vid DAVIS:

- `${dataset_root}/tapvid_davis/tapvid_davis.pkl`

TAP-Vid RGB stacking:

- `${dataset_root}/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl`

TAP-Vid RoboTAP:

- `${dataset_root}/tapvid_robotap/` (directory)

TAP-Vid Kinetics:

- `${dataset_root}/kinetics/kinetics-dataset/k700-2020/tapvid_kinetics` (directory)

PointOdyssey retrack:

- `${dataset_root}/retrack.pkl`

If your datasets live elsewhere, pass an explicit `dataset_root` to the wrapper script or set
`RETRACKER_DATA_ROOT` (see `scripts/evaluation/README.md`).

## Outputs

The evaluation CLI writes into `exp_dir`:

- `expconfig.yaml` (the resolved Hydra config)
- `result_eval_<dataset_name>.json` (final metrics + runtime)

The wrapper scripts default to:

- `./outputs/eval/<run_id>/<config>/<timestamp>/`

## How to run

Recommended: wrapper scripts (resolve run_id -> checkpoint automatically):

```bash
bash scripts/evaluation/retracker_davis_first.sh <run_id|run_name_prefix|ckpt_path> [dataset_root]
```

### Reproduce: TAP-Vid DAVIS (retracker_b1.7.ckpt)

This is the exact setup we use to reproduce DAVIS metrics for the checkpoint:

- `./weights/retracker_b1.7.ckpt`
- dataset root: `data/tapvid_local` (writable local copy)

Prepare a writable DAVIS pickle (optional but recommended on shared machines where `data/tapvid` is read-only):

```bash
mkdir -p data/tapvid_local
unzip -o data/davis_val/tapvid_davis.zip -d data/tapvid_local
```

Single GPU:

```bash
conda activate retracker_env
bash scripts/evaluation/retracker_davis_first.sh ./weights/retracker_b1.7.ckpt data/tapvid_local
```

Multi GPU (sequence-level sharding; supports resume via `done/` markers and cached `dumps/*.npz`):

```bash
conda activate retracker_env
RETRACKER_EVAL_SAVE_VIDEO=true \
bash scripts/evaluation/retracker_davis_first_multigpu.sh ./weights/retracker_b1.7.ckpt data/tapvid_local 0,1
```

If the run is interrupted and you want to resume into the same output folder, set `RETRACKER_EVAL_OUT_DIR`
to the previous run directory and re-run the same command:

```bash
conda activate retracker_env
RETRACKER_EVAL_OUT_DIR=outputs/eval/retracker/eval_tapvid_davis_first/<timestamp> \
bash scripts/evaluation/retracker_davis_first_multigpu.sh ./weights/retracker_b1.7.ckpt data/tapvid_local 0,1
```

Outputs are written under:

- `outputs/eval/retracker/eval_tapvid_davis_first/<timestamp>/`

Advanced: call the Hydra entrypoint directly:

```bash
retracker eval \
  --config-name eval_tapvid_davis_first \
  checkpoint=/path/to/last.ckpt \
  dataset_root=/path/to/dataset_root \
  exp_dir=./outputs/eval/manual_run
```

## Reproducibility tips

- Pin the checkpoint path you evaluate, and keep it next to the JSON results.
- Set `CUDA_VISIBLE_DEVICES` explicitly if running on multi-GPU machines.
- Prefer wrapper scripts to avoid Hydra working-directory confusion.
