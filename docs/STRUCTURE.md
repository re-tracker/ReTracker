# Repository Structure

This project keeps a strict, clean root layout for readability and reuse.

## Root layout (tracked)

- `retracker/`: core library (models, inference, training, evaluation, data, io, visualization)
- `configs/`: training and evaluation configs
- `retracker/apps/`: runnable applications (tracking/streaming/multiview)
- `scripts/`: launchers and utilities (`apps/`, `smoke/`, `tools/`, `train/`, `evaluation/`)
- `docs/`: project documentation
- `tests/`: pytest suite
- `third_party/`: external dependencies and submodules

## Entry points

Preferred user-facing entry point (after `pip install -e .`):

- `retracker ...` (see `retracker/cli/`)

Convenience wrappers (thin `bash` scripts):

- Apps: `scripts/apps/*.sh`
- Training: `scripts/train/train.sh`
- Evaluation: `scripts/evaluation/retracker_*_first.sh` (see `scripts/evaluation/README.md`)

Related research docs:

- Evaluation notes: `experiments/evaluation/README.md`

## configs layout

Root `configs/` is for training/evaluation/research configuration (repo checkout use).

- `configs/paths.yaml`: portable path template (defaults). Override locally via
  `configs/paths_local.yaml` (gitignored).
- `configs/model/`: shared model config fragments referenced from training configs
  (e.g. `configs/model/retracker.yaml`).
- `configs/train/`: training configs (Hydra/LightningCLI inputs).
- `configs/eval/`: evaluation configs (Hydra CLI inputs).

Installable apps ship their own example configs under `retracker/apps/configs/`.

## retracker/apps layout

`retracker/apps/` contains runnable applications and the app runtime layer:

- `retracker/apps/runtime/`: pipelines, trackers (app runtime building blocks)
- `retracker/apps/components/`: optional app-level components (e.g. detection/segmentation)
- `retracker/apps/multiview/`: domain-specific pipelines

## retracker core layout

- `retracker/inference/`: inference engines (minimal dependencies)
  - `retracker/inference/engine.py`: stable public facade (backwards compatible)
  - `retracker/inference/engines/`: internal implementations (split by engine type)
- `retracker/training/`: training stack (Lightning, etc.) and training-only utilities under `retracker/training/utils/`
- `retracker/evaluation/`: evaluation stack and CLI helpers
- `retracker/data/`: datasets and data utilities (dataset IO lives under `retracker/data/datasets/`)
- `retracker/io/`: serialization / export helpers (e.g. dumping results, saving frame sequences)
- `retracker/utils/`: small shared runtime utilities only (kept intentionally minimal)

## experiments/tools policy

Local scripts and scratch utilities live under `experiments/tools/`:

- Keep `experiments/tools/README.md` + `experiments/tools/.gitignore` tracked
- Ignore everything else under `experiments/tools/` by default

## Local-only folders (ignored by git)

These directories are expected to be local on each machine and are ignored:

- `data/`, `weights/`, `outputs/`, `logs/`
- `checkpoints/`, `notebooks/`, `scratch/`, `test_env/`

See `.gitignore` for the full list.
