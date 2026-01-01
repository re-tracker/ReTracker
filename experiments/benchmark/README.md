# Tracking Benchmark (ReTracker + Third-Party)

Goal: given a video and an explicit set of query points (t, x, y), run multiple
trackers and produce:

- per-tracker overlay videos
- a single mosaic `.mp4` for side-by-side comparison

Trackers included by default:

- ReTracker (this repo)
- Track-On2 (third-party)
- CoTracker3 offline + online (third-party)
- TAPIR (third-party, PyTorch model; method name: `tapir`)
- TapNext (third-party, JAX/Flax model; method name: `tapnext`)

## Quick Start (after deps exist)

```bash
python experiments/benchmark/benchmark.py \
  --pair /path/to/video.mp4 /path/to/queries.txt \
  --out-dir outputs/benchmark
```

Queries file format:

```
# format: t x y
0 100 200
0 120 200
...
```

## Install third-party code + checkpoints (optional)

This repo does **not** ship third-party repos by default.

```bash
bash experiments/benchmark/scripts/install_third_party.sh
```

## Create conda envs (optional)

If you want to run the full multi-method benchmark via subprocess runners, create the
separate conda envs used by third-party methods:

```bash
bash experiments/benchmark/scripts/setup_envs.sh
```

Defaults:
- `trackon2`: Track-On2 + CoTracker3
- `tapnext`: TAPIR (PyTorch)
- `tapnet`: TapNext (JAX/Flax, CPU by default)

This downloads the following checkpoints into `experiments/benchmark/checkpoints/`:
- Track-On2: `trackon2_dinov2_checkpoint.pt`
- CoTracker3: `scaled_offline.pth`, `scaled_online.pth`
- TAPIR (PyTorch): `tapir_checkpoint_panning.pt`, `causal_bootstapir_checkpoint.pt`
- TapNext (JAX): `bootstapnext_ckpt.npz`

And clones the following repos into `experiments/third_party/`:
- Track-On2: `track_on/`
- CoTracker3: `co-tracker/`
- TapNet (for TAPIR imports / reference): `tapnet/`

## Smoke test

```bash
bash experiments/benchmark/scripts/smoke_test.sh
```

## Extending

Add a new tracker by implementing the `Tracker` interface in:

- `experiments/benchmark/trackers/base.py`

and register it in:

- `experiments/benchmark/benchmark.py`

The shared data model lives in:

- `experiments/benchmark/core/types.py`
