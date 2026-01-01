# Model Zoo

This repository does not store large binary checkpoints in git. Place downloaded
weights under `weights/` (gitignored) or set an absolute path in
`configs/paths_local.yaml`.

## ReTracker Checkpoints

| Name | Description | Download |
|------|-------------|----------|
| `retracker_b1.7.ckpt` | Full model (ICCV 2025) | [Google Drive](https://drive.google.com/file/d/1BfCBAwj_jJLdI4wwforleTXEUS5NsDOq/view?usp=drive_link) |
| `retracker_light_b1.7.ckpt` | Lightweight version | [Google Drive](https://drive.google.com/file/d/1KxXcJeDw1O7FdbhVayXL_1ZKyodI0p7D/view?usp=drive_link) |

### Download Instructions

```bash
# Create weights directory
mkdir -p weights

# Download using gdown (pip install gdown)
pip install gdown

# Full model
gdown 1BfCBAwj_jJLdI4wwforleTXEUS5NsDOq -O weights/retracker_b1.7.ckpt

# Lightweight model
gdown 1KxXcJeDw1O7FdbhVayXL_1ZKyodI0p7D -O weights/retracker_light_b1.7.ckpt
```

### Notes

- If you change the file location, update `PRETRAINED_CKPT` in
  `configs/paths_local.yaml`.
- Apps and evaluation scripts accept explicit checkpoint paths via CLI
  flags (e.g., `--ckpt_path`).

## Backbone Weights (DINOv3)

DINOv3 weights are not bundled. You can either:

- **Auto-download**: set `DINOV3_WEIGHTS: null` and `DINOV3_REPO: "auto"` in
  `configs/paths_local.yaml`, or
- **Offline mode**: download the weights separately and point `DINOV3_WEIGHTS`
  to the local file, optionally setting `DINOV3_REPO` to a local clone of the
  DINOv3 repo for torch.hub.

## Third-Party Benchmark Weights

The multi-method benchmark under `experiments/benchmark/` has a helper script:

```bash
bash experiments/benchmark/scripts/install_third_party.sh
```

It downloads Track-On2, CoTracker3, and TAPIR checkpoints into
`experiments/benchmark/checkpoints/` and clones the required third-party repos.
