# ReTracker

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **ReTracker: Exploring Image Matching for Robust Online Any Point Tracking** (ICCV 2025).

[[Paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_ReTracker_Exploring_Image_Matching_for_Robust_Online_Any_Point_Tracking_ICCV_2025_paper.html) | [[Project Page]](https://github.com/re-tracker/ReTracker)

## News

- **2025-01**: Code and pretrained models released.

## Installation

```bash
# Clone the repository
git clone https://github.com/re-tracker/ReTracker.git
cd ReTracker

# Create conda environment
conda env create -f environment.yaml
conda activate retracker_env
```

Install PyTorch (pick ONE option):

```bash
# Option A (recommended, CUDA): adjust pytorch-cuda version to match your driver (e.g. 11.8 / 12.1 / 12.4)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Option B (CPU-only)
conda install pytorch torchvision -c pytorch
```

Install ReTracker:

```bash
pip install -e .
```

For evaluation (Hydra-based):

```bash
pip install -e ".[eval]"
```

For training:

```bash
pip install -e ".[train]"
```

For runnable apps (tracking/streaming/multiview):

```bash
pip install -e ".[apps]"
```

For development and testing (includes pytest):

```bash
pip install -e ".[dev]"
```

## Pretrained Models

Download pretrained checkpoints from Google Drive:

| Model | Description | Download |
|-------|-------------|----------|
| `retracker_b1.7.ckpt` | Full model (ICCV 2025) | [Google Drive](https://drive.google.com/file/d/1BfCBAwj_jJLdI4wwforleTXEUS5NsDOq/view?usp=drive_link) |
| `retracker_light_b1.7.ckpt` | Lightweight version | [Google Drive](https://drive.google.com/file/d/1KxXcJeDw1O7FdbhVayXL_1ZKyodI0p7D/view?usp=drive_link) |

```bash
# Download using gdown
pip install gdown
mkdir -p weights
gdown 1BfCBAwj_jJLdI4wwforleTXEUS5NsDOq -O weights/retracker_b1.7.ckpt
```

See `docs/MODEL_ZOO.md` for more details.

## Quick Start

### Demo Video Tracking

```bash
# Download a checkpoint first (see above)
retracker apps tracking --demo --ckpt_path weights/retracker_b1.7.ckpt --fast_start
```

### Track Your Own Video

```bash
retracker apps tracking --video_path path/to/video.mp4 --ckpt_path weights/retracker_b1.7.ckpt --fast_start
```

### Streaming/Online Tracking

```bash
retracker apps streaming --source video_file --video_path path/to/video.mp4 --ckpt_path weights/retracker_b1.7.ckpt --fast_start --record
```

Notes:
- `--ckpt_path` is required. Place checkpoints under `weights/` (gitignored).
- `--fast_start` skips DINOv3 hub weight loading (recommended when using full ReTracker checkpoint).

## CLI

After installation, the main entry point is:

```bash
retracker --help
```

From a source checkout, you can also run:

```bash
python -m retracker --help
```

## Data and Checkpoints

Copy the local path template and edit it for your machine:

```bash
cp configs/paths_local.yaml.example configs/paths_local.yaml
```

Large assets (datasets, weights, outputs, logs) are ignored by git. Keep them
under `data/`, `weights/`, and `outputs/`, or point `configs/paths_local.yaml`
to your existing locations.

For dataset setup and licensing notes, see `docs/DATASETS.md`.

## Evaluation

```bash
# Wrapper scripts (recommended)
bash scripts/evaluation/retracker_davis_first.sh <ckpt_path> [dataset_root]

# Advanced: call the Hydra entrypoint directly
# retracker eval --config-name eval_tapvid_davis_first checkpoint=/path/to/last.ckpt dataset_root=/path/to/dataset_root
```

See `scripts/evaluation/README.md` for environment variables and wrapper usage.

## Training

```bash
# Wrapper script (recommended)
bash scripts/train/train.sh stage4 <run_name> [extra_args...]

# Advanced: call the LightningCLI directly
# retracker train fit ...
```

See `scripts/train/README.md` for output layout and more examples.

Smoke test (no external datasets required):

```bash
bash scripts/smoke/train_stage4_dummy.sh
```

## Tests

```bash
pytest tests/ -m "not slow"
```

## Repository Structure

```
ReTracker/
├── retracker/          # Core library (models, training, evaluation, utils)
│   └── apps/           # Runnable applications (tracking/streaming/multiview)
├── configs/            # Training and evaluation configs
├── scripts/            # Launchers and utilities
├── docs/               # Documentation
├── tests/              # Pytest suite
└── third_party/        # External dependencies
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@InProceedings{Tan_2025_ICCV,
    author    = {Tan, Dongli and He, Xingyi and Peng, Sida and Gong, Yiqing and Zhu, Xing and Sun, Jiaming and Hu, Ruizhen and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei},
    title     = {ReTracker: Exploring Image Matching for Robust Online Any Point Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {4306-4316}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the authors of [CoTracker](https://github.com/facebookresearch/co-tracker), [TAPIR](https://github.com/google-deepmind/tapnet), and [DINOv2](https://github.com/facebookresearch/dinov2) for their excellent work.

## Contact

For questions or issues, please open a [GitHub Issue](https://github.com/re-tracker/ReTracker/issues).
