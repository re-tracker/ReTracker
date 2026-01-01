# Training Experiments

This folder collects small, reproducible training commands/configs for quick
sanity checks.

## MegaDepth matching (10-step smoke)

This uses the debug config:
- `configs/train/debug/matching_megadepth.yaml`

It is configured to:
- train *matching only* (no tracking datasets),
- use the repo-local MegaDepth index under `data/megadepth/`,
- run for `max_steps=10`,
- keep query counts small for GPU memory (`fixed_*_queries_num=100`).

### 1 GPU

```bash
conda activate retracker_env

bash scripts/train/train.sh stage4 megadepth_matching_10step \
  --no_resume --no_git_sha \
  --exp_config configs/train/debug/matching_megadepth.yaml \
  --trainer.accelerator=gpu --trainer.devices=1
```

### 2 GPUs (DDP)

```bash
conda activate retracker_env

CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train/train.sh stage4 megadepth_matching_10step_ddp2 \
  --no_resume --no_git_sha \
  --exp_config configs/train/debug/matching_megadepth.yaml \
  --trainer.accelerator=gpu --trainer.devices=2
```

