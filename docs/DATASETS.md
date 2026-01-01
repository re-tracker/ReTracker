# Datasets

This repository does not distribute datasets. You are responsible for
downloading data from the original sources and complying with their licenses
and terms of use. The expected paths are configured in `configs/paths.yaml`
and can be overridden in `configs/paths_local.yaml`.

## TAP-Vid (TAP-Vid, RoboTAP, RGB-Stacking, Kinetics)

- **Project page**: https://tapvid.github.io/
- **License notes**: TAP-Vid annotations and derived videos are released under
  CC-BY 4.0; DAVIS and Kinetics source videos carry their own licenses (see the
  TAPNet repository for details).
- **Expected path**: `paths.TAPVID_ROOT` (default `data/tapvid_local` for smoke
  tests; point this to your full TAP-Vid storage for real evaluation).

## MegaDepth

- **Project page**: https://research.cs.cornell.edu/megadepth/
- **License**: CC BY 4.0 for depth/maps and SfM models; original images retain
  their own licenses.
- **Expected path**: `paths.MEGADEPTH_INDEX_ROOT` and `paths.DATA_ROOT/megadepth`.

## ScanNet

- **Project page**: https://www.scan-net.org/
- **License**: ScanNet Terms of Use (data) and MIT (code).
- **Expected path**: `paths.SCANNET_ROOT` (default `data/scannet_plus`).

## FlyingThings3D (Scene Flow Datasets)

- **Project page**: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
- **License**: research-only; commercial use prohibited (see terms of use).
- **Expected path**: under `paths.VIDEO_DATA_ROOT` (e.g., `data/flyingthings`).

## Kubric / Kubrics

- **Project page**: https://github.com/google-research/kubric
- **License**: Apache-2.0 for the generator; dataset licenses vary by release.
- **Expected path**: under `paths.VIDEO_DATA_ROOT` (e.g., `data/kubric`).

## PointOdyssey

- **Project page**: https://pointodyssey.com/
- **License**: CC BY-NC-SA 4.0 (see the official PointOdyssey repository).
- **Expected path**: under `paths.VIDEO_DATA_ROOT` (e.g., `data/pointodyssey`).

## DAVIS (TAP-Vid source videos)

- **Project page**: https://davischallenge.org/
- **License**: DAVIS videos are provided by their original creators; review the
  dataset terms on the DAVIS site.
