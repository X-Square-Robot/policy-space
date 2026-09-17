# OpenPI (π0.5) ArtiXon Arm-6A Joint

This deployment loads the `pi05_x2real_split280` JAX checkpoint and exposes the PolicySpace `bimanual_rgb_joint@1` -> `bimanual_joint_position@1` contract. The model input and output are 14D: six arm joints plus normalized gripper per arm. OpenPI pads internally to its pretrained 32D model width.

The adapter maps the three PolicySpace cameras to OpenPI's front, left-wrist, and right-wrist image keys, concatenates `follow1_pos` and `follow2_pos`, and passes the episode instruction as the OpenPI prompt. The returned action chunk must have shape `(T, 14)` and contain only finite absolute joint targets.

## Environment

Run this integration with the OpenPI dependency stack. The deployment values can be overridden without editing the tracked YAML:

```bash
export OPENPI_ROOT=/path/to/openpi
export OPENPI_CHECKPOINT_PATH=/path/to/checkpoint
export OPENPI_TRAIN_CONFIG=pi05_x2real_split280
export OPENPI_DEVICE=cuda:0
export PYTHONPATH="$PWD/src:$PWD/packages/protocol/src:$PWD"
```

Install Policy Space into the OpenPI environment in editable mode when preparing a new host; keep OpenPI and checkpoint-specific dependencies out of the core Policy Space environment definition.

## Validate

```bash
python -m policy_space.cli check policies/openpi/pi05_artixon_arm_6a_joint/deploy.yaml
python -m pytest -q tests/test_openpi_pi05_x2real_joint_policy.py
```

## Serve

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m policy_space.cli serve policies/openpi/pi05_artixon_arm_6a_joint/deploy.yaml \
  --host 0.0.0.0 --port 8001
```

The service declares `recommended_ingest_observation_each_step: false` because each inference returns a 32-step action chunk. This allows a compatible evaluator to execute vectorized environments while the server serializes model inference.

[简体中文](README.zh-CN.md)
