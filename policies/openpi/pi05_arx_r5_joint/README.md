# OpenPI (π0.5) ARX R5 Joint

This deployment loads the `pi05_arx_r5_joint` OpenPI checkpoint and exposes the PolicySpace `bimanual_rgb_joint@1` -> `bimanual_joint_position@1` contract for the `arx_r5@1` embodiment.

## Contract

- Observation: three uint8 HWC RGB cameras (`camera_front`, `camera_left`, `camera_right`) mapped to OpenPI's `observation/image`, `observation/left_wrist_image` and `observation/right_wrist_image`.
- State: `follow1_pos` and `follow2_pos`, 7D each, concatenated into the 14D `observation/state`. Left arm first.
- Action: absolute joint targets, `(T, 14)` float32 with `T` up to 32 at 20 Hz. Six arm joints plus a normalized gripper per arm; grippers use `normalized_01`.
- A non-empty instruction is required. The per-observation `instruction` wins over the episode instruction, which wins over the configured default.
- Returned chunks are validated for shape and finiteness, then truncated to `action_horizon`.

## Environment

Run this integration with the OpenPI dependency stack. The tracked YAML values can be overridden without editing the file:

```bash
export OPENPI_ROOT=/path/to/openpi
export OPENPI_CHECKPOINT_PATH=/path/to/checkpoint
export OPENPI_TRAIN_CONFIG=pi05_arx_r5_joint
export OPENPI_DEVICE=cuda:0
export PYTHONPATH="$PWD/src:$PWD/packages/protocol/src:$PWD"
```

A relative `OPENPI_CHECKPOINT_PATH` resolves against `OPENPI_ROOT`. The train config must declare `action_horizon: 32`; the loader rejects other values.

## Validate

```bash
python -m policy_space.cli check policies/openpi/pi05_arx_r5_joint/deploy.yaml
```

## Serve

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m policy_space.cli serve policies/openpi/pi05_arx_r5_joint/deploy.yaml \
  --host 0.0.0.0 --port 8001
```

Client schema selection:

```yaml
observation_schema: bimanual_rgb_joint@1
action_schema: bimanual_joint_position@1
embodiment_schema: arx_r5@1
```

The service declares `recommended_ingest_observation_each_step: false` and `recommended_connect_chunks: true` because each inference returns a 32-step chunk. A handshake and a finite action chunk verify the integration, not task success; rollout evaluation is still needed to measure the policy.

[简体中文](README.zh-CN.md)
