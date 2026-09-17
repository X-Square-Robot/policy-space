# OpenPI (π0.5) ArtiXon Arm-6A EEF

This Policy Space service loads an OpenPI π0.5 checkpoint for the ArtiXon Arm-6A desktop bimanual robot. The ManaEnv wire contract remains a 14-D episode-initial-relative end-effector pose:

```text
[left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper]
```

At the OpenPI boundary, the adapter converts it to the checkpoint's 20-D representation:

```text
[left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]
```

The simulation client uses the embodiment contract to convert the Euler trajectory into executable robot commands.

## Environment

```bash
export OPENPI_ROOT=/path/to/openpi
cd "$OPENPI_ROOT"
GIT_LFS_SKIP_SMUDGE=1 uv sync
cd /path/to/policy-space
uv pip install --python "$OPENPI_ROOT"/.venv/bin/python -e .
```

The checkpoint must use 20-D bimanual EE-pose state and action statistics compatible with `pi05_x2real_lerobotv2_finetune`. If that workspace-only config is unavailable, the adapter uses the inference-equivalent `pi05_ex001_6r_joint_finetune` configuration.

The adapter replaces the generic 14-channel `AlohaOutputs` transform with a 20-channel output layer, projects rotation-6D values onto valid rotations, and returns Euler14. Startup also validates 20-D state and action quantile statistics and fails closed when transforms or statistics are incompatible.

The default X2Real checkpoint stores the first two rotation-matrix rows, so `rotation_layout: row` is the deployment default. Set `OPENPI_ROTATION_LAYOUT=column` only for a checkpoint trained with the standard first-two-columns representation.

`torch.compile(max-autotune)` is disabled by default to avoid long first-request compilation and shared-cache failures. Enable `pytorch_compile_mode` only with a suitable node-local compile cache.

## Serve

```bash
export OPENPI_ROOT=/path/to/openpi
OPENPI_CHECKPOINT_PATH=/path/to/openpi-pi05-artixon-arm-6a-checkpoint \
  "$OPENPI_ROOT"/.venv/bin/policy-space serve \
  policies/openpi/pi05_artixon_arm_6a_eef/deploy.yaml \
  --port 8001
```

Optional overrides: `OPENPI_TRAIN_CONFIG`, `OPENPI_CHECKPOINT_PATH`, `OPENPI_DEVICE`, `OPENPI_ROTATION_LAYOUT`, and `OPENPI_PYTORCH_COMPILE_MODE`.

[简体中文](README.zh-CN.md)
