# OpenPI (π0.5) ARX R5 EEF

This integration loads a JAX π0.5 checkpoint for the ARX R5 end-effector contract. Set `OPENPI_CHECKPOINT_PATH` to the checkpoint root.

## Contract and training parity

- Embodiment: `arx_r5@1`.
- Observation: `bimanual_rgb_ee@1`; three uint8 HWC RGB cameras.
- Action: `bimanual_ee_absolute@1`, exactly 32 x 14 float32 values at 20 Hz.
- Each arm is `[x, y, z, roll, pitch, yaw, gripper]`; left arm first.
- Poses are targets in the episode-initial end-effector frame, in meters and XYZ Euler radians. They are not per-timestep deltas or world-frame targets.
- Grippers use normalized opening: 0 closed, 1 open. The client converts to the selected ARX embodiment's native gripper units; this adapter never multiplies by the source scale of 4.5.
- Camera roles front / left / right map to faceImg / leftImg / rightImg. All three model image masks, including the right wrist, are enabled.
- Use checkpoint quantile normalization, the pi0.5 state tokenizer and 32D internal padding. After unnormalization, drop the padded dimensions.
- XYZ/RPY predictions pass through without conversion or a current-state echo. Only predicted gripper channels are clipped to [0,1].
- Execute the 32 returned rows directly. No extra interpolation or smoothing is performed. A non-empty task instruction is required.
- The service uses an exclusive session and clears observation/instruction state at episode boundaries.

`model.py` builds a self-contained inference config. It does not import the external ARX training launcher, mutate OpenPI's config registry, or open the training dataset. The configured OpenPI environment still supplies the model, tokenizer and checkpoint reader.

## Environment and checkpoint

Point `OPENPI_ROOT` / `openpi_root` at a local OpenPI checkout, and `OPENPI_CHECKPOINT_PATH` / `checkpoint_path` at the checkpoint root (not its `params/` child). Override the interpreter with `POLICY_SPACE_PYTHON`. The loader verifies a completed checkpoint commit and 14D state/action normalization statistics. No base checkpoint or optimizer state is needed for inference.

The existing offline PaliGemma tokenizer cache must remain available.

Reference training environment: JAX/JAXlib 0.5.3, Flax 0.10.2, Orbax 0.11.13, NumPy 1.26.4, Transformers 4.53.2.

## Validate and serve

From the Policy Space project root:

```bash
export CUDA_VISIBLE_DEVICES=<available-gpu-index>
export OPENPI_ROOT=/path/to/openpi
export OPENPI_CHECKPOINT_PATH=/path/to/checkpoint
bash policies/openpi/pi05_arx_r5_eef/serve.sh --check
bash policies/openpi/pi05_arx_r5_eef/serve.sh --host 0.0.0.0 --port 8001
```

The first command runs `python -m policy_space.cli check` through a complete model lifecycle. It performs real checkpoint inference on a synthetic fixture. The second starts the Policy Space v2 WebSocket service. The launcher disables JAX preallocation and defaults the memory fraction to 0.25. Select GPU resources after checking live usage.

Client schema selection:

```yaml
observation_schema: bimanual_rgb_ee@1
action_schema: bimanual_ee_absolute@1
embodiment_schema: arx_r5@1
```

Connect to `ws://127.0.0.1:8001`. For a client on another machine, open an SSH tunnel and use the same URL:

```bash
ssh -N -L 8001:127.0.0.1:8001 my-server
```

Use `--port` to select another available port if needed.

Run the regression tests with the project source directories on PYTHONPATH:

```bash
python -m pytest -q tests/test_openpi_pi05_arx_ee_policy.py
```

A handshake and finite action chunk verify integration, not task success. Rollout evaluation is still needed to measure the trained policy.

[简体中文](README.zh-CN.md)
