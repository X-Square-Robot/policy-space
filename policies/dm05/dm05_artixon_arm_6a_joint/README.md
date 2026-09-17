# DM0.5 ArtiXon Arm-6A Joint

This independent Policy Space v2 bundle is a thin adapter to the OpenDM HTTP inference service. Only the GPU backend loads OpenDM, CUDA, the checkpoint, and `norm_stats.json`; no model-specific registration is needed in ManaEnv.

## Contract

- Input: front, left-wrist, and right-wrist HWC RGB `uint8` images; two finite 7D joint/gripper vectors; and a non-empty current task instruction.
- Backend request: images map to contiguous OpenDM slots `1`, `2`, and `3`; state is left 7D followed by right 7D; images retain their native resolution and are PNG/base64 encoded without a `data:` prefix.
- Backend output: exact finite `float32[50,14]` absolute joint targets.
- Policy Space output: exact finite `float32[50,26]`; columns 0:14 are physical actions and columns 14:26 are zero-filled record-only master-EE slots.
- Rate: 20 Hz. Grippers use normalized `[0,1]` values.

The adapter checks H50 before packing, and the generic runtime independently requires the wire horizon to equal `metadata.action_horizon`.

## Backend configuration

`backend_url` must be the full HTTP(S) `/v1/infer` endpoint. Environment variables override `deploy.yaml`:

- `DM05_X2REAL_BACKEND_URL`
- `DM05_X2REAL_CONNECT_TIMEOUT_SECONDS`
- `DM05_X2REAL_READ_TIMEOUT_SECONDS`
- `DM05_X2REAL_CHECKPOINT_ID`
- `DM05_X2REAL_SAMPLING_SEED`

Automatic POST retries and stale-action fallback are disabled. HTTP failures, timeouts, malformed JSON, wrong shapes, or non-finite values fail the current request.

```bash
export DM05_X2REAL_BACKEND_URL=http://<gpu-host>:8002/v1/infer
export DM05_X2REAL_CHECKPOINT_ID=<opendm-checkpoint-id>
policy-space check policies/dm05/dm05_artixon_arm_6a_joint/deploy.yaml
policy-space serve policies/dm05/dm05_artixon_arm_6a_joint/deploy.yaml --port 8001
```

The verified probe returned finite raw `float32[50,14]` and packed `float32[50,26]` actions from the full checkpoint.

[简体中文](README.zh-CN.md)
